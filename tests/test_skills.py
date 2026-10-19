# Copyright 2026 Juan Pablo de la Cruz Gutiérrez
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from IPython.core.error import UsageError
from IPython.core.interactiveshell import InteractiveShell

from ipyllm.magics import IpyllmMagics
from ipyllm.skills import discover_skills


def write_skill(root: Path, name: str, body: str = "Use pandas effectively.") -> Path:
    skill_dir = root / name
    skill_dir.mkdir(parents=True)
    skill_path = skill_dir / "SKILL.md"
    skill_path.write_text(
        f"---\nname: {name}\ndescription: Helps with {name}.\n---\n\n{body}\n",
        encoding="utf-8",
    )
    return skill_path


class SkillDiscoveryTests(unittest.TestCase):
    def test_discovers_metadata_without_loading_instructions(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            skill_path = write_skill(root, "data-analysis", "x" * 40_000)

            catalog = discover_skills(
                [str(root)],
                project_dir=root / "project",
                user_dir=root / "user",
            )

        self.assertEqual(
            catalog.skills["data-analysis"].description,
            "Helps with data-analysis.",
        )
        self.assertEqual(catalog.skills["data-analysis"].path, skill_path)
        self.assertFalse(catalog.errors)

    def test_reports_malformed_and_duplicate_skills(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            first = root / "first"
            second = root / "second"
            first_path = write_skill(first, "plotting")
            second_path = write_skill(second, "plotting")
            malformed = first / "broken" / "SKILL.md"
            malformed.parent.mkdir()
            malformed.write_text("not frontmatter", encoding="utf-8")

            catalog = discover_skills(
                [str(first), str(second)],
                project_dir=root / "project",
                user_dir=root / "user",
            )

        self.assertNotIn("plotting", catalog.skills)
        self.assertEqual(set(catalog.duplicates["plotting"]), {first_path, second_path})
        self.assertTrue(any(str(malformed) in error for error in catalog.errors))

    def test_reports_configured_missing_directory(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            missing = root / "missing"

            catalog = discover_skills(
                [str(missing)],
                project_dir=root / "project",
                user_dir=root / "user",
            )

        self.assertEqual(
            catalog.errors,
            [f"Configured skill directory does not exist: {missing}"],
        )


class SkillInvocationTests(unittest.TestCase):
    def setUp(self):
        self.shell = InteractiveShell.instance()
        self.magics = IpyllmMagics(self.shell)
        self.output = StringIO()
        self.default_paths_patcher = patch(
            "ipyllm.skills.default_skill_directories", return_value=[]
        )
        self.default_paths_patcher.start()
        self.addCleanup(self.default_paths_patcher.stop)

    def test_explicit_skill_is_injected_as_per_turn_context(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            write_skill(root, "pandas", "Prefer vectorized pandas operations.")
            self.magics.skill_paths = [str(root)]
            self.magics._conversation = FakeConversation()

            with redirect_stdout(self.output):
                self.magics.llm("/pandas Plot my DataFrame")

        conversation = self.magics._conversation
        self.assertIn("Plot my DataFrame", conversation.prompt)
        self.assertNotIn("/pandas", conversation.prompt)
        self.assertEqual(len(conversation.fragments), 1)
        self.assertIn("Prefer vectorized pandas operations.", conversation.fragments[0])
        self.assertIn("Selected skill: pandas", conversation.fragments[0])

    def test_cell_prompt_can_select_a_skill(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            write_skill(root, "seaborn", "Use seaborn themes.")
            self.magics.skill_paths = [str(root)]
            self.magics._conversation = FakeConversation()

            with redirect_stdout(self.output):
                self.magics.llm("/seaborn", "Plot distributions.")

        self.assertIn("Plot distributions.", self.magics._conversation.prompt)
        self.assertIn("Use seaborn themes.", self.magics._conversation.fragments[0])

    def test_prompts_without_selector_have_no_skill_fragment(self):
        self.magics._conversation = FakeConversation()

        with redirect_stdout(self.output):
            self.magics.llm("Explain this expression.")

        self.assertIsNone(self.magics._conversation.fragments)

    def test_unknown_skill_and_missing_request_are_clear_errors(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            write_skill(root, "numpy")
            self.magics.skill_paths = [str(root)]

            with self.assertRaisesRegex(UsageError, "Unknown skill /pandas"):
                self.magics._select_skill("/pandas compute the mean")
            with self.assertRaisesRegex(UsageError, "Add a prompt after"):
                self.magics._select_skill("/numpy")

    def test_duplicate_skill_invocation_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            write_skill(root / "first", "numpy")
            write_skill(root / "second", "numpy")
            self.magics.skill_paths = [str(root / "first"), str(root / "second")]

            with self.assertRaisesRegex(UsageError, "Skill /numpy is ambiguous"):
                self.magics._select_skill("/numpy compute the mean")

    def test_skill_size_limit_is_enforced_when_invoked(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            write_skill(root, "large", "x" * 33_000)
            self.magics.skill_paths = [str(root)]
            self.magics._conversation = FakeConversation()

            with self.assertRaisesRegex(UsageError, "character limit"):
                self.magics._select_skill("/large summarize")

    def test_skill_listing_shows_descriptions_and_invalid_files(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            write_skill(root, "matplotlib", "Plot with matplotlib.")
            broken = root / "broken" / "SKILL.md"
            broken.parent.mkdir()
            broken.write_text("bad metadata", encoding="utf-8")
            self.magics.skill_paths = [str(root)]

            with redirect_stdout(self.output):
                self.magics._print_skills()

        output = self.output.getvalue()
        self.assertIn("/matplotlib - Helps with matplotlib.", output)
        self.assertIn(str(broken), output)


class FakeConversation:
    def __init__(self):
        self.prompt = None
        self.fragments = None

    def chain(self, prompt, *, system, fragments=None):
        self.prompt = prompt
        self.fragments = fragments
        return FakeResponse()


class FakeResponse:
    def text(self):
        return "answer"


if __name__ == "__main__":
    unittest.main()
