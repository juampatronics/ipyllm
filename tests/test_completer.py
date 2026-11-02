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

import os
import tempfile
import unittest
from pathlib import Path

from IPython.core.interactiveshell import InteractiveShell

from ipyllm import load_ipython_extension, unload_ipython_extension
from ipyllm.completer import at_file_matcher, install_completer, uninstall_completer


class FakeContext:
    """Minimal stand-in for IPython's ``CompletionContext`` used by the matcher."""

    def __init__(self, token: str, text_until_cursor: str):
        self.token = token
        self.text_until_cursor = text_until_cursor


def texts(result: dict) -> list[str]:
    return [getattr(completion, "text", completion) for completion in result["completions"]]


class AtFileMatcherTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        (self.root / "alpha.py").write_text("alpha = 1\n", encoding="utf-8")
        (self.root / "beta.txt").write_text("beta", encoding="utf-8")
        self.sub = self.root / "sub"
        self.sub.mkdir()
        (self.sub / "gamma.py").write_text("gamma = 3\n", encoding="utf-8")
        self._old_cwd = os.getcwd()
        os.chdir(self.root)
        self.addCleanup(os.chdir, self._old_cwd)

    def tearDown(self):
        self._tmp.cleanup()

    def test_lists_files_and_directories(self):
        result = at_file_matcher(FakeContext("", "prefix @"))
        self.assertEqual(texts(result), ["alpha.py", "beta.txt", "sub/"])
        self.assertTrue(result["suppress"])

    def test_directories_are_marked_with_trailing_slash(self):
        result = at_file_matcher(FakeContext("sub", "prefix @sub"))
        self.assertEqual(texts(result), ["sub/"])

    def test_filters_by_prefix(self):
        result = at_file_matcher(FakeContext("al", "prefix @al"))
        self.assertEqual(texts(result), ["alpha.py"])

    def test_completes_into_subdirectory(self):
        result = at_file_matcher(FakeContext("sub/", "prefix @sub/"))
        self.assertEqual(texts(result), ["gamma.py"])

    def test_prefix_matching_within_subdirectory(self):
        result = at_file_matcher(FakeContext("sub/ga", "prefix @sub/ga"))
        self.assertEqual(texts(result), ["gamma.py"])

    def test_does_not_trigger_for_in_word_at(self):
        result = at_file_matcher(FakeContext("", "email user@"))
        self.assertEqual(texts(result), [])
        self.assertFalse(result["suppress"])

    def test_does_not_trigger_without_at(self):
        result = at_file_matcher(FakeContext("alpha", "prefix alpha"))
        self.assertEqual(texts(result), [])

    def test_suppresses_competing_matchers_only_when_matching(self):
        self.assertTrue(at_file_matcher(FakeContext("a", "x @a"))["suppress"])
        self.assertFalse(at_file_matcher(FakeContext("", "email user@"))["suppress"])

    def test_empty_directory_yields_no_match(self):
        empty = self.root / "empty"
        empty.mkdir()
        result = at_file_matcher(FakeContext("empty/", "x @empty/"))
        self.assertEqual(texts(result), [])
        self.assertFalse(result["suppress"])

    def test_unreadable_directory_yields_no_match(self):
        blocked = self.root / "blocked"
        blocked.mkdir()
        os.chmod(blocked, 0)
        try:
            result = at_file_matcher(FakeContext("blocked/", "x @blocked/"))
            self.assertEqual(texts(result), [])
        finally:
            os.chmod(blocked, 0o755)


class CompleterInstallTests(unittest.TestCase):
    def test_install_and_uninstall_toggle_the_matcher(self):
        shell = InteractiveShell.instance()
        install_completer(shell)
        install_completer(shell)  # idempotent

        self.assertEqual(
            shell.Completer.custom_matchers.count(at_file_matcher), 1
        )

        uninstall_completer(shell)
        uninstall_completer(shell)  # safe when absent

        self.assertNotIn(at_file_matcher, shell.Completer.custom_matchers)

    def test_extension_load_installs_and_unload_removes_the_matcher(self):
        shell = InteractiveShell.instance()
        self.assertNotIn(at_file_matcher, shell.Completer.custom_matchers)
        self.addCleanup(unload_ipython_extension, shell)

        load_ipython_extension(shell)
        self.assertIn(at_file_matcher, shell.Completer.custom_matchers)


if __name__ == "__main__":
    unittest.main()
