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

import subprocess
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import llm
from IPython.core.error import UsageError
from IPython.core.interactiveshell import InteractiveShell

from ipyllm import load_ipython_extension, unload_ipython_extension
from ipyllm.magics import IpyllmMagics


class FakeResponse:
    def text(self):
        return "model response"


class FakeConversation:
    def __init__(self, **kwargs):
        self.options = kwargs
        self.prompt = None
        self.system = None

    def chain(self, prompt, *, system, fragments=None):
        self.prompt = prompt
        self.system = system
        self.fragments = fragments
        return FakeResponse()


class FakeModel:
    def conversation(self, **kwargs):
        self.created_conversation = FakeConversation(**kwargs)
        return self.created_conversation


class IpyllmMagicsTests(unittest.TestCase):
    def setUp(self):
        self.shell = InteractiveShell.instance()
        self.shell.user_ns["visible_test_value"] = [1, 2, 3]
        self.shell.user_ns["_hidden_test_value"] = "secret"
        self.magics = IpyllmMagics(self.shell)
        self.fake_model = FakeModel()
        self.model_patcher = patch(
            "ipyllm.magics.llm.get_model", return_value=self.fake_model
        )
        self.get_model = self.model_patcher.start()
        self.addCleanup(self.model_patcher.stop)

    def tearDown(self):
        self.shell.user_ns.pop("visible_test_value", None)
        self.shell.user_ns.pop("_hidden_test_value", None)

    def test_extension_registers_magic(self):
        load_ipython_extension(self.shell)
        self.assertIsNotNone(self.shell.find_line_magic("llm"))
        unload_ipython_extension(self.shell)
        self.assertIsNone(self.shell.find_line_magic("llm"))

    def test_prompt_includes_only_namespace_names_and_types(self):
        self.magics.llm("Tell me about visible_test_value")

        prompt = self.fake_model.created_conversation.prompt
        self.assertIn("visible_test_value: list", prompt)
        self.assertNotIn("[1, 2, 3]", prompt)
        self.assertNotIn("_hidden_test_value", prompt)
        self.assertIn("Tell me about visible_test_value", prompt)
        tools = self.fake_model.created_conversation.options["tools"]
        self.assertEqual([tool.__name__ for tool in tools][-1], "execute_code")
        for tool in tools:
            llm.Tool.function(tool)

    def test_model_request_executes_in_live_namespace(self):
        response = self.magics.execute_code("visible_test_value.append(4)")

        self.assertEqual(self.shell.user_ns["visible_test_value"], [1, 2, 3, 4])
        self.assertEqual(response, "Execution completed successfully.")

    def test_execution_errors_are_returned_to_the_model(self):
        response = self.magics.execute_code("raise ValueError('broken')")

        self.assertIn("Execution failed: ValueError: broken", response)

    def test_variable_inspection_is_bounded(self):
        class LargeRepresentation:
            def __repr__(self):
                return "x" * 10_000

        self.shell.user_ns["large_test_value"] = LargeRepresentation()
        self.addCleanup(self.shell.user_ns.pop, "large_test_value", None)

        response = self.magics.inspect_variable("large_test_value")
        self.assertEqual(
            len(response),
            4_000
            + len("large_test_value: LargeRepresentation\n")
            + len("... (truncated)"),
        )

    def test_history_range_is_read_and_bounded(self):
        history = self.shell.history_manager
        entries = [(1, number, f"value = {number}") for number in range(40)]
        with patch.object(
            history, "get_range_by_str", return_value=iter(entries)
        ) as get_range:
            response = self.magics.get_history("1-40")

        get_range.assert_called_once_with("1-40", raw=True)
        self.assertIn("In [1:0]:", response)
        self.assertNotIn("In [1:30]:", response)

    def test_approval_mode_requires_exact_yes(self):
        self.magics.require_approval = True
        call = SimpleNamespace(name="execute_code", arguments={"code": "x = 1"})

        with patch("builtins.input", return_value="no"):
            with self.assertRaises(llm.CancelToolCall):
                self.magics._before_tool_call(None, call)
        with patch("builtins.input", return_value="yes"):
            self.magics._before_tool_call(None, call)

    def test_approval_can_be_overridden_and_commands_validate(self):
        self.magics.require_approval = True
        self.magics._handle_command("--approval off")
        self.assertFalse(self.magics._current_approval_policy())

        with self.assertRaises(UsageError):
            self.magics._handle_command("--approval sometimes")

    def test_help_command_does_not_send_a_prompt(self):
        self.assertTrue(self.magics._handle_command("--help"))
        self.assertIsNone(self.magics._conversation)

    def test_save_records_requests_as_comments_and_executed_code(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "session_file.py"
            self.magics._enable_session_save(str(path))
            self.magics._record_request("Plot `data`.\nUse two lines.")
            self.magics._record_executed_code("result = len(data)\nprint(result)")

            content = path.read_text(encoding="utf-8")

        self.assertIn("# Request:\n# Plot `data`.\n# Use two lines.\n\n", content)
        self.assertIn("result = len(data)\nprint(result)\n", content)
        self.assertLess(content.index("# Request:"), content.index("result = len(data)"))
        compile(content, str(path), "exec")

    def test_save_command_refuses_existing_file_and_can_be_disabled(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "session.py"
            path.write_text("existing = True\n", encoding="utf-8")

            with self.assertRaisesRegex(UsageError, "Refusing to overwrite"):
                self.magics._handle_command(f"--save {path}")
            self.assertEqual(path.read_text(encoding="utf-8"), "existing = True\n")

            new_path = Path(temporary_directory) / "new_session.py"
            self.magics._handle_command(f"--save {new_path}")
            self.assertEqual(self.magics._save_path, new_path)
            self.magics._handle_command("--save off")
            self.assertIsNone(self.magics._save_path)

    def test_ipython_syntax_is_saved_as_comments(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "session.py"
            self.magics._enable_session_save(str(path))
            with redirect_stdout(StringIO()):
                self.magics._record_executed_code("%time print('hello')")
            content = path.read_text(encoding="utf-8")

        self.assertIn("# %time print('hello')", content)
        self.assertIn("IPython-only input", content)
        compile(content, str(path), "exec")

    def test_execute_code_records_code_before_execution(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "session.py"
            self.magics._enable_session_save(str(path))
            self.magics._record_request("Set up the result.")

            response = self.magics.execute_code("saved_value = 42")

            content = path.read_text(encoding="utf-8")

        self.assertEqual(response, "Execution completed successfully.")
        self.assertIn("saved_value = 42", content)
        self.assertLess(content.index("# Set up the result."), content.index("saved_value = 42"))
        compile(content, str(path), "exec")
        self.assertEqual(self.shell.user_ns["saved_value"], 42)
        self.shell.user_ns.pop("saved_value", None)

    def test_reset_discards_conversation(self):
        self.magics._get_conversation()
        self.assertIsNotNone(self.magics._conversation)

        self.magics.reset()

        self.assertIsNone(self.magics._conversation)

    def test_run_cell_captures_command_output_for_diagnosis(self):
        conversation = FakeConversation()
        self.magics._conversation = conversation
        completed = subprocess.CompletedProcess(
            args="pytest -q",
            returncode=1,
            stdout="2 failed\n",
            stderr="ImportError: missing package\n",
        )
        output = StringIO()
        with patch("ipyllm.magics.subprocess.run", return_value=completed) as run:
            with redirect_stdout(output):
                self.magics.llm(
                    "--run Explain the failure",
                    "pytest -q",
                )

        run.assert_called_once_with(
            "pytest -q",
            shell=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
            check=False,
        )
        self.assertIn("stdout:\n2 failed", output.getvalue())
        self.assertIn("stderr:\nImportError: missing package", output.getvalue())
        self.assertIn("Command:\npytest -q", conversation.prompt)
        self.assertIn("exit status: 1", conversation.prompt)
        self.assertIn("Explain the failure", conversation.prompt)
        self.assertIn("ImportError: missing package", conversation.prompt)

    def test_run_cell_handles_timeouts_and_bounds_output_context(self):
        conversation = FakeConversation()
        self.magics._conversation = conversation
        timeout = subprocess.TimeoutExpired(
            cmd="slow-command",
            timeout=120,
            output=b"partial stdout",
            stderr=b"partial stderr",
        )
        output = StringIO()
        with patch("ipyllm.magics.subprocess.run", side_effect=timeout):
            with redirect_stdout(output):
                self.magics.llm("--run", "slow-command")

        self.assertIn("timed out after 120 seconds", conversation.prompt)
        self.assertIn("partial stdout", conversation.prompt)
        self.assertIn("partial stderr", conversation.prompt)

        large_output = "first" + ("x" * 20_000) + "last"
        truncated = self.magics._truncate_command_output(large_output)
        self.assertLessEqual(len(truncated), 12_100)
        self.assertIn("characters omitted", truncated)
        self.assertTrue(truncated.startswith("first"))
        self.assertTrue(truncated.endswith("last"))

    def test_run_cell_requires_a_command_body(self):
        with self.assertRaisesRegex(UsageError, "shell command"):
            self.magics.llm("--run", " \n")

    def test_registered_cell_magic_runs_diagnostic_syntax(self):
        load_ipython_extension(self.shell)
        magics = self.shell.magics_manager.registry["IpyllmMagics"]
        conversation = FakeConversation()
        magics._conversation = conversation
        completed = subprocess.CompletedProcess(
            args="python -V",
            returncode=0,
            stdout="Python 3\n",
            stderr="",
        )
        with patch("ipyllm.magics.subprocess.run", return_value=completed):
            result = self.shell.run_cell(
                "%%llm --run Explain this output.\npython -V",
                store_history=False,
                silent=True,
            )
        unload_ipython_extension(self.shell)

        self.assertTrue(result.success)
        self.assertIn("Python 3", conversation.prompt)
        self.assertIn("Explain this output.", conversation.prompt)


if __name__ == "__main__":
    unittest.main()
