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
import subprocess
import sys
import tempfile
import unittest
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import llm
from IPython.core.error import UsageError
from IPython.core.interactiveshell import InteractiveShell

from ipyllm import load_ipython_extension, unload_ipython_extension
from ipyllm.diagnostics import _truncate_command_output
from ipyllm.magics import IpyllmMagics, _display_response, _display_tool_result
from ipyllm.recorder import _read_saved_requests


class FakeResponse:
    def __init__(self, text="model response", *, input=0, output=0):
        self._text = text
        self._input = input
        self._output = output

    def text(self):
        return self._text

    def stream_events(self):
        yield SimpleNamespace(type="text", chunk=self._text)

    def usage(self):
        return SimpleNamespace(input=self._input, output=self._output)


class FakeChainResponse:
    def __init__(self, responses):
        self._responses = responses

    def responses(self):
        return iter(self._responses)

    def stream_events(self):
        for response in self._responses:
            yield from response.stream_events()

    def usage(self):
        return SimpleNamespace(
            input=sum(r.usage().input for r in self._responses),
            output=sum(r.usage().output for r in self._responses),
        )


class FakeConversation:
    def __init__(self, **kwargs):
        self.options = kwargs
        self.prompt = None
        self.system = None
        self.model = SimpleNamespace(model_id="fake-model")
        self.next_tokens = (0, 0)  # (input, output) for the next chain() call

    def chain(self, prompt, *, system, fragments=None, attachments=None, chain_limit=None):
        self.prompt = prompt
        self.system = system
        self.fragments = fragments
        self.attachments = attachments
        self.chain_limit = chain_limit
        input_tokens, output_tokens = self.next_tokens
        return FakeResponse(input=input_tokens, output=output_tokens)


class FakeModel:
    def conversation(self, **kwargs):
        self.created_conversation = FakeConversation(**kwargs)
        return self.created_conversation


@contextmanager
def _color_state(on: bool):
    """Force deterministic color detection by controlling NO_COLOR/FORCE_COLOR."""
    saved = {
        key: os.environ.get(key)
        for key in ("NO_COLOR", "FORCE_COLOR")
        if key in os.environ
    }
    os.environ.pop("NO_COLOR", None)
    os.environ.pop("FORCE_COLOR", None)
    if on:
        os.environ["FORCE_COLOR"] = "1"
    else:
        os.environ["NO_COLOR"] = "1"
    try:
        yield
    finally:
        os.environ.pop("NO_COLOR", None)
        os.environ.pop("FORCE_COLOR", None)
        for key, value in saved.items():
            if value is not None:
                os.environ[key] = value


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

    def test_successful_execution_output_remains_visible(self):
        stdout = StringIO()
        with redirect_stdout(stdout):
            self.magics.execute_code("print('visible output')")

        self.assertIn("visible output", stdout.getvalue())

    def test_execution_errors_are_returned_to_the_model(self):
        stdout = StringIO()
        stderr = StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            response = self.magics.execute_code("raise ValueError('broken')")

        self.assertIn("Execution failed: ValueError: broken", response)
        self.assertNotIn("ValueError", stdout.getvalue())
        self.assertNotIn("ValueError", stderr.getvalue())

    def test_debug_mode_displays_execution_tracebacks(self):
        self.magics.debug = True
        stdout = StringIO()
        stderr = StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            response = self.magics.execute_code("raise ValueError('broken')")

        self.assertIn("Execution failed: ValueError: broken", response)
        output = stdout.getvalue() + stderr.getvalue()
        self.assertIn("Traceback", output)
        self.assertIn("broken", output)

    def test_debug_command_can_be_toggled_and_validates_values(self):
        self.assertFalse(self.magics.debug)
        self.magics._handle_command("--debug on")
        self.assertTrue(self.magics.debug)
        self.magics._handle_command("--debug off")
        self.assertFalse(self.magics.debug)

        with self.assertRaises(UsageError):
            self.magics._handle_command("--debug sometimes")

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

    def test_chain_limit_defaults_to_ten_and_is_passed_to_model(self):
        self.magics.llm("Do a small task.")
        conversation = self.fake_model.created_conversation
        self.assertEqual(conversation.chain_limit, 10)

        self.magics._handle_command("--chain-limit 25")
        self.magics.llm("Continue with a larger task.")
        self.assertEqual(conversation.chain_limit, 25)

    def test_chain_limit_must_be_a_positive_integer(self):
        for value in ("0", "-2", "many"):
            with self.subTest(value=value), self.assertRaisesRegex(
                UsageError, "positive integer"
            ):
                self.magics._handle_command(f"--chain-limit {value}")

    def test_help_command_does_not_send_a_prompt(self):
        self.assertTrue(self.magics._handle_command("--help"))
        self.assertIsNone(self.magics._conversation)

    def test_response_display_dims_reasoning_not_answer(self):
        class ReasoningResponse:
            def stream_events(self):
                yield SimpleNamespace(type="reasoning", chunk="thinking about it")
                yield SimpleNamespace(type="text", chunk="the final answer")

        stdout = StringIO()
        with redirect_stdout(stdout), _color_state(True):
            _display_response(ReasoningResponse())

        output = stdout.getvalue()
        self.assertIn("\x1b[2mthinking about it\x1b[0m", output)
        self.assertIn("the final answer", output)
        self.assertNotIn("\x1b[2mthe final answer", output)

    def test_response_display_plain_when_color_disabled(self):
        class PlainResponse:
            def stream_events(self):
                yield SimpleNamespace(type="reasoning", chunk="thinking")
                yield SimpleNamespace(type="text", chunk="the answer")

        stdout = StringIO()
        with redirect_stdout(stdout), _color_state(False):
            _display_response(PlainResponse())

        output = stdout.getvalue()
        self.assertNotIn("\x1b[2m", output)
        self.assertIn("thinking", output)
        self.assertIn("the answer", output)

    def test_display_response_returns_single_round_usage(self):
        response = FakeResponse(text="done", input=12, output=4)

        stdout = StringIO()
        with redirect_stdout(stdout), _color_state(False):
            result = _display_response(response)

        self.assertEqual(result, (12, 4))

    def test_display_response_sums_all_chain_rounds(self):
        first = FakeResponse(text="Let me check.", input=10, output=5)
        second = FakeResponse(text="It is done.", input=7, output=3)
        chain = FakeChainResponse([first, second])

        stdout = StringIO()
        with redirect_stdout(stdout), _color_state(False):
            result = _display_response(chain)

        self.assertEqual(result, (17, 8))
        output = stdout.getvalue()
        self.assertIn("Let me check.", output)
        self.assertIn("It is done.", output)
        # The chain's round-boundary space keeps the two answers apart.
        self.assertIn("Let me check. It is done.", output)

    def test_display_response_without_usage_reports_zero(self):
        class NoUsageResponse:
            def stream_events(self):
                yield SimpleNamespace(type="text", chunk="answer")

        stdout = StringIO()
        with redirect_stdout(stdout), _color_state(False):
            result = _display_response(NoUsageResponse())

        self.assertEqual(result, (0, 0))
        self.assertIn("answer", stdout.getvalue())

    def test_ask_records_and_accumulates_tokens_per_model(self):
        conversation = self.magics._get_conversation()
        conversation.next_tokens = (10, 5)
        stdout = StringIO()
        with redirect_stdout(stdout):
            self.magics.llm("first prompt")
        output = stdout.getvalue()
        self.assertIn("last prompt: 15 (10 in, 5 out)", output)
        self.assertIn("session: 15 (10 in, 5 out)", output)
        self.assertIn("model: fake-model", output)
        self.assertEqual(
            self.magics._token_usage["fake-model"],
            {"in": 10, "out": 5, "prompts": 1},
        )

        conversation.next_tokens = (7, 3)
        stdout = StringIO()
        with redirect_stdout(stdout):
            self.magics.llm("second prompt")
        output = stdout.getvalue()
        self.assertIn("last prompt: 10 (7 in, 3 out)", output)
        self.assertIn("session: 25 (17 in, 8 out)", output)
        self.assertEqual(
            self.magics._token_usage["fake-model"],
            {"in": 17, "out": 8, "prompts": 2},
        )

    def test_tokens_command_lists_per_model_totals(self):
        conversation = self.magics._get_conversation()
        conversation.next_tokens = (10, 5)
        with redirect_stdout(StringIO()):
            self.magics.llm("first prompt")

        stdout = StringIO()
        with redirect_stdout(stdout):
            self.magics._handle_command("--tokens")
        output = stdout.getvalue()
        self.assertIn("fake-model", output)
        self.assertIn("15 tokens (10 in, 5 out)", output)
        self.assertIn("over 1 prompt(s)", output)

    def test_tokens_command_empty_before_any_prompt(self):
        stdout = StringIO()
        with redirect_stdout(stdout):
            self.magics._handle_command("--tokens")
        self.assertIn("No token usage recorded yet.", stdout.getvalue())

    def test_tool_result_is_shown_dim(self):
        stdout = StringIO()
        tool_call = SimpleNamespace(name="execute_code")
        tool_result = SimpleNamespace(output="Execution completed successfully.")
        with redirect_stdout(stdout), _color_state(True):
            _display_tool_result(None, tool_call, tool_result)

        output = stdout.getvalue()
        self.assertIn("execute_code", output)
        self.assertIn("Execution completed successfully.", output)
        self.assertIn("\x1b[2m", output)

    def test_tool_result_long_output_is_truncated(self):
        stdout = StringIO()
        tool_call = SimpleNamespace(name="get_history")
        tool_result = SimpleNamespace(output="x" * 1000)
        with redirect_stdout(stdout), _color_state(False):
            _display_tool_result(None, tool_call, tool_result)

        output = stdout.getvalue()
        self.assertIn("…", output)
        self.assertLess(output.count("x"), 1000)

    def test_save_records_requests_as_comments_and_executed_code(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "session_file.py"
            self.magics._recorder.enable(str(path))
            self.magics._recorder.record_request("Plot `data`.\nUse two lines.")
            self.magics._recorder.record_executed_code("result = len(data)\nprint(result)")

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
            self.assertEqual(self.magics._recorder.save_path, new_path)
            self.magics._handle_command("--save off")
            self.assertIsNone(self.magics._recorder.save_path)

    def test_resume_replays_script_and_appends_new_activity(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "saved session.py"
            missing = object()
            previous_file = self.shell.user_ns.get("__file__", missing)
            if previous_file is missing:
                self.addCleanup(self.shell.user_ns.pop, "__file__", None)
            else:
                self.addCleanup(
                    self.shell.user_ns.__setitem__, "__file__", previous_file
                )
            path.write_text(
                "# Request:\n"
                "# Create the baseline value.\n\n"
                "baseline_value = 41\n",
                encoding="utf-8",
            )
            output = StringIO()
            with redirect_stdout(output):
                self.magics._handle_command(f'--resume "{path}"')

            self.assertEqual(self.shell.user_ns["baseline_value"], 41)
            self.assertEqual(self.magics._recorder.save_path, path)
            self.assertEqual(
                self.magics._resume_requests, ["Create the baseline value."]
            )
            self.assertEqual(self.shell.user_ns["__file__"], str(path))

            self.magics._conversation = FakeConversation()
            with redirect_stdout(output):
                self.magics.llm("Add one to the baseline.")
            prompt = self.magics._conversation.prompt
            updated_source = path.read_text(encoding="utf-8")

        self.assertIn("Create the baseline value.", prompt)
        self.assertIn("previous model responses were not saved", prompt)
        self.assertIn("baseline_value = 41", updated_source)
        self.assertIn("# Add one to the baseline.", updated_source)
        self.assertEqual(self.magics._resume_requests, [])
        self.shell.user_ns.pop("baseline_value", None)

    def test_resume_requires_valid_existing_python_and_no_active_save(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            invalid = root / "invalid.py"
            invalid.write_text("if True\n", encoding="utf-8")
            missing = root / "missing.py"

            with self.assertRaisesRegex(UsageError, "Cannot read session file"):
                self.magics._recorder.resume(str(missing))
            with self.assertRaisesRegex(UsageError, "not valid Python"):
                self.magics._recorder.resume(str(invalid))
            with self.assertRaisesRegex(UsageError, "Stop recording"):
                self.magics._recorder.enable(str(root / "active.py"))
                self.magics._recorder.resume(str(invalid))

    def test_resume_execution_failure_does_not_enable_recording(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "fails.py"
            path.write_text(
                "partial_resume_value = 1\n"
                "raise RuntimeError('resume failed')\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(RuntimeError, "resume failed"):
                self.magics._recorder.resume(str(path))

        self.assertIsNone(self.magics._recorder.save_path)
        self.assertNotIn("partial_resume_value", self.magics._resume_requests)
        self.assertEqual(self.shell.user_ns["partial_resume_value"], 1)
        self.shell.user_ns.pop("partial_resume_value", None)

    def test_read_saved_requests_preserves_multiline_prompts(self):
        source = (
            "# Request:\n"
            "# First line\n"
            "#\n"
            "# Third line\n\n"
            "value = 1\n\n"
            "# Request:\n"
            "# Second request\n\n"
        )

        self.assertEqual(
            _read_saved_requests(source),
            ["First line\n\nThird line", "Second request"],
        )

    def test_ipython_syntax_is_saved_as_comments(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "session.py"
            self.magics._recorder.enable(str(path))
            with redirect_stdout(StringIO()):
                self.magics._recorder.record_executed_code("%time print('hello')")
            content = path.read_text(encoding="utf-8")

        self.assertIn("# %time print('hello')", content)
        self.assertIn("IPython-only input", content)
        compile(content, str(path), "exec")

    def test_execute_code_records_code_before_execution(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "session.py"
            self.magics._recorder.enable(str(path))
            self.magics._recorder.record_request("Set up the result.")

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
        with patch("ipyllm.diagnostics.subprocess.run", return_value=completed) as run:
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
        with patch("ipyllm.diagnostics.subprocess.run", side_effect=timeout):
            with redirect_stdout(output):
                self.magics.llm("--run", "slow-command")

        self.assertIn("timed out after 120 seconds", conversation.prompt)
        self.assertIn("partial stdout", conversation.prompt)
        self.assertIn("partial stderr", conversation.prompt)

        large_output = "first" + ("x" * 20_000) + "last"
        truncated = _truncate_command_output(large_output)
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
        with patch("ipyllm.diagnostics.subprocess.run", return_value=completed):
            result = self.shell.run_cell(
                "%%llm --run Explain this output.\npython -V",
                store_history=False,
                silent=True,
            )
        unload_ipython_extension(self.shell)

        self.assertTrue(result.success)
        self.assertIn("Python 3", conversation.prompt)
        self.assertIn("Explain this output.", conversation.prompt)


class ConfigFileTests(unittest.TestCase):
    """Cover ``.ipyllmrc`` discovery, partial loading, and saving."""

    def setUp(self):
        self.shell = InteractiveShell.instance()
        self.workdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.workdir.cleanup)
        self.home = tempfile.TemporaryDirectory()
        self.addCleanup(self.home.cleanup)
        self.old_cwd = os.getcwd()
        self.addCleanup(os.chdir, self.old_cwd)
        os.chdir(self.workdir.name)

    def _new_magics(self):
        with patch("ipyllm.magics.Path.home", return_value=Path(self.home.name)):
            return IpyllmMagics(self.shell)

    def test_config_in_current_directory_is_loaded(self):
        (Path(self.workdir.name) / ".ipyllmrc").write_text(
            "[ipyllm]\n"
            "model = test-model\n"
            "require_approval = on\n"
            "debug = on\n"
            "chain_limit = 25\n",
            encoding="utf-8",
        )
        magics = self._new_magics()
        self.assertEqual(magics._model_id, "test-model")
        self.assertTrue(magics.require_approval)
        self.assertTrue(magics.debug)
        self.assertEqual(magics.chain_limit, 25)
        self.assertEqual(magics._config_path, Path(self.workdir.name) / ".ipyllmrc")

    def test_config_in_home_when_none_in_current_directory(self):
        (Path(self.home.name) / ".ipyllmrc").write_text(
            "[ipyllm]\nchain_limit = 7\n", encoding="utf-8"
        )
        magics = self._new_magics()
        self.assertEqual(magics.chain_limit, 7)
        self.assertFalse(magics.debug)  # missing entry keeps default

    def test_current_directory_config_wins_over_home(self):
        (Path(self.home.name) / ".ipyllmrc").write_text(
            "[ipyllm]\nchain_limit = 3\n", encoding="utf-8"
        )
        (Path(self.workdir.name) / ".ipyllmrc").write_text(
            "[ipyllm]\nchain_limit = 9\n", encoding="utf-8"
        )
        magics = self._new_magics()
        self.assertEqual(magics.chain_limit, 9)

    def test_flat_file_without_section_is_accepted(self):
        (Path(self.workdir.name) / ".ipyllmrc").write_text(
            "debug = on\n", encoding="utf-8"
        )
        magics = self._new_magics()
        self.assertTrue(magics.debug)

    def test_partial_config_uses_defaults_for_missing_entries(self):
        (Path(self.workdir.name) / ".ipyllmrc").write_text(
            "[ipyllm]\ndebug = on\n", encoding="utf-8"
        )
        magics = self._new_magics()
        self.assertTrue(magics.debug)
        self.assertIsNone(magics._model_id)
        self.assertEqual(magics.chain_limit, 10)

    def test_no_config_file_uses_defaults(self):
        magics = self._new_magics()
        self.assertIsNone(magics._config_path)
        self.assertIsNone(magics._model_id)
        self.assertEqual(magics.chain_limit, 10)
        self.assertFalse(magics.require_approval)

    def test_invalid_values_warn_and_fall_back_to_defaults(self):
        (Path(self.workdir.name) / ".ipyllmrc").write_text(
            "[ipyllm]\n"
            "chain_limit = notanumber\n"
            "require_approval = maybe\n"
            "bogus_key = 1\n",
            encoding="utf-8",
        )
        stderr = StringIO()
        with redirect_stderr(stderr):
            magics = self._new_magics()
        self.assertEqual(magics.chain_limit, 10)
        self.assertFalse(magics.require_approval)
        reported = stderr.getvalue()
        self.assertIn("chain_limit", reported)
        self.assertIn("require_approval", reported)
        self.assertIn("bogus_key", reported)

    def test_save_config_writes_current_values_and_reloads(self):
        magics = self._new_magics()
        magics._model_id = "my-model"
        magics.debug = True
        magics.chain_limit = 42
        with redirect_stdout(StringIO()):
            self.assertTrue(magics._handle_command("--save-config"))

        config_path = Path(self.workdir.name) / ".ipyllmrc"
        self.assertTrue(config_path.is_file())
        content = config_path.read_text(encoding="utf-8")
        self.assertIn("model = my-model", content)
        self.assertIn("debug = on", content)
        self.assertIn("chain_limit = 42", content)

        reloaded = self._new_magics()
        self.assertEqual(reloaded._model_id, "my-model")
        self.assertEqual(reloaded.chain_limit, 42)
        self.assertTrue(reloaded.debug)

    def test_save_config_to_explicit_path(self):
        magics = self._new_magics()
        magics.chain_limit = 5
        target = Path(self.workdir.name) / "custom.rc"
        with redirect_stdout(StringIO()):
            self.assertTrue(magics._handle_command(f"--save-config {target}"))
        self.assertIn("chain_limit = 5", target.read_text(encoding="utf-8"))


class LastErrorTests(unittest.TestCase):
    """Cover ``%llm --last-error``: diagnose the most recent shell exception."""

    def setUp(self):
        self.shell = InteractiveShell.instance()
        self.magics = IpyllmMagics(self.shell)
        self.fake_model = FakeModel()
        self.model_patcher = patch(
            "ipyllm.magics.llm.get_model", return_value=self.fake_model
        )
        self.get_model = self.model_patcher.start()
        self.addCleanup(self.model_patcher.stop)
        # ``sys.last_*`` is global; snapshot and clear it so tests are isolated.
        self._saved = {
            attr: getattr(sys, attr, None)
            for attr in ("last_type", "last_value", "last_traceback", "last_exc")
        }
        for attr in self._saved:
            sys.__dict__.pop(attr, None)
        self.addCleanup(self._restore_last_error)

    def _restore_last_error(self):
        for attr, value in self._saved.items():
            if value is None:
                sys.__dict__.pop(attr, None)
            else:
                setattr(sys, attr, value)

    def _raise_in_cell(self, code: str) -> None:
        """Run ``code`` the way a user cell does, so IPython records the failure."""
        self.shell.run_cell(code, store_history=False, silent=True)

    def test_requires_a_prior_error(self):
        with self.assertRaisesRegex(UsageError, "No error has occurred"):
            self.magics.llm("--last-error")

    def test_captures_most_recent_exception(self):
        conversation = FakeConversation()
        self.magics._conversation = conversation
        self._raise_in_cell("raise ValueError('boom at x=1')")
        output = StringIO()
        with redirect_stdout(output):
            self.magics.llm("--last-error")

        self.assertIn("Last error:", output.getvalue())
        self.assertIn("ValueError: boom at x=1", output.getvalue())
        self.assertIn("Diagnose the most recent exception", conversation.prompt)
        self.assertIn("ValueError: boom at x=1", conversation.prompt)

    def test_includes_custom_request(self):
        conversation = FakeConversation()
        self.magics._conversation = conversation
        self._raise_in_cell("raise ValueError('boom')")
        self.magics.llm("--last-error why did this fail and how do I fix it")
        self.assertIn("why did this fail and how do I fix it", conversation.prompt)

    def test_exception_groups_list_sub_exceptions(self):
        conversation = FakeConversation()
        self.magics._conversation = conversation
        self._raise_in_cell(
            "raise BaseExceptionGroup("
            "'group', [ValueError('a'), RuntimeError('b')])"
        )
        self.magics.llm("--last-error")
        self.assertIn("group", conversation.prompt)
        self.assertIn("ValueError: a", conversation.prompt)
        self.assertIn("RuntimeError: b", conversation.prompt)

    def test_help_lists_last_error(self):
        output = StringIO()
        with redirect_stdout(output):
            self.magics.llm("--help")
        self.assertIn("--last-error", output.getvalue())


class AtReferenceTests(unittest.TestCase):
    """Cover ``@`` file references: content fragments, attachments, and safety."""

    def setUp(self):
        self.shell = InteractiveShell.instance()
        self.magics = IpyllmMagics(self.shell)
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        (self.root / "alpha.py").write_text("alpha = 1\n", encoding="utf-8")
        (self.root / "beta.txt").write_text("beta text", encoding="utf-8")
        self.sub = self.root / "sub"
        self.sub.mkdir()
        (self.sub / "gamma.py").write_text("gamma = 3\n", encoding="utf-8")
        self.photo = self.root / "photo.png"
        self.photo.write_bytes(b"\x89PNG\r\n\x1a\n" + b"image-bytes")
        self._old_cwd = os.getcwd()
        os.chdir(self.root)
        self.addCleanup(os.chdir, self._old_cwd)

    def tearDown(self):
        self._tmp.cleanup()

    def test_text_file_becomes_a_content_fragment(self):
        prompt, fragments, attachments = self.magics._parse_resources(
            "Summarize @alpha.py"
        )
        self.assertEqual(prompt, "Summarize [file: alpha.py]")
        self.assertEqual(len(fragments), 1)
        self.assertIn("[file: alpha.py]", fragments[0])
        self.assertIn("alpha = 1", fragments[0])
        self.assertEqual(attachments, [])

    def test_image_becomes_an_attachment(self):
        prompt, fragments, attachments = self.magics._parse_resources(
            "Describe @photo.png"
        )
        self.assertEqual(prompt, "Describe [file: photo.png]")
        self.assertEqual(fragments, [])
        self.assertEqual(len(attachments), 1)
        self.assertTrue(attachments[0].path.endswith("photo.png"))

    def test_directory_expands_to_its_files(self):
        prompt, fragments, attachments = self.magics._parse_resources(
            "Inspect @sub/ and tell me what is there"
        )
        self.assertIn("[directory: sub", prompt)
        self.assertEqual(len(fragments), 1)
        self.assertIn("gamma = 3", fragments[0])
        self.assertEqual(attachments, [])

    def test_directory_skips_hidden_entries(self):
        with_hidden = self.root / "withhidden"
        with_hidden.mkdir()
        (with_hidden / "ok.py").write_text("ok = 1\n", encoding="utf-8")
        hidden_dir = with_hidden / ".git"
        hidden_dir.mkdir()
        (hidden_dir / "cfg.py").write_text("hidden = 2\n", encoding="utf-8")

        prompt, fragments, _ = self.magics._parse_resources("Check @withhidden")

        self.assertEqual(len(fragments), 1)
        self.assertIn("ok = 1", fragments[0])
        self.assertNotIn("hidden = 2", "".join(fragments))
        self.assertIn("[directory: withhidden", prompt)

    def test_multiple_references_mixed_types(self):
        prompt, fragments, attachments = self.magics._parse_resources(
            "Compare @alpha.py and @beta.txt against @photo.png"
        )
        self.assertEqual(
            prompt,
            "Compare [file: alpha.py] and [file: beta.txt] against [file: photo.png]",
        )
        self.assertEqual(len(fragments), 2)
        self.assertEqual(len(attachments), 1)

    def test_nonexistent_reference_is_left_literal(self):
        prompt, fragments, attachments = self.magics._parse_resources(
            "See @missing.xyz here"
        )
        self.assertEqual(prompt, "See @missing.xyz here")
        self.assertEqual(fragments, [])
        self.assertEqual(attachments, [])

    def test_in_word_at_is_preserved(self):
        prompt, fragments, attachments = self.magics._parse_resources(
            "Email user@example.com please"
        )
        self.assertEqual(prompt, "Email user@example.com please")
        self.assertEqual(fragments, [])
        self.assertEqual(attachments, [])

    def test_repeated_reference_is_deduplicated(self):
        prompt, fragments, attachments = self.magics._parse_resources(
            "@alpha.py and again @alpha.py"
        )
        self.assertEqual(len(fragments), 1)
        self.assertEqual(attachments, [])

    def test_binary_non_media_file_is_rejected(self):
        blob = self.root / "data.bin"
        blob.write_bytes(b"\x00\x01\x02\x03\x04binary")
        with self.assertRaisesRegex(UsageError, "not a text file"):
            self.magics._parse_resources("Read @data.bin")

    def test_content_is_capped(self):
        big = self.root / "big.txt"
        big.write_text("x" * 20_000, encoding="utf-8")
        _, fragments, _ = self.magics._parse_resources("Read @big.txt")
        content = fragments[0].split("\n", 1)[1]
        self.assertLessEqual(len(content), 12_000)

    def test_llm_passes_fragments_and_attachments_to_the_conversation(self):
        self.magics._conversation = FakeConversation()
        with redirect_stdout(StringIO()):
            self.magics.llm("Compare @alpha.py with @photo.png")

        conversation = self.magics._conversation
        self.assertIn("[file: alpha.py]", conversation.prompt)
        self.assertIn("[file: photo.png]", conversation.prompt)
        self.assertEqual(len(conversation.fragments), 1)
        self.assertIn("alpha = 1", conversation.fragments[0])
        self.assertEqual(len(conversation.attachments), 1)
        self.assertTrue(conversation.attachments[0].path.endswith("photo.png"))

    def test_prompt_ending_in_question_mark_is_preserved(self):
        self.magics._conversation = FakeConversation()
        with redirect_stdout(StringIO()):
            self.magics.llm("@alpha.py what do you think of this plan?")

        conversation = self.magics._conversation
        self.assertIn("[file: alpha.py]", conversation.prompt)
        self.assertIn("what do you think of this plan?", conversation.prompt)
        self.assertIn("alpha = 1", conversation.fragments[0])


if __name__ == "__main__":
    unittest.main()
