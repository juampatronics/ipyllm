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

import re
import shlex
import subprocess
import sys
import traceback
from collections.abc import Callable

from IPython.core.error import UsageError
from IPython.core.interactiveshell import InteractiveShell

_MAX_COMMAND_OUTPUT_CHARS = 12_000
_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def _decode_output(output: str | bytes | None) -> str:
    if output is None:
        return ""
    if isinstance(output, bytes):
        return output.decode("utf-8", errors="replace")
    return output


def _format_command_output(stdout: str, stderr: str) -> str:
    sections = []
    if stdout:
        sections.append(f"stdout:\n{stdout.rstrip()}")
    if stderr:
        sections.append(f"stderr:\n{stderr.rstrip()}")
    return "\n\n".join(sections) if sections else "(command produced no output)"


def _truncate_command_output(output: str) -> str:
    if len(output) <= _MAX_COMMAND_OUTPUT_CHARS:
        return output
    omitted = len(output) - _MAX_COMMAND_OUTPUT_CHARS
    half = _MAX_COMMAND_OUTPUT_CHARS // 2
    return (
        f"{output[:half]}\n\n"
        f"... {omitted} characters omitted ...\n\n"
        f"{output[-half:]}"
    )


def _strip_ansi(text: str) -> str:
    """Remove ANSI escape codes so the model receives clean text."""
    return _ANSI_ESCAPE.sub("", text)


def _get_last_exc_info() -> tuple[type[BaseException], BaseException, object]:
    """Return the last exception IPython recorded, or raise ``UsageError``.

    IPython stores the most recent unhandled exception in ``sys.last_type``,
    ``sys.last_value``, and ``sys.last_traceback`` whenever a cell raises one.
    """
    if getattr(sys, "last_type", None) is None:
        raise UsageError(
            "No error has occurred in this session yet. Run some code that "
            "fails, then use `%llm --last-error` to analyze it."
        )
    return sys.last_type, sys.last_value, sys.last_traceback


def _is_exception_group(value: BaseException | None) -> bool:
    while value is not None:
        if isinstance(value, BaseExceptionGroup):
            return True
        value = value.__context__
    return False


class DiagnosticRunner:
    """Run diagnostic shell commands and diagnose the last shell exception."""

    def __init__(self, shell: InteractiveShell, ask: Callable[[str], None]) -> None:
        self._shell = shell
        self._ask = ask

    def run_command(self, line: str, command: str, timeout: int) -> None:
        try:
            args = shlex.split(line)
        except ValueError as exc:
            raise UsageError(str(exc)) from exc
        if not args or args[0] != "--run":
            raise UsageError("Use `%%llm --run [diagnostic request]`.")
        if not command.strip():
            raise UsageError("Put the shell command to run in the cell body.")
        diagnostic_request = " ".join(args[1:]) or (
            "Diagnose this command output, explain the likely cause of any "
            "failure, and suggest next steps. Do not rerun the command unless "
            "I ask you to."
        )

        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                check=False,
            )
            return_code = result.returncode
            stdout = result.stdout
            stderr = result.stderr
            status = f"exit status: {return_code}"
        except subprocess.TimeoutExpired as exc:
            return_code = None
            stdout = _decode_output(exc.stdout)
            stderr = _decode_output(exc.stderr)
            status = f"timed out after {timeout} seconds"

        captured = _format_command_output(stdout, stderr)
        print(f"Command finished ({status}).")
        print(captured)
        model_output = _truncate_command_output(captured)
        prompt = (
            f"Diagnose the result of this user-requested shell command.\n"
            f"Command:\n{command}\n\n"
            f"Result: {status}\n"
            f"Captured output (possibly truncated):\n{model_output}\n\n"
            f"User request: {diagnostic_request}"
        )
        self._ask(prompt)

    def diagnose_last_error(self, request: str) -> None:
        """Diagnose the most recent exception in this IPython session."""
        etype, value, tb = _get_last_exc_info()
        traceback_text = _strip_ansi(
            self._format_last_traceback(etype, value, tb)
        )
        print("Last error:")
        print(traceback_text)
        model_output = _truncate_command_output(traceback_text)
        prompt = (
            "Diagnose the most recent exception in this IPython session.\n"
            f"Exception: {etype.__name__}: {value}\n\n"
            f"Traceback (possibly truncated):\n{model_output}\n\n"
            f"User request: {request}"
        )
        self._ask(prompt)

    def _format_last_traceback(
        self,
        etype: type[BaseException],
        value: BaseException,
        tb: object,
    ) -> str:
        """Render the last exception the way IPython displays it to the user.

        Exception groups fall back to the standard-library formatter, which is
        the only one that lists sub-exceptions; anything else uses IPython's
        structured traceback (with source context), with a standard-library
        fallback if that is unavailable.
        """
        if _is_exception_group(value):
            return "".join(traceback.format_exception(etype, value, tb))
        shell = self._shell
        try:
            structured = shell.InteractiveTB.structured_traceback(etype, value, tb)
            return shell.InteractiveTB.stb2text(structured)
        except Exception:
            return "".join(traceback.format_exception(etype, value, tb))
