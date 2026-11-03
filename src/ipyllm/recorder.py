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

import ast
import sys
from pathlib import Path

from IPython.core.error import UsageError
from IPython.core.interactiveshell import InteractiveShell


def _read_saved_requests(source: str) -> list[str]:
    requests = []
    lines = source.splitlines()
    index = 0
    while index < len(lines):
        if lines[index] != "# Request:":
            index += 1
            continue
        index += 1
        request_lines = []
        while index < len(lines) and lines[index].startswith("#"):
            line = lines[index][1:]
            request_lines.append(line[1:] if line.startswith(" ") else line)
            index += 1
        request = "\n".join(request_lines).strip()
        if request:
            requests.append(request)
    return requests


class SessionRecorder:
    """Save, resume, and record IPython session activity into a Python file."""

    def __init__(self, shell: InteractiveShell) -> None:
        self._shell = shell
        self.save_path: Path | None = None

    def enable(self, filename: str) -> None:
        path = Path(filename).expanduser()
        if path.suffix.lower() != ".py":
            raise UsageError("The session file must have a `.py` extension.")
        path = path.resolve()
        try:
            with path.open("x", encoding="utf-8") as session_file:
                session_file.write(
                    "#!/usr/bin/env python3\n"
                    "# -*- coding: utf-8 -*-\n"
                    "# Recorded by ipyllm. Prompts are comments; executed Python "
                    "follows.\n\n"
                )
        except FileExistsError as exc:
            raise UsageError(
                f"Refusing to overwrite existing session file: {path}"
            ) from exc
        except OSError as exc:
            raise UsageError(f"Cannot create session file {path}: {exc}") from exc

        self.save_path = path
        print(f"Recording new prompts and executed Python to {path}.")

    def disable(self) -> None:
        self.save_path = None
        print("Session recording disabled.")

    def resume(self, filename: str) -> list[str]:
        if self.save_path is not None:
            raise UsageError(
                f"Stop recording to {self.save_path} with `%llm --save off` "
                "before resuming another session."
            )
        path = Path(filename).expanduser()
        if path.suffix.lower() != ".py":
            raise UsageError("The session file must have a `.py` extension.")
        path = path.resolve()
        try:
            source = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise UsageError(f"Cannot read session file {path}: {exc}") from exc
        try:
            compile(source, str(path), "exec")
        except SyntaxError as exc:
            raise UsageError(f"Session file is not valid Python: {exc}") from exc

        try:
            with path.open("a", encoding="utf-8"):
                pass
        except OSError as exc:
            raise UsageError(f"Cannot append to session file {path}: {exc}") from exc

        self._execute_session_script(path)
        self.save_path = path
        print(
            f"Resumed Python state from {path}. New requests and executed code "
            "will be appended to this file."
        )
        return _read_saved_requests(source)

    def record_request(self, request: str) -> None:
        if self.save_path is None:
            return
        lines = ["# Request:"]
        lines.extend(f"# {line}" if line else "#" for line in request.splitlines())
        self._append_session_text("\n".join(lines) + "\n\n")

    def record_executed_code(self, code: str) -> None:
        if self.save_path is None:
            return
        try:
            ast.parse(code)
        except SyntaxError:
            lines = [
                "# IPython-only input (not executable as a standalone Python script):"
            ]
            lines.extend(f"# {line}" if line else "#" for line in code.splitlines())
            print(
                "The executed input uses IPython syntax and is saved as comments, "
                "so the session file remains valid Python."
            )
            self._append_session_text("\n".join(lines) + "\n\n")
            return
        self._append_session_text(code.rstrip() + "\n\n")

    def _append_session_text(self, text: str) -> None:
        assert self.save_path is not None
        try:
            with self.save_path.open("a", encoding="utf-8") as session_file:
                session_file.write(text)
        except OSError as exc:
            raise RuntimeError(
                f"Cannot append to session file {self.save_path}: {exc}"
            ) from exc

    def _execute_session_script(self, path: Path) -> None:
        shell = self._shell
        namespace = shell.user_ns
        missing = object()
        previous_name = namespace.get("__name__", missing)
        previous_argv = sys.argv
        namespace["__name__"] = path.stem
        namespace["__file__"] = str(path)
        sys.argv = [str(path)]
        try:
            shell.safe_execfile(
                str(path),
                namespace,
                namespace,
                raise_exceptions=True,
                shell_futures=True,
            )
        finally:
            sys.argv = previous_argv
            if previous_name is missing:
                namespace.pop("__name__", None)
            else:
                namespace["__name__"] = previous_name
