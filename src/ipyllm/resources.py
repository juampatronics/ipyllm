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

import mimetypes
import re
from pathlib import Path

import llm
from IPython.core.error import UsageError

MAX_FILE_CONTENT_CHARS = 12_000
MAX_DIR_FILES = 100
MEDIA_PREFIXES = ("image/", "audio/", "video/")
MEDIA_TYPES = {"application/pdf"}
_AT_REFERENCE = re.compile(r"@([^\s@]+)")


def _list_dir_files(root: Path) -> tuple[list[Path], bool]:
    """List regular files breadth-first, skipping hidden entries and capping size."""
    files: list[Path] = []
    queue: list[Path] = [root]
    index = 0
    while index < len(queue) and len(files) < MAX_DIR_FILES:
        directory = queue[index]
        index += 1
        try:
            entries = sorted(directory.iterdir())
        except OSError:
            continue
        for entry in entries:
            if entry.name.startswith("."):
                continue
            if entry.is_dir():
                queue.append(entry)
            elif entry.is_file():
                files.append(entry)
                if len(files) >= MAX_DIR_FILES:
                    break
    return files, len(files) >= MAX_DIR_FILES


def _detect_type(path: Path) -> str | None:
    try:
        from llm.utils import mimetype_from_path

        detected = mimetype_from_path(str(path))
        if detected:
            return detected
    except Exception:
        pass
    return mimetypes.guess_type(path.name)[0]


def _display_name(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def _is_media_file(path: Path) -> bool:
    detected = _detect_type(path)
    if detected is None:
        return False
    return detected.startswith(MEDIA_PREFIXES) or detected in MEDIA_TYPES


def _read_text_resource(path: Path) -> str:
    detected = _detect_type(path) or ""
    is_known_text = detected.startswith("text/")
    try:
        with path.open("rb") as handle:
            raw = handle.read(MAX_FILE_CONTENT_CHARS * 4 + 4)
    except OSError as exc:
        raise UsageError(f"Cannot read {path}: {exc}") from exc
    if b"\x00" in raw[:8192]:
        raise UsageError(
            f"{path} is not a text file and this model cannot attach it. "
            "Use an image, audio, video, or PDF, or convert it to a text "
            "file."
        )
    try:
        content = raw.decode("utf-8")
    except UnicodeDecodeError:
        if is_known_text:
            content = raw.decode("utf-8", errors="ignore")
        else:
            raise UsageError(
                f"{path} is not a text file and this model cannot attach it. "
                "Use an image, audio, video, or PDF, or convert it to a "
                "text file."
            ) from None
    if len(content) > MAX_FILE_CONTENT_CHARS:
        content = content[:MAX_FILE_CONTENT_CHARS]
    return content


def _resolve_reference(
    reference: str, seen: set[Path]
) -> tuple[str, list[str], list]:
    path = Path(reference).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path

    if path.is_dir():
        files, truncated = _list_dir_files(path)
        if not files:
            return f"[directory: {_display_name(path)} (no files)]", [], []
        count = f"first {len(files)} files" if truncated else f"{len(files)} files"
        marker = f"[directory: {_display_name(path)} ({count})]"
    elif path.is_file():
        files = [path]
        marker = f"[file: {_display_name(path)}]"
    else:
        return f"@{reference}", [], []

    fragments: list[str] = []
    attachments: list = []
    for file in files:
        resolved = file.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if _is_media_file(file):
            attachments.append(llm.Attachment(path=str(file)))
        else:
            content = _read_text_resource(file)
            fragments.append(f"[file: {_display_name(file)}]\n{content}")
    return marker, fragments, attachments


def parse_resources(prompt: str) -> tuple[str, list[str], list]:
    """Replace resolvable ``@path`` references and collect their contents."""
    fragments: list[str] = []
    attachments: list = []
    seen: set[Path] = set()
    parts: list[str] = []
    last = 0
    for match in _AT_REFERENCE.finditer(prompt):
        start, end = match.span()
        if start == 0 or prompt[start - 1].isspace():
            marker, resource_fragments, resource_attachments = _resolve_reference(
                match.group(1), seen
            )
            parts.append(prompt[last:start])
            parts.append(marker)
            fragments.extend(resource_fragments)
            attachments.extend(resource_attachments)
        else:
            parts.append(prompt[last:end])
        last = end
    parts.append(prompt[last:])
    return "".join(parts), fragments, attachments
