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

import configparser
from pathlib import Path

from IPython.core.error import UsageError

CONFIG_FILE = ".ipyllmrc"
CONFIG_SECTION = "ipyllm"
CONFIG_KEYS = ("model", "require_approval", "debug", "save_session", "chain_limit")


def find_config_file() -> Path | None:
    """Return the config file to use: current directory first, then ``$HOME``."""
    for candidate in (Path.cwd() / CONFIG_FILE, Path.home() / CONFIG_FILE):
        if candidate.is_file():
            return candidate
    return None


def read_config_values(path: Path) -> dict[str, str]:
    """Parse a config file with a section, a single section, or no header."""
    try:
        content = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise UsageError(f"Cannot read config file {path}: {exc}") from exc
    parser = configparser.ConfigParser(interpolation=None)
    try:
        parser.read_string(content)
    except configparser.MissingSectionHeaderError:
        parser.read_string(f"[{CONFIG_SECTION}]\n{content}")
    except configparser.Error as exc:
        raise UsageError(f"Cannot parse config file {path}: {exc}") from exc
    if CONFIG_SECTION in parser:
        selected = parser[CONFIG_SECTION]
    elif parser.sections():
        selected = parser[parser.sections()[0]]
    else:
        selected = parser.defaults()
    return {
        key: value for key, value in selected.items() if key not in ("DEFAULT", "__name__")
    }


def parse_bool(value: str, key: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "on", "yes", "y"}:
        return True
    if normalized in {"0", "false", "off", "no", "n"}:
        return False
    raise ValueError(f"expected on or off, got {value!r}")


def format_config_file(values: dict[str, str]) -> str:
    """Render config values as an ``.ipyllmrc`` document."""
    lines = [f"[{CONFIG_SECTION}]"]
    for key in CONFIG_KEYS:
        if key in values:
            lines.append(f"{key} = {values[key]}")
    return "\n".join(lines) + "\n"
