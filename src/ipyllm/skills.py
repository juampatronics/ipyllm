"""Discovery and loading for local Agent Skills."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re

import yaml

_SKILL_DIRECTORY_NAMES = (
    ".agents/skills",
    ".claude/skills",
    ".github/skills",
    ".cursor/skills",
    ".windsurf/skills",
)
_MAX_FRONTMATTER_CHARS = 16_384
MAX_SKILL_CHARS = 32_000
_SKILL_NAME = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*\Z")


class SkillError(ValueError):
    """Raised when a skill cannot be validated or loaded."""


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    path: Path

    def read_instructions(self) -> str:
        """Read validated skill instructions on explicit invocation."""
        try:
            with self.path.open(encoding="utf-8") as skill_file:
                content = skill_file.read(MAX_SKILL_CHARS + 1)
        except (OSError, UnicodeError) as exc:
            raise SkillError(f"Cannot read {self.path}: {exc}") from exc

        if len(content) > MAX_SKILL_CHARS:
            raise SkillError(
                f"{self.path} exceeds the {MAX_SKILL_CHARS}-character limit."
            )
        name, description, body = parse_skill_document(content, self.path.parent)
        if name != self.name or description != self.description:
            raise SkillError(
                f"Metadata in {self.path} changed after skill discovery; "
                "run `%llm --skills` and try again."
            )
        if not body.strip():
            raise SkillError(f"{self.path} has no skill instructions after its frontmatter.")
        return body.strip()


@dataclass
class SkillCatalog:
    skills: dict[str, Skill] = field(default_factory=dict)
    duplicates: dict[str, list[Path]] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


def default_skill_directories(
    project_dir: Path | None = None,
    user_dir: Path | None = None,
) -> list[Path]:
    """Return common project- and user-level skill roots."""
    project_dir = project_dir or Path.cwd()
    user_dir = user_dir or Path.home()
    return [
        base / relative
        for base in (project_dir, user_dir)
        for relative in _SKILL_DIRECTORY_NAMES
    ]


def discover_skills(
    extra_directories: list[str] | tuple[str, ...] = (),
    *,
    project_dir: Path | None = None,
    user_dir: Path | None = None,
) -> SkillCatalog:
    """Discover valid skills, reporting malformed and ambiguous entries."""
    roots = [
        (root, False) for root in default_skill_directories(project_dir, user_dir)
    ]
    roots.extend((Path(directory).expanduser(), True) for directory in extra_directories)
    catalog = SkillCatalog()
    seen_roots: set[Path] = set()
    found: dict[str, list[Skill]] = {}

    for root, configured in roots:
        try:
            resolved_root = root.resolve()
        except OSError as exc:
            catalog.errors.append(f"Cannot resolve skill directory {root}: {exc}")
            continue
        if resolved_root in seen_roots:
            continue
        seen_roots.add(resolved_root)

        if not resolved_root.exists():
            if configured:
                catalog.errors.append(f"Configured skill directory does not exist: {root}")
            continue
        if not resolved_root.is_dir():
            catalog.errors.append(f"Skill directory is not a directory: {root}")
            continue

        try:
            skill_directories = sorted(
                (entry for entry in resolved_root.iterdir() if entry.is_dir()),
                key=lambda entry: entry.name,
            )
        except OSError as exc:
            catalog.errors.append(f"Cannot list skill directory {root}: {exc}")
            continue

        for skill_directory in skill_directories:
            skill_path = skill_directory / "SKILL.md"
            if not skill_path.is_file():
                continue
            try:
                with skill_path.open(encoding="utf-8") as skill_file:
                    frontmatter = skill_file.read(_MAX_FRONTMATTER_CHARS + 1)
                name, description, _ = parse_skill_document(
                    frontmatter, skill_directory, metadata_only=True
                )
            except (OSError, UnicodeError, SkillError, yaml.YAMLError) as exc:
                catalog.errors.append(f"Invalid skill at {skill_path}: {exc}")
                continue
            found.setdefault(name, []).append(
                Skill(name=name, description=description, path=skill_path)
            )

    for name, skills in found.items():
        if len(skills) == 1:
            catalog.skills[name] = skills[0]
        else:
            catalog.duplicates[name] = [skill.path for skill in skills]
    return catalog


def parse_skill_document(
    content: str,
    skill_directory: Path,
    *,
    metadata_only: bool = False,
) -> tuple[str, str, str]:
    """Validate Agent Skills frontmatter and return its metadata and body."""
    content = content.replace("\r\n", "\n")
    if not content.startswith("---\n"):
        raise SkillError("expected YAML frontmatter beginning with `---`.")

    closing = content.find("\n---\n", 4)
    if closing == -1:
        raise SkillError("could not find the closing `---` for YAML frontmatter.")
    if closing > _MAX_FRONTMATTER_CHARS:
        raise SkillError(
            f"frontmatter exceeds the {_MAX_FRONTMATTER_CHARS}-character limit."
        )

    try:
        metadata = yaml.safe_load(content[4:closing])
    except yaml.YAMLError as exc:
        raise SkillError(f"invalid YAML frontmatter: {exc}") from exc
    if not isinstance(metadata, dict):
        raise SkillError("frontmatter must contain a YAML mapping.")
    name = metadata.get("name")
    description = metadata.get("description")
    if not isinstance(name, str) or not _SKILL_NAME.fullmatch(name) or "--" in name:
        raise SkillError("`name` must be lowercase letters, numbers, or single hyphens.")
    if len(name) > 64:
        raise SkillError("`name` cannot exceed 64 characters.")
    if skill_directory.name != name:
        raise SkillError(
            f"`name` must match the skill directory name {skill_directory.name!r}."
        )
    if not isinstance(description, str) or not description.strip():
        raise SkillError("`description` must be a non-empty string.")
    if len(description) > 1024:
        raise SkillError("`description` cannot exceed 1024 characters.")

    body = content[closing + len("\n---\n") :]
    return name, description.strip(), "" if metadata_only else body
