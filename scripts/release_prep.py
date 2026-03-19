from __future__ import annotations

from enum import IntEnum
import re
from typing import NamedTuple


class BumpLevel(IntEnum):
    NONE = 0
    PATCH = 1
    MINOR = 2
    MAJOR = 3


class ConventionalCommit(NamedTuple):
    type: str
    scope: str | None
    bump: BumpLevel


_CONVENTIONAL_COMMIT_PATTERN = re.compile(
    r"^(?P<type>[a-z]+)(?:\((?P<scope>[^)]+)\))?(?P<breaking>!)?:\s+.+$"
)


def parse_conventional_commit(subject: str, body: str) -> ConventionalCommit:
    match = _CONVENTIONAL_COMMIT_PATTERN.match(subject.strip())
    if match is None:
        return ConventionalCommit(type="unknown", scope=None, bump=BumpLevel.NONE)

    commit_type = match.group("type")
    scope = match.group("scope")
    is_breaking = match.group("breaking") == "!" or "BREAKING CHANGE:" in body

    if is_breaking:
        bump = BumpLevel.MAJOR
    elif commit_type == "feat":
        bump = BumpLevel.MINOR
    elif commit_type == "fix":
        bump = BumpLevel.PATCH
    else:
        bump = BumpLevel.NONE

    return ConventionalCommit(type=commit_type, scope=scope, bump=bump)


def determine_bump(commits: list[ConventionalCommit]) -> BumpLevel:
    return max((commit.bump for commit in commits), default=BumpLevel.NONE)


def bump_version(version: str, bump: BumpLevel) -> str:
    major, minor, patch = (int(part) for part in version.split("."))

    if bump == BumpLevel.MAJOR:
        return f"{major + 1}.0.0"
    if bump == BumpLevel.MINOR:
        return f"{major}.{minor + 1}.0"
    if bump == BumpLevel.PATCH:
        return f"{major}.{minor}.{patch + 1}"
    return version


def cargo_semver_to_pep440(version: str) -> str:
    return (
        version.replace("-alpha.", "a")
        .replace("-beta.", "b")
        .replace("-rc.", "rc")
    )


def normalize_tag_version(tag: str) -> str:
    normalized = tag.removeprefix("refs/tags/")
    return normalized.removeprefix("v")
