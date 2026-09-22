#!/usr/bin/env python
"""Check the comment hygiene of the lines this branch adds.

Diffs the working tree against the merge base with the base ref, collects the
added lines in Python and shell files, and applies the rules below. Untracked
in-scope files count entirely as added.

Scoping to added lines is what makes the check usable: existing comment debt
stays out of the way, and a branch is answerable only for what it writes.

The standard it enforces is the code-comments skill, named in the failure output.
"""

from __future__ import annotations

import argparse
import ast
import io
import json
import re
import subprocess
import sys
import tokenize
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

MAX_BLOCK_CHARS = 200
DEFAULT_MAX_LINE_LENGTH = 120
TAB_WIDTH = 4

SCOPED_PATH_SPECS = [
    "machine/*.py",
    "tests/*.py",
    "scripts/*.py",
    "local_check.sh",
]

# Built from code points rather than literals so the set survives a source
# re-encoding. Only these are reported; a comment may otherwise carry any script
# the language data needs.
NON_ASCII_PUNCTUATION = [
    "—",
    "–",
    "→",
    "←",
    "↔",
    "…",
    "•",
    "×",
    "‘",
    "’",
    "“",
    "”",
    "§",
]

_ABSENCE_NARRATION = (
    r"\b(?:it|this|that|these|those|we|they|which)\s+used to\b"
    r"|\bused to be\b|\bpreviously (?:read|worked|did|returned|used|called)\b"
    r"|\b(?:was|were) removed\b|\b(?:was|were) stale\b|\brenamed from\b|\bfirst shipped\b"
    r"|\bno longer (?:used|needed|exists|exist|supported|present|applies|apply|valid)\b"
)

# A bare "used to" or "no longer" usually reads as purpose or present state, so
# both require explicitly historical phrasing.
_RULES: dict[str, str] = {
    "process-framing": r"\bPhase[\s-]?\d+\b|\blater we'll\b|\bwe'll (?:later|eventually)\b",
    "doc-pointer": r"\b[\w./-]+\.md\b|\bsection\s+\d+[a-z]?\b",
    "absence-narration": _ABSENCE_NARRATION,
    "cross-file-pointer": r"\bsee [A-Za-z]+'s note\b|\bas documented (?:on|in) [A-Za-z]+\b",
    "provenance": r"\bshared by \w+ and \w+\b|\bthe only caller\b|\bthe sole caller\b|\bextracted from\b",
    "non-ascii-punctuation": "(?:" + "|".join(re.escape(c) for c in NON_ASCII_PUNCTUATION) + ")",
}

CATEGORIES = {name: re.compile(pattern, re.IGNORECASE) for name, pattern in _RULES.items()}


@dataclass(frozen=True)
class Violation:
    file: str
    line: int
    category: str
    text: str


@dataclass
class CommentLine:
    """One whole-line comment: its text, and whether the budget counts it."""

    line: int
    body: str
    exempt: bool


def display_width(line: str) -> int:
    """Return a line's width in columns, expanding tabs to the next tab stop."""
    width = 0
    for char in line:
        if char == "\t":
            width += TAB_WIDTH - (width % TAB_WIDTH)
        else:
            width += 1
    return width


def max_line_length(repo_root: Path) -> int:
    """Read the width limit from black's setting so the two cannot drift apart."""
    pyproject = repo_root / "pyproject.toml"
    if not pyproject.exists():
        return DEFAULT_MAX_LINE_LENGTH
    in_black = False
    for raw in pyproject.read_text(encoding="utf-8").splitlines():
        stripped = raw.strip()
        if stripped.startswith("["):
            in_black = stripped == "[tool.black]"
            continue
        if not in_black:
            continue
        match = re.match(r"^line-length\s*=\s*(\d+)$", stripped)
        if match:
            return int(match.group(1))
    return DEFAULT_MAX_LINE_LENGTH


def _docstring_rows(tree: ast.AST) -> set[int]:
    """Return every line a module, class, or function docstring occupies."""
    rows: set[int] = set()
    holders = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
    for node in ast.walk(tree):
        if not isinstance(node, holders) or not node.body:
            continue
        first = node.body[0]
        if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
            if isinstance(first.value.value, str) and first.end_lineno is not None:
                rows.update(range(first.lineno, first.end_lineno + 1))
    return rows


def _python_comments(text: str) -> list[CommentLine]:
    """Classify a Python file's whole-line comments and its docstrings.

    Uses the tokenizer rather than line prefixes so a number sign inside a
    string literal is never mistaken for a comment. A docstring is exempt from
    the budget, not from the content rules or the width limit.
    """
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(text).readline))
        tree = ast.parse(text)
    except (tokenize.TokenError, IndentationError, SyntaxError, ValueError):
        return []

    lines = text.splitlines()
    skip = (tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT, tokenize.ENDMARKER)
    code_rows = {token.start[0] for token in tokens if token.type not in skip}

    comments: list[CommentLine] = []
    for token in tokens:
        if token.type != tokenize.COMMENT:
            continue
        row = token.start[0]
        if row in code_rows:
            continue
        if row == 1 and token.string.startswith("#!"):
            continue
        comments.append(CommentLine(row, token.string.lstrip("#"), exempt=False))

    for row in sorted(_docstring_rows(tree)):
        if row <= len(lines):
            comments.append(CommentLine(row, lines[row - 1], exempt=True))

    return sorted(comments, key=lambda c: c.line)


def _shell_comments(text: str) -> list[CommentLine]:
    """Classify a shell script's whole-line comments. A shebang is not one."""
    comments: list[CommentLine] = []
    for index, raw in enumerate(text.splitlines(), start=1):
        stripped = raw.strip()
        if index == 1 and stripped.startswith("#!"):
            continue
        if stripped.startswith("#"):
            comments.append(CommentLine(index, stripped[1:], exempt=False))
    return comments


def classify(path: Path, text: str) -> Optional[list[CommentLine]]:
    """Return the file's whole-line comments, or None when it is out of scope."""
    if path.suffix == ".py":
        return _python_comments(text)
    if path.suffix == ".sh" or path.name == "local_check.sh":
        return _shell_comments(text)
    return None


def scan_file(path: Path, repo_root: Path, allowed: Optional[set[int]], width_limit: int) -> list[Violation]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []

    comments = classify(path, text)
    if comments is None:
        return []

    lines = text.splitlines()
    relative = path.relative_to(repo_root).as_posix()
    violations: list[Violation] = []

    for comment in comments:
        if allowed is not None and comment.line not in allowed:
            continue
        for category, pattern in CATEGORIES.items():
            if pattern.search(comment.body):
                violations.append(Violation(relative, comment.line, category, comment.body.strip()))
        # Width applies to every comment line including a docstring: those are
        # exempt from what they may say, not from how wide they may run.
        raw = lines[comment.line - 1] if comment.line <= len(lines) else ""
        width = display_width(raw)
        if width > width_limit:
            text_ = f"{width} columns (max {width_limit}): {comment.body.strip()}"
            violations.append(Violation(relative, comment.line, "comment-line-too-long", text_))

    violations.extend(_budget_violations(comments, relative, allowed))
    return violations


def _budget_violations(comments: list[CommentLine], relative: str, allowed: Optional[set[int]]) -> list[Violation]:
    """Report a run of consecutive non-exempt comment lines over the budget.

    A block whose untouched lines alone already exceed the budget is left to a
    separate cleanup, so a branch is never blocked by debt it did not write.
    """
    violations: list[Violation] = []
    by_line = {c.line: c for c in comments if not c.exempt}
    block: list[CommentLine] = []

    def flush() -> None:
        if not block:
            return
        total = sum(len(c.body.strip()) for c in block)
        if total <= MAX_BLOCK_CHARS:
            return
        if allowed is not None:
            if not any(c.line in allowed for c in block):
                return
            untouched = sum(len(c.body.strip()) for c in block if c.line not in allowed)
            if untouched > MAX_BLOCK_CHARS:
                return
        text = f"{total} chars (budget {MAX_BLOCK_CHARS}): {block[0].body.strip()}"
        violations.append(Violation(relative, block[0].line, "comment-too-long", text))

    for line in sorted(by_line):
        if block and line != block[-1].line + 1:
            flush()
            block = []
        block.append(by_line[line])
    flush()
    return violations


def _git(args: Sequence[str], cwd: Optional[Path] = None) -> str:
    result = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, encoding="utf-8")
    if result.returncode != 0:
        return ""
    return result.stdout


def repo_root() -> Path:
    top = _git(["rev-parse", "--show-toplevel"]).strip()
    if not top:
        raise SystemExit("comment-hygiene must run inside a git working tree.")
    return Path(top)


def resolve_base_ref(explicit: Optional[str]) -> str:
    """Return the first ref git can resolve, preferring an explicit value.

    git remote show is deliberately not used: resolving a base must not need
    network access on a developer machine.
    """
    import os

    candidates = [explicit] if explicit else []
    base_ref = os.environ.get("GITHUB_BASE_REF")
    if base_ref:
        candidates.append(f"origin/{base_ref}")
    candidates += ["origin/HEAD", "origin/main"]
    for candidate in candidates:
        if _git(["rev-parse", "--verify", "--quiet", f"{candidate}^{{commit}}"]).strip():
            return candidate
    raise SystemExit(f"No usable base ref. Tried: {', '.join(candidates)}. Fetch the base branch, or pass --base-ref.")


def added_line_filter(base: str, root: Path) -> tuple[str, dict[Path, set[int]]]:
    """Map each in-scope path to the line numbers this branch adds.

    Diffs the working tree against the merge base so local edits are checked
    before they are committed.
    """
    merge_base = _git(["merge-base", base, "HEAD"]).strip()
    if not merge_base:
        raise SystemExit(f"No merge base between {base} and HEAD. Fetch enough history (fetch-depth: 0 in CI).")

    filter_: dict[Path, set[int]] = {}
    current: Optional[Path] = None
    line_number = 0
    diff = _git(["diff", "--unified=0", merge_base, "--", *SCOPED_PATH_SPECS])
    for line in diff.splitlines():
        if line.startswith("+++ "):
            path = line[4:].strip()
            if path == "/dev/null":
                current = None
                continue
            current = root / (path[2:] if path.startswith("b/") else path)
            filter_.setdefault(current, set())
            continue
        match = re.match(r"^@@ -\S+ \+(\d+)(?:,\d+)? @@", line)
        if match:
            line_number = int(match.group(1))
            continue
        if current is not None and line.startswith("+"):
            filter_[current].add(line_number)
            line_number += 1

    untracked = _git(["ls-files", "--others", "--exclude-standard", "--", *SCOPED_PATH_SPECS])
    for path in untracked.splitlines():
        if not path.strip():
            continue
        full = root / path.strip()
        if not full.exists():
            continue
        count = len(full.read_text(encoding="utf-8", errors="replace").splitlines())
        filter_[full] = set(range(1, count + 1))

    return merge_base, filter_


SELF_TEST_CASES: list[tuple[str, str, list[str], list[str]]] = [
    ("clean python is clean", "clean.py", ["# Guards against a missing row.", "a = 1"], []),
    (
        "docstring is budget exempt",
        "doc.py",
        ['"""' + "y" * 60, "y" * 60, "y" * 60, "y" * 60 + '"""', "a = 1"],
        [],
    ),
    (
        "block budget aggregates lines",
        "block.py",
        [f"# {'y' * 60}"] * 4 + ["a = 1"],
        ["comment-too-long"],
    ),
    ("exactly at budget is clean", "exact.py", [f"# {'z' * 50}"] * 4 + ["a = 1"], []),
    (
        "one over budget is reported",
        "over.py",
        [f"# {'z' * 50}"] * 3 + [f"# {'z' * 50}a", "a = 1"],
        ["comment-too-long"],
    ),
    (
        "blank line ends a block",
        "split.py",
        [f"# {'y' * 60}", f"# {'y' * 60}", "", f"# {'y' * 60}", f"# {'y' * 60}", "a = 1"],
        [],
    ),
    ("width counts the marker and indent", "wide.py", [f"# {'x' * 130}", "a = 1"], ["comment-line-too-long"]),
    ("em dash is reported", "dash.py", ["# Uses a dash — here.", "a = 1"], ["non-ascii-punctuation"]),
    (
        "absence narration is reported",
        "absence.py",
        ["# This field is no longer used.", "a = 1"],
        ["absence-narration"],
    ),
    ("markdown pointer is reported", "pointer.py", ["# See design-notes.md for why.", "a = 1"], ["doc-pointer"]),
    ("provenance is reported", "prov.py", ["# Extracted from the old tokenizer.", "a = 1"], ["provenance"]),
    ("a number sign in a string is not a comment", "str.py", ['a = "# no longer used"'], []),
    (
        "a multi-line docstring reports its content once",
        "multi.py",
        ['"""Summary.', "", "    Extracted from the old tokenizer.", '    """', "a = 1"],
        ["provenance"],
    ),
    ("a trailing comment is not a whole-line comment", "trail.py", ["a = 1  # no longer used"], []),
    ("shell shebang is exempt", "run.sh", ["#!/bin/bash", "echo hi"], []),
    ("shell comment uses the same budget", "budget.sh", [f"# {'y' * 60}"] * 4 + ["echo hi"], ["comment-too-long"]),
    ("unscoped extension is skipped", "notes.md", ["# no longer relevant"], []),
]


def self_test() -> int:
    """Exercise the rules against fixtures, so they stay verifiable anywhere."""
    import tempfile

    failures: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "pyproject.toml").write_text("[tool.black]\nline-length = 120\n", encoding="utf-8")
        width = max_line_length(root)
        for name, filename, content, expected in SELF_TEST_CASES:
            path = root / filename
            path.write_text("\n".join(content) + "\n", encoding="utf-8")
            found = sorted(v.category for v in scan_file(path, root, None, width))
            if found != sorted(expected):
                failures.append(f"{name}: expected {sorted(expected)} but found {found}")

    if failures:
        print("comment-hygiene self-test FAILED")
        for failure in failures:
            print(f"  {failure}")
        return 1
    print("comment-hygiene self-test passed")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Check comment hygiene over the lines this branch adds.")
    parser.add_argument("--base-ref", help="Ref to diff against. Defaults to the PR base, then origin/HEAD, main.")
    parser.add_argument("--full", action="store_true", help="Scan every tracked in-scope file, to size existing debt.")
    parser.add_argument("--advisory", action="store_true", help="Report violations and still exit 0.")
    parser.add_argument("--report-path", help="Optional path for a JSON report.")
    parser.add_argument("--self-test", action="store_true", help="Run the built-in rule cases and exit.")
    args = parser.parse_args(argv)

    if args.self_test:
        return self_test()

    root = repo_root()
    width = max_line_length(root)
    line_filter: Optional[dict[Path, set[int]]] = None

    if args.full:
        tracked = _git(["ls-files", "--", *SCOPED_PATH_SPECS])
        files = [root / p.strip() for p in tracked.splitlines() if p.strip()]
        scope = f"all {len(files)} tracked in-scope files"
    else:
        base = resolve_base_ref(args.base_ref)
        merge_base, line_filter = added_line_filter(base, root)
        files = list(line_filter)
        scope = f"lines added since {base} (merge base {merge_base[:8]})"

    if not files:
        print(f"comment-hygiene: no in-scope files to check ({scope}).")
        return 0

    violations: list[Violation] = []
    for path in files:
        allowed = line_filter.get(path) if line_filter is not None else None
        violations.extend(scan_file(path, root, allowed, width))
    violations.sort(key=lambda v: (v.file, v.line, v.category))

    if args.report_path:
        report = Path(args.report_path)
        report.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "scope": scope,
            "advisory": args.advisory,
            "violationCount": len(violations),
            "violations": [v.__dict__ for v in violations],
        }
        report.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if not violations:
        print(f"comment-hygiene: clean ({scope}).")
        return 0

    print(f"comment-hygiene: {len(violations)} violation(s) in {scope}.")
    print()
    _report(violations, advisory=args.advisory)
    print()
    print("The standard is .claude/skills/code-comments/SKILL.md.")
    print("Reproduce locally with: poetry run python scripts/comment_hygiene.py")

    if args.advisory:
        print("Advisory only; not failing the run.")
        return 0
    return 1


def _report(violations: Iterable[Violation], advisory: bool) -> None:
    import os

    in_actions = os.environ.get("GITHUB_ACTIONS") == "true"
    for violation in violations:
        print(f"{violation.file}:{violation.line}: {violation.category}: {violation.text}")
        if advisory and in_actions:
            print(f"::warning file={violation.file},line={violation.line}::{violation.category}: {violation.text}")


if __name__ == "__main__":
    sys.exit(main())
