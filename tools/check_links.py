"""Walk every relative link in the project's docs/ and root .md files.

A relative ``.md`` link or asset reference that resolves to a missing
file is a doc-rot bug: a maintainer who edits in good faith leaves a
404 on the docs site, the README, or the dashboard.  This script is
the CI gate that catches the mistake before a release.

    python -m tools.check_links          # exit 0 if all links resolve
    phm-check-links                     # after ``pip install -e .``

Scopes walked:
    *.md in the repo root
    *.md in docs/
    *.html in docs/
    *.md in docs/PROVENANCE.md, REMEDIATION.md, ... (any nested .md)

What is checked:
    - ``./foo``, ``../foo``, ``foo/bar`` (no scheme) -> exists on disk
    - relative links inside Markdown ``[text](target)`` blocks
    - relative ``src=`` and ``href=`` attributes in HTML
    - Markdown image links ``![alt](target)``

What is *not* checked:
    - absolute URLs (``https://...``) -- they may be online, we cannot tell
    - ``#fragment``-only links -- they're navigation
    - mailto: / tel: -- they're contact, not file refs
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


def _repo_root() -> Path:
    here = Path(__file__).resolve().parent
    for ancestor in (here, *here.parents):
        if (ancestor / "pyproject.toml").is_file():
            return ancestor
    return Path.cwd()


_REPO = _repo_root()


# Files we are willing to walk.  Markdown and HTML only -- other extensions
# are unlikely to contain maintainer-authored relative links.
_EXTENSIONS = (".md", ".markdown", ".html", ".htm")

# Regex for a Markdown link target inside ``[text](target)`` or ``![alt](target)``.
_MD_LINK_RE = re.compile(r"\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
# Regex for an HTML attribute value: src=..., href=... (single or double quoted).
_HTML_ATTR_RE = re.compile(r"""\b(?:src|href)\s*=\s*["']([^"']+)["']""")


def _is_external(target: str) -> bool:
    """True if the link target is something we cannot resolve locally."""
    if not target:
        return True
    lowered = target.lower()
    return (
        lowered.startswith(("http://", "https://", "mailto:", "tel:", "#",
                            "data:", "javascript:"))
        or target.startswith("//")
    )


def _candidate_files() -> list[Path]:
    """List every .md / .html in the repo root and docs/ (recursively)."""
    files: list[Path] = []
    for ext in _EXTENSIONS:
        files.extend(_REPO.rglob(f"*{ext}"))
    return sorted(p for p in files if ".venv" not in p.parts
                  and "node_modules" not in p.parts
                  and "__pycache__" not in p.parts)


def _extract_targets(path: Path) -> list[str]:
    """Pull every relative link target from a markdown or html file."""
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return []
    targets: list[str] = []
    if path.suffix in (".md", ".markdown"):
        targets.extend(m.group(1) for m in _MD_LINK_RE.finditer(text))
    elif path.suffix in (".html", ".htm"):
        targets.extend(m.group(1) for m in _HTML_ATTR_RE.finditer(text))
        # Markdown-style links inside <script>-free HTML are still common
        # in this project (e.g. <a href="REMEDIATION.md">).
        targets.extend(m.group(1) for m in _MD_LINK_RE.finditer(text))
    return targets


def _resolve(linker: Path, target: str) -> Path:
    """Resolve a relative target against the linking file's directory.

    Strips any ``#fragment`` and ``?query`` from the target before
    resolving -- the file must exist for the link to be a real link.
    """
    cleaned = target.split("#", 1)[0].split("?", 1)[0]
    if not cleaned:
        return linker  # fragment-only link, not a file ref
    if cleaned.startswith("/"):
        return _REPO / cleaned.lstrip("/")
    return (linker.parent / cleaned).resolve()


def main() -> int:
    broken: list[tuple[Path, str, str]] = []
    for linker in _candidate_files():
        for target in _extract_targets(linker):
            if _is_external(target):
                continue
            resolved = _resolve(linker, target)
            if not resolved.exists():
                rel_linker = linker.relative_to(_REPO)
                broken.append((rel_linker, target, str(resolved)))

    if not broken:
        scanned = sum(1 for _ in _candidate_files())
        print(f"link-freshness OK: {scanned} files scanned, 0 broken links")
        return 0

    for linker, target, resolved in broken:
        print(f"::error::{linker}: broken link '{target}' -> {resolved}")
    print(f"link-freshness FAIL: {len(broken)} broken link(s)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
