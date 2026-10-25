"""Offline checks for this repository's documentation and saved notebook evidence.

The Markdown reader supports ATX headings, inline links/images, and HTML href/src
attributes. Use these conventions for checked documents: reference-style links,
setext headings, and nested parentheses in link destinations are not parsed.
Notebook cells must contain plain Python. Nothing is executed or rewritten.
"""

from __future__ import annotations

import argparse
import ast
from decimal import Decimal, InvalidOperation
from html import unescape
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit


NOTEBOOK = "Football_Player_Value_Prediction.ipynb"
IGNORED_DIRECTORIES = {
    ".git", ".venv", "venv", "node_modules", "__pycache__", ".ipynb_checkpoints"
}
NUMBER = r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"
SAVED_METRIC = re.compile(
    rf"^\s*(R2(?: Score)?|R²(?: Score)?|MAE|RMSE)\s*:\s*({NUMBER})\s*$",
    re.IGNORECASE | re.MULTILINE,
)
INLINE_LINK = re.compile(
    r"!?\[[^\]\n]*\]\(\s*(?:<([^>\n]+)>|([^\s)\n]+))"
    r"(?:\s+[\"'][^\"']*[\"'])?\s*\)"
)


def _text(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return "".join(value)
    raise ValueError("expected text or a list of text lines")


def _without_fences(text: str) -> str:
    lines = []
    fence = None
    for line in text.splitlines():
        match = re.match(r"^ {0,3}(`{3,}|~{3,})(.*)$", line)
        if fence:
            if match and match[1][0] == fence[0] and len(match[1]) >= len(fence):
                if not match[2].strip():
                    fence = None
            lines.append("")
        elif match:
            fence = match[1]
            lines.append("")
        else:
            lines.append(line)
    return "\n".join(lines)


def _plain(text: str) -> str:
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    return unescape(re.sub(r"<[^>]+>", "", text)).replace("`", "").replace("*", "")


class _HTMLReferences(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.targets: list[str] = []
        self.anchors: set[str] = set()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        for name, value in attrs:
            if value and name in {"href", "src"}:
                self.targets.append(value)
            if value and (name == "id" or (tag == "a" and name == "name")):
                self.anchors.add(value)


def _markdown_facts(text: str) -> tuple[set[str], list[str]]:
    visible = _without_fences(text)
    # Inline code examples are not links or HTML declarations.
    links_text = re.sub(r"(`+).*?\1", "", visible)
    parser = _HTMLReferences()
    parser.feed(links_text)
    anchors = set(parser.anchors)
    generated = set()
    for heading in re.findall(r"^ {0,3}#{1,6}\s+(.+?)\s*#*\s*$", visible, re.MULTILINE):
        base = re.sub(r"[^\w -]", "", _plain(heading).lower()).replace(" ", "-")
        slug = base
        suffix = 0
        while slug in generated:
            suffix += 1
            slug = f"{base}-{suffix}"
        generated.add(slug)
        anchors.add(slug)
    targets = parser.targets + [a or b for a, b in INLINE_LINK.findall(links_text)]
    return anchors, targets


def _markdown_files(root: Path) -> list[Path]:
    paths = []
    for directory, names, files in os.walk(root):
        names[:] = sorted(name for name in names if name not in IGNORED_DIRECTORIES)
        paths.extend(Path(directory) / name for name in sorted(files) if name.endswith(".md"))
    return paths


def _check_target(root: Path, document: Path, target: str, anchors: dict[Path, set[str]]) -> str | None:
    try:
        url = urlsplit(unescape(target))
    except ValueError:
        return f"invalid link destination: {target}"
    if url.scheme or url.netloc:
        return None
    path = unquote(url.path)
    destination = (root / path.lstrip("/") if path.startswith("/") else document.parent / path)
    # abspath normalizes '..' without correcting case on Windows, unlike resolve.
    destination = Path(os.path.abspath(destination)) if path else document
    try:
        parts = destination.relative_to(root).parts
        destination.resolve().relative_to(root)
    except ValueError:
        return f"local target escapes the repository: {target}"
    current = root
    for part in parts:
        if not current.is_dir():
            return f"local target does not exist: {target}"
        names = {entry.name for entry in current.iterdir()}
        if part not in names:
            if part.casefold() in {name.casefold() for name in names}:
                return f"filename case does not match: {target}"
            return f"local target does not exist: {target}"
        current = current / part
    if url.fragment and destination.suffix.lower() == ".md":
        fragment = unquote(url.fragment)
        if fragment not in anchors.get(destination, set()):
            return f"heading or explicit anchor does not exist: {target}"
    return None


def _metric_name(label: str) -> str:
    return _plain(label).strip().upper().replace("²", "2").replace(" SCORE", "")


def _notebook_facts(path: Path, errors: list[str]) -> tuple[list[str] | None, dict[str, Decimal]]:
    features = None
    metrics = {}
    try:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(notebook, dict) or not isinstance(notebook.get("cells"), list):
            raise ValueError("expected a notebook object with a cells list")
    except (OSError, UnicodeError, ValueError) as exc:
        errors.append(f"{path.name}: cannot read notebook: {exc}")
        return None, {}
    for index, cell in enumerate(notebook["cells"], 1):
        location = f"{path.name}: cell {index}"
        if not isinstance(cell, dict) or cell.get("cell_type") not in {"code", "markdown", "raw"}:
            errors.append(f"{location}: invalid cell structure")
            continue
        try:
            source = _text(cell.get("source"))
        except ValueError as exc:
            errors.append(f"{location}: {exc}")
            continue
        if cell["cell_type"] != "code":
            continue
        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            errors.append(f"{location}: invalid Python on line {exc.lineno}: {exc.msg}")
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "features" for target in node.targets
            ):
                try:
                    value = ast.literal_eval(node.value)
                    if not isinstance(value, list) or not value or not all(isinstance(x, str) for x in value):
                        raise ValueError("features must be a nonempty list of names")
                    if len(set(value)) != len(value):
                        raise ValueError("feature names must be unique")
                    if features is not None and features != value:
                        raise ValueError("conflicting features assignments")
                    features = value
                except (ValueError, TypeError) as exc:
                    errors.append(f"{location}: cannot read features: {exc}")
        outputs = cell.get("outputs", [])
        if not isinstance(outputs, list):
            errors.append(f"{location}: outputs must be a list")
            continue
        for output in outputs:
            if not isinstance(output, dict) or output.get("output_type") != "stream":
                continue
            try:
                output_text = _text(output.get("text"))
            except ValueError as exc:
                errors.append(f"{location}: invalid output: {exc}")
                continue
            for label, value in SAVED_METRIC.findall(output_text):
                name, number = _metric_name(label), Decimal(value)
                if name in metrics and metrics[name] != number:
                    errors.append(f"{location}: conflicting saved {name} values")
                metrics[name] = number
    if features is None:
        errors.append(f"{path.name}: missing literal features list")
    for name in ("R2", "MAE", "RMSE"):
        if name not in metrics:
            errors.append(f"{path.name}: missing saved {name} output")
    return features, metrics


def _tables(text: str) -> list[list[list[str]]]:
    tables = []
    current = []
    for line in _without_fences(text).splitlines() + [""]:
        if line.startswith("|") and line.rstrip().endswith("|"):
            row = [cell.strip() for cell in re.split(r"(?<!\\)\|", line.strip())[1:-1]]
            if not all(re.fullmatch(r":?-{2,}:?", cell) for cell in row):
                current.append(row)
        elif current:
            tables.append(current)
            current = []
    return tables


def _check_evidence(text: str, features: list[str] | None, metrics: dict[str, Decimal]) -> list[str]:
    errors = []
    documented_features = []
    seen_metrics = set()
    for table in _tables(text):
        for column, header in enumerate(table[0]):
            if _plain(header).strip().lower() == "feature":
                documented_features.append([
                    _plain(row[column]).strip() for row in table[1:] if len(row) > column
                ])
        for row in table[1:]:
            if len(row) < 2:
                continue
            name = _metric_name(row[0])
            if name not in metrics:
                continue
            seen_metrics.add(name)
            raw = _plain(row[1]).replace("€", "").replace(",", "").strip()
            try:
                number = Decimal(raw)
                if not number.is_finite() or metrics[name].quantize(Decimal(1).scaleb(number.as_tuple().exponent)) != number:
                    raise ValueError("does not match saved output at the displayed precision")
            except (InvalidOperation, ValueError) as exc:
                errors.append(f"README.md: {name} result {row[1]!r} is invalid: {exc}")
    if features is not None and features not in documented_features:
        errors.append("README.md: no Feature table matches the notebook's feature names and order")
    for name in metrics.keys() - seen_metrics:
        errors.append(f"README.md: missing documented {name} metric table row")
    for label, value in SAVED_METRIC.findall(text):
        name = _metric_name(label)
        if name in metrics and Decimal(value) != metrics[name]:
            errors.append(f"README.md: exact {name} output differs from saved notebook output")
    return errors


def check_repository(root: Path) -> list[str]:
    """Check local documentation and saved evidence, without data or execution."""
    root = root.resolve()
    if not root.is_dir():
        return [f"repository directory does not exist: {root}"]
    errors = []
    documents = {}
    facts = {}
    for path in _markdown_files(root):
        try:
            documents[path] = path.read_text(encoding="utf-8")
            facts[path] = _markdown_facts(documents[path])
        except (OSError, UnicodeError) as exc:
            errors.append(f"{path.relative_to(root)}: cannot read Markdown: {exc}")
    anchors = {path: value[0] for path, value in facts.items()}
    for path, (_, targets) in facts.items():
        for target in targets:
            error = _check_target(root, path, target, anchors)
            if error:
                errors.append(f"{path.relative_to(root)}: {error}")
    features, metrics = _notebook_facts(root / NOTEBOOK, errors)
    readme = documents.get(root / "README.md")
    if readme is None:
        errors.append("README.md: required documentation file is missing or unreadable")
    else:
        errors.extend(_check_evidence(readme, features, metrics))
    return sorted(set(errors))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args(argv)
    errors = check_repository(args.root)
    if errors:
        for error in errors:
            print(f"FAIL: {error}")
        return 1
    print("PASS: local documentation, notebook syntax, features, and saved metrics agree.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
