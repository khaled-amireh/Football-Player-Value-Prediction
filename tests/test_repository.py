"""Temporary fixtures exercise the offline repository validator without training."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

from tools.check_repository import check_repository, main


class RepositoryChecksTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.notebook = {
            "cells": [
                {"cell_type": "code", "source": ["features = ['age', 'overall']\n"], "outputs": []},
                {
                    "cell_type": "code", "source": "pass\n",
                    "outputs": [{"output_type": "stream", "text": [
                        "R2 Score : 0.9877130651871356\n",
                        "MAE: 134716.98784464868\n",
                        "RMSE: 590496.9112839565\n",
                    ]}],
                },
            ],
        }
        self.readme = """# Example

[Overview](#example) · [Guide](docs/Guide.md#setup)
![Plot](Images/chart.png)

| # | Feature | Meaning |
|---|---|---|
| 1 | `age` | Age |
| 2 | `overall` | Rating |

| Metric | Result |
|---|---:|
| **R²** | **0.987713** |
| **MAE** | **€134,716.99** |
| **RMSE** | **€590,496.91** |

```text
R2 Score : 0.9877130651871356
MAE: 134716.98784464868
RMSE: 590496.9112839565
```
"""
        self.write("README.md", self.readme)
        self.write("docs/Guide.md", "# Setup\n\n[Home](../README.md#example)\n")
        self.write("Images/chart.png", "fixture image; existence is checked, not image decoding")
        self.save_notebook()

    def write(self, name, text):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def save_notebook(self):
        self.write("Football_Player_Value_Prediction.ipynb", json.dumps(self.notebook))

    def errors(self):
        return "\n".join(check_repository(self.root))

    def test_valid_repository_and_read_only_behavior(self):
        before = {p.relative_to(self.root): p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        self.assertEqual(check_repository(self.root), [])
        after = {p.relative_to(self.root): p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        self.assertEqual(after, before)

    def test_missing_local_image(self):
        (self.root / "Images/chart.png").unlink()
        self.assertIn("local target does not exist: Images/chart.png", self.errors())

    def test_exact_filename_case_on_all_platforms(self):
        self.write("README.md", self.readme.replace("Images/chart.png", "images/chart.png"))
        self.assertIn("filename case does not match", self.errors())

    def test_missing_cross_document_anchor(self):
        self.write("README.md", self.readme.replace("Guide.md#setup", "Guide.md#missing"))
        self.assertIn("anchor does not exist: docs/Guide.md#missing", self.errors())

    def test_fences_inline_code_and_external_links_are_not_local_targets(self):
        extra = """
```markdown
[Bad](missing.md)
# Fake
```
~~~markdown
<img src="missing.png">
~~~
`[Example](also-missing.md)`
[Website](https://example.invalid/missing#no-network)
[Email](mailto:example@example.invalid)
<img src="//example.invalid/remote.png">
"""
        self.write("README.md", self.readme + extra)
        self.assertEqual(check_repository(self.root), [])

    def test_fenced_headings_do_not_create_real_anchors(self):
        self.write("README.md", self.readme + "\n```\n# Hidden\n```\n[Bad](#hidden)\n")
        self.assertIn("anchor does not exist: #hidden", self.errors())

    def test_duplicate_heading_slugs_and_inline_formatting(self):
        self.write("README.md", self.readme + "\n## **Repeated**\n## Repeated\n[Second](#repeated-1)\n")
        self.assertEqual(check_repository(self.root), [])

    def test_encoded_paths_explicit_anchors_and_html_images(self):
        self.write("docs/Extra Guide.md", '<a id="special"></a>\n')
        self.write("README.md", self.readme + '\n[Extra](docs/Extra%20Guide.md#special)\n<img src="Images/chart.png">\n')
        self.assertEqual(check_repository(self.root), [])

    def test_repository_escape_is_rejected(self):
        self.write("README.md", self.readme + "\n[Outside](../outside.md)\n")
        self.assertIn("escapes the repository", self.errors())

    def test_ignored_virtual_environment_documents(self):
        self.write(".venv/README.md", "[Bad](missing.md)\n")
        self.assertEqual(check_repository(self.root), [])

    def test_invalid_json_and_missing_notebook(self):
        self.write("Football_Player_Value_Prediction.ipynb", "{")
        self.assertIn("cannot read notebook", self.errors())
        (self.root / "Football_Player_Value_Prediction.ipynb").unlink()
        self.assertIn("cannot read notebook", self.errors())

    def test_invalid_cell_structure_and_source(self):
        self.notebook["cells"].extend([None, {"cell_type": "code", "source": [1]}])
        self.save_notebook()
        self.assertIn("invalid cell structure", self.errors())
        self.assertIn("expected text", self.errors())

    def test_python_syntax_failure_is_actionable(self):
        self.notebook["cells"].append({"cell_type": "code", "source": "if broken\n"})
        self.save_notebook()
        self.assertIn("cell 3: invalid Python on line 1", self.errors())

    def test_feature_extraction_does_not_depend_on_cell_positions(self):
        self.notebook["cells"].insert(0, {"cell_type": "markdown", "source": "# New introduction"})
        self.save_notebook()
        self.assertEqual(check_repository(self.root), [])

    def test_feature_table_must_match_names_and_order(self):
        self.write("README.md", self.readme.replace("`overall`", "`potential`"))
        self.assertIn("no Feature table matches", self.errors())

    def test_dynamic_or_duplicate_feature_lists_are_reported(self):
        for source in ("features = get_features()", "features = ['age', 'age']"):
            with self.subTest(source=source):
                self.notebook["cells"][0]["source"] = source
                self.save_notebook()
                self.assertIn("cannot read features", self.errors())

    def test_stale_or_invalid_documented_metric(self):
        for value in ("€134,000.00", "unknown", "NaN"):
            with self.subTest(value=value):
                self.write("README.md", self.readme.replace("€134,716.99", value))
                self.assertIn("MAE result", self.errors())

    def test_displayed_precision_is_respected(self):
        self.write("README.md", self.readme.replace("**0.987713**", "**0.9877**"))
        self.assertEqual(check_repository(self.root), [])

    def test_missing_and_conflicting_saved_metrics(self):
        output = self.notebook["cells"][1]["outputs"][0]
        output["text"] = "MAE: 1\nMAE: 2\n"
        self.save_notebook()
        errors = self.errors()
        self.assertIn("conflicting saved MAE", errors)
        self.assertIn("missing saved R2", errors)
        self.assertIn("missing saved RMSE", errors)

    def test_exact_saved_output_lines_are_checked_separately(self):
        self.write("README.md", self.readme.replace("MAE: 134716.98784464868", "MAE: 134716.99"))
        self.assertIn("exact MAE output differs", self.errors())

    def test_missing_metric_row_and_readme(self):
        self.write("README.md", self.readme.replace("| **MAE** | **€134,716.99** |\n", ""))
        self.assertIn("missing documented MAE", self.errors())
        (self.root / "README.md").unlink()
        self.assertIn("required documentation file", self.errors())

    def test_cli_status_and_actionable_output(self):
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(main(["--root", str(self.root)]), 0)
        self.assertIn("PASS:", output.getvalue())
        self.write("docs/Guide.md", "[Bad](missing.md)")
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(main(["--root", str(self.root)]), 1)
        self.assertIn("docs", output.getvalue())
        self.assertIn("missing.md", output.getvalue())


if __name__ == "__main__":
    unittest.main()
