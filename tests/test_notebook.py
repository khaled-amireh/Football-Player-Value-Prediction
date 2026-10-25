"""Exercise small notebook cells without loading the player dataset or training."""

import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest

import pandas as pd


NOTEBOOK = Path(__file__).resolve().parents[1] / "Football_Player_Value_Prediction.ipynb"


def cell_source(cell_id):
    """Use stable cell IDs so markdown edits do not break the tests."""
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cell = next(cell for cell in notebook["cells"] if cell["id"] == cell_id)
    return "".join(cell["source"])


@contextlib.contextmanager
def temporary_working_directory():
    """Isolate the notebook's relative input path from the real checkout."""
    previous = Path.cwd()
    with tempfile.TemporaryDirectory() as directory:
        try:
            os.chdir(directory)
            yield Path(directory)
        finally:
            os.chdir(previous)


class NotebookDiagnosticTests(unittest.TestCase):
    def test_cleaning_diagnostic_displays_missing_counts_and_row_count(self):
        frame = pd.DataFrame({"pace": [None, 55.0], "value_eur": [100, 200]})
        output = io.StringIO()

        with contextlib.redirect_stdout(output):
            exec(cell_source("2b4ca065"), {"df": frame})

        self.assertRegex(output.getvalue(), r"(?m)^pace\s+1$")
        self.assertRegex(output.getvalue(), r"(?m)^value_eur\s+0$")
        self.assertRegex(output.getvalue(), r"Rows:\s+2")
        self.assertEqual(frame.shape, (2, 2))
        self.assertTrue(pd.isna(frame.loc[0, "pace"]))


class NotebookLoadingTests(unittest.TestCase):
    def test_missing_input_explains_filename_and_working_directory(self):
        with temporary_working_directory():
            with self.assertRaises(FileNotFoundError) as raised:
                exec(cell_source("48d16b18"), {"pd": pd})

        message = str(raised.exception)
        self.assertIn("footballData.csv", message)
        self.assertIn("repository root", message)
        self.assertIn("README.md", message)
        self.assertIsInstance(raised.exception.__cause__, FileNotFoundError)

    def test_existing_input_loads_without_transforming_values(self):
        with temporary_working_directory() as directory:
            (directory / "footballData.csv").write_text(
                "age,value_eur\n21,123456\n32,250000\n", encoding="utf-8"
            )
            namespace = {"pd": pd}
            exec(cell_source("48d16b18"), namespace)

        self.assertEqual(
            namespace["df"].to_dict("list"),
            {"age": [21, 32], "value_eur": [123456, 250000]},
        )


if __name__ == "__main__":
    unittest.main()
