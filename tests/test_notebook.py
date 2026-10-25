"""Exercise small notebook cells without loading the player dataset or training."""

import contextlib
import io
import json
from pathlib import Path
import unittest

import pandas as pd


NOTEBOOK = Path(__file__).resolve().parents[1] / "Football_Player_Value_Prediction.ipynb"


def cell_source(cell_id):
    """Use stable cell IDs so markdown edits do not break the tests."""
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cell = next(cell for cell in notebook["cells"] if cell["id"] == cell_id)
    return "".join(cell["source"])


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


if __name__ == "__main__":
    unittest.main()
