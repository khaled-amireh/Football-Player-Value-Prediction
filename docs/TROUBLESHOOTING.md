# Notebook setup troubleshooting

Run terminal commands from the repository root. Install the existing requirements
in your virtual environment before launching the notebook. These remedies address
setup; they do not reproduce the recorded ML results by themselves.

## The notebook cannot find footballData.csv

The loader uses a path relative to the kernel's working directory. Run this in a
temporary notebook cell to inspect that directory:

```python
from pathlib import Path

print(Path.cwd())
print(Path("footballData.csv").is_file())
```

Start Jupyter from the repository root and place the original CSV beside the
notebook. A different dataset should be reported as a new experiment. See the
[dataset guide](DATA.md) for access and provenance limitations.

## An installed package is missing in the notebook

The terminal and notebook may use different Python environments. Inspect the
terminal interpreter:

```bash
python -c "import sys; print(sys.executable)"
python -m pip --version
```

In a temporary notebook cell, inspect the kernel interpreter:

```python
import sys

print(sys.executable)
```

If these paths point to different environments, register the activated project
environment as a named kernel:

```bash
python -m ipykernel install --user --name football-player-value --display-name "Python (football-player-value)"
```

Select **Python (football-player-value)** in Jupyter and restart the kernel before
running cells in order. Registering the kernel writes a user-level kernelspec;
using the same name again replaces that named registration. See the
[official IPython kernel instructions](https://ipython.readthedocs.io/en/stable/install/kernel_install.html).

## PowerShell blocks Activate.ps1

You can use the environment's interpreter directly without changing execution
policy or activating the environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m notebook Football_Player_Value_Prediction.ipynb
```

The README also provides activation for Windows Command Prompt and macOS/Linux.

## A fresh run differs from the saved results

Check the dataset source/version and checksum, interpreter, and installed package
versions first. Dependencies are unpinned, and the original environment was not
fully captured. A fixed seed alone does not establish exact reproducibility.

Record your current packages locally with:

```bash
python -m pip freeze
```

Keep that environment report separate from the historical notebook evidence.
Do not overwrite saved outputs merely to make them match a new environment.
