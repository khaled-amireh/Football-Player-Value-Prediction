# Dataset provenance and local setup

The notebook expects `footballData.csv` in its working directory. Launch Jupyter
from the repository root and place the CSV beside
`Football_Player_Value_Prediction.ipynb`. The input is excluded by `.gitignore`;
the repository does not redistribute it.

## Obtain the original input

The repository cites [The Devastator's Kaggle listing](https://www.kaggle.com/datasets/thedevastator/footballpriceprediction).
Direct checks during the README audit returned HTTP 404 in that environment,
while a cached listing remained visible. This is an access limitation, not proof
that the dataset is permanently deleted. Check the listing for current access.

If it is inaccessible, obtain the original CSV and its provenance from the project
author or original provider. Do not assume that another FIFA CSV is identical
because its columns or dimensions look similar. Review the provider's usage terms;
the repository's MIT license does not establish a license for third-party data.

## Check a local copy before training

The recorded input had **18,944 rows and 106 columns**. Its exact download version,
original archive filename, FIFA edition, and checksum were not recorded. Those
facts cannot be reconstructed from the notebook's outputs alone.

The notebook selects these 15 columns, including the target:

```text
age, overall, potential, height_cm, weight_kg, weak_foot, skill_moves,
international_reputation, pace, shooting, passing, dribbling, defending,
physic, value_eur
```

It removes rows missing any selected value, leaving **16,861 records** in the
recorded run. It does not impute, rescale, or log-transform the target. A matching
shape and schema are useful checks, but do not establish that two files contain
the same player records.

## Record provenance for a new run

Keep a local note of the source URL, provider, download date, dataset version,
license or usage terms, original filename, and SHA-256 checksum. The following
commands read the file without changing it; choose one for your platform.

Windows PowerShell:

```powershell
Get-FileHash -LiteralPath .\footballData.csv -Algorithm SHA256
```

macOS:

```bash
shasum -a 256 footballData.csv
```

Linux:

```bash
sha256sum footballData.csv
```

There is no original checksum to compare against. A newly recorded checksum
identifies your local copy for future runs; it does not authenticate the historical
training input. Report a run on a different dataset as a separate experiment and
preserve the existing saved results.
