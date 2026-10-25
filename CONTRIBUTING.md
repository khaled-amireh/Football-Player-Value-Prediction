# Contributing

Keep changes focused, reviewable, and supported by the repository's actual code
and recorded evidence. Start with the [README](README.md) for setup and project
scope, and inspect `git status --short` before editing so unrelated work stays
outside your change.

## Prepare the environment

Use the README's virtual-environment and requirements commands. Notebook metadata
records Python 3.11.9, and CI uses Python 3.11; dependency versions remain unpinned.
See [setup troubleshooting](docs/TROUBLESHOOTING.md) for kernel mismatches or
activation problems.

The [dataset provenance guide](docs/DATA.md) describes the original input and
current access limitations. Keep `footballData.csv` local and do not redistribute
third-party data without verified rights. The checks below need no player CSV.

## Edit notebooks without changing historical evidence

- Preserve cell IDs, order, execution counts, and saved outputs when changing
  prose, comments, or formatting. The README's audit trail refers to cell positions.
- Do not run all cells merely to validate a documentation change. The saved
  figures and metrics describe a historical experiment.
- Keep model parameters, selected features, data preparation, and evaluation
  behavior unchanged unless the change explicitly calls for a new experiment.
- Treat new training results as a separate, documented experiment. Record data
  provenance, environment versions, evaluation conditions, and validation evidence.
- For behavior fixes, include a focused regression test that exercises the
  affected cell or utility with small temporary fixtures, without training.

## Validate locally

Run these commands from the repository root after installing requirements:

```bash
python tools/check_repository.py
python -m unittest discover -s tests -v
git diff --check
git status --short
```

The checker uses only the Python standard library and performs no network
requests. It validates the direct Markdown links, HTML image references, and
heading conventions used here; parser limitations are described in
[its module documentation](tools/check_repository.py). External links and actual
GitHub rendering still require separate review when relevant.

The [CI workflow](.github/workflows/checks.yml) runs the checker and regression
tests on Ubuntu and Windows. Passing those checks establishes repository
consistency, not reproduction of the historical ML results.

## Submit one cohesive change

Review the diff and stage only files belonging to your improvement. Include code,
its regression tests, and necessary documentation in the same change. Use a clear
Conventional Commit message, such as `fix: explain a missing dataset error` or
`docs: clarify kernel selection`. Update the README's project tree when adding
project files.

Describe what changed, why it matters, which checks ran, and what could not be
verified. Keep source-access limitations and unimplemented roadmap items explicit.
