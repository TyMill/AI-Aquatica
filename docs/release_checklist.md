# AI-Aquatica v2.3.0 release checklist

Run these commands from a clean checkout of the exact commit intended for the release.

## 1. Create a clean environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[testing]"
```

## 2. Static analysis and tests

```bash
python -m ruff check .
python -m pytest --cov=ai_aquatica --cov-report=term-missing --cov-fail-under=70
python -m compileall src/ai_aquatica examples
```

Expected reference result: `76 passed, 1 skipped` and whole-package coverage above 70%. The single optional skip concerns TensorFlow when the deep-learning extra is not installed.

## 3. Reproduce the SoftwareX results

```bash
rm -rf validation/real_dataset
python examples/real_dataset_workflow.py --output validation/real_dataset --bootstrap 1000
```

Check that the numerical results agree with `VALIDATION.md`. The files under `validation/real_dataset/` are the canonical release outputs used for the revised manuscript.

## 4. Build the distributions

```bash
rm -rf build dist src/*.egg-info
python -m build
```

Expected files:

```text
dist/ai_aquatica-2.3.0-py3-none-any.whl
dist/ai_aquatica-2.3.0.tar.gz
```

## 5. Smoke-test the exact wheel

```bash
python -m pip install --force-reinstall --no-deps dist/ai_aquatica-2.3.0-py3-none-any.whl
cd /tmp
python -c "import ai_aquatica; assert ai_aquatica.__version__ == '2.3.0'; print(ai_aquatica.__version__)"
```

The command must print `2.3.0`.

## 6. Freeze the release

Only after the local checks and GitHub Actions are green:

1. commit the verified source and `validation/real_dataset/` outputs;
2. create and push tag `v2.3.0`;
3. create the GitHub release from that exact tag;
4. publish the same version to PyPI so the installation command resolves to the reviewed release;
5. archive the exact GitHub tag in Zenodo;
6. add the new Zenodo DOI to the manuscript and release metadata after the DOI exists.

Do not reuse an older Zenodo DOI as the archive for v2.3.0.
