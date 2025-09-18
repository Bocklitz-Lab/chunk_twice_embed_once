# Contributing to Chunk Twice, Embed Once

First off, thank you for considering contributing to this project! 🎉  
We welcome bug reports, feature requests, code improvements, and documentation updates.

---

## 📋 Code of Conduct

Please review and follow our [Code of Conduct](CODE_OF_CONDUCT.md).  
We are committed to providing a welcoming and inclusive environment.

---

## 🛠 How to Contribute

### 1. Fork & Clone
- Fork the repository on GitHub.
- Clone your fork locally:

```bash
git clone https://github.com/<your-username>/chunk-twice-embed-once.git
cd chunk-twice-embed-once
````

### 2. Set Up Environment

Install dependencies and the local package:

```bash
pip install -r requirements.txt
pip install -e .
```

Run the tests to verify everything is working:

```bash
pytest tests/
```

### 3. Create a Branch

Use a descriptive branch name:

```bash
git checkout -b fix-chunk-overlap-bug
```

### 4. Make Your Changes

* Keep functions small and focused.
* Use docstrings and type hints.
* Follow the repo’s structure (`src/ragchem/*` for code, `tests/*` for tests).

### 5. Add/Update Tests

Ensure new features or bug fixes include tests in `tests/`.
Run tests locally before pushing:

```bash
pytest
```

### 6. Commit Convention

Use clear commit messages:

* `fix: resolve off-by-one in chunking`
* `feat: add support for new embedding model`
* `docs: update README with quickstart`

### 7. Push & Open a PR

Push your branch and open a pull request against `main`:

```bash
git push origin fix-chunk-overlap-bug
```

In your PR:

* Describe the changes clearly.
* Reference related issues (`Fixes #123`).
* Include screenshots/plots if relevant.

---

## 🧪 Testing Guidelines

We use **pytest**.

* Place new tests in `tests/` with filenames starting with `test_`.
* Use small mock inputs rather than large datasets when possible.
* Ensure tests pass on Python 3.9+.

---

## 📊 Experiments & Reproducibility

* Large runs (25 chunking configs × 48 embedding models) should not be committed.
* Place intermediate logs in `experiments/` (ignored by `.gitignore` if large).
* Only commit small sample outputs or configs.
* Use Git LFS for large artifacts in `results/`.

---

## 💡 Suggestions for First Contributions

* Improve docstrings and typing.
* Add missing unit tests.
* Expand `README.md` with usage examples.
* Write a tutorial notebook (`notebooks/`) for a single config run.

---

## 🙏 Thanks

Every contribution counts, whether it’s fixing a typo, adding a test, or implementing a new retrieval strategy.
We appreciate your help in making **Chunk Twice, Embed Once** better for the community!