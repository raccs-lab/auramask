## v0.2.0 (2026-06-08)

### BREAKING CHANGE

- relative links to classes are likely to break

### Fix

- **datasets.py,insta_filter.py**: change albumentations import as clahe is not exposed as a functional processing step anymore

### Refactor

- **auramask**: move auramask logic to a src folder and update pyproject to use uv build system
