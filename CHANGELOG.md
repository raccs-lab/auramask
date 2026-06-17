## v1.2.2 (2026-06-17)

### BREAKING CHANGE

- Closes #82

### Fix

- **cli/train.py,utils/datasets.py**: implement tensorflow dataset loading

## v1.2.1 (2026-06-16)

### Fix

- **utils/datasets.py,insta_filter.py**: change collater function for visual transforms to fix crashing
- **cli/train.py**: remove unused table arguments to callback
- **callbacks/metriclogger.py**: remove reliance on tensorflow.keras
- **callbacks/evaluate.py**: remove reliance on wandb keras integration

### Refactor

- **callbacks/softadapt.py,stop_on_nan.py**: remove full callbacks import and instead import specific class
- **callbacks/checkpoint.py**: remove wandb keras integration import that was unused and avoid use of superfluous typing imports

## v1.2.0 (2026-06-15)

### BREAKING CHANGE

- dsssim,gsssim,ffl removed as loss options
- testing and training scripts will no longer be used the same way

### Feat

- **cli/train.py**: remove losses that were no longer supported by the pipeline from the cli
- **metrics/ssim.py**: implement and export new ssim approach
- **test.py,train.py**: add new implementation of ssim as an option for loss
- **ssim.py**: implementation of ssim with keras operations
- **pyproject.toml**: cpu based optional dependency for pytorch

### Fix

- **cli/test.py**: add ssim metric to testing script
- **cli/train.py**: rename lambda argument to lam to avoid conflict with reserved name in python
- **pyproject.toml**: accidental commit with two tf extras instead of tf-cpu
- **ssim.py**: fix bugs that kept ssim from running in channels first mode
- **ssim.py,test_ssim.py**: recover ssim implementation from before and update test for clearer keyword parameters
- **ssim.py**: syntax issues when using ops vs tf functions
- **ssim.py**: replace shape with correct newshape keyword argument for reshape op
- **datasets.py**: use huggingface iterative dataset and map to apply alterations and dataloader for conversion to pytorch tensor
- **datasets.py**: data transforms done in huggingface dataset library
- **preprocessing.py**: albumentations removed internal ref to cv2 so needed to import cv2
- **insta_filter.py**: select the 'image' key of the clahe output

### Refactor

- **cli/__init__.py,cli/train.py**: generalize argument parsing logic for ease of command differentiation
- **cli/train.py**: move checking logic for the lam and losses arguments to come before any instantiations
- **__init__.py,ssim.py**: remove duplicate implementation and align more with skimage implementation
- **tests/**: organizing existing tests into folders
- **test.py,train.py,cli,pyproject.toml,utils/cli.py**: move executable testing and training scripts to be subcommands under an auramask command

## v1.1.0 (2026-06-08)

### BREAKING CHANGE

- relative links to classes are likely to break

### Fix

- **datasets.py,insta_filter.py**: change albumentations import as clahe is not exposed as a functional processing step anymore

### Refactor

- **auramask**: move auramask logic to a src folder and update pyproject to use uv build system

## v1.0.0 (2026-06-08)
