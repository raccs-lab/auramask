# Requirements

- [uv](https://docs.astral.sh/uv/)

# Setup

To train with the AuraMask pipeline, first clone the repository:

```sh
git clone https://gitlab.com/raccs-lab/auramask
```

Then, use `uv` to set up the local environment

```sh
uv sync
```

be sure to include the `--extra` option to select the backend you would like to use

```sh
uv sync --extra torch
# or
uv sync --extra tf
```
