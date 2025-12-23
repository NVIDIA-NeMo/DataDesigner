# 📓 Notebooks in `.py` Format

In this folder you can find all our tutorial notebooks in `.py` format. They can be converted to actual Jupyter notebooks by typing

```bash
make convert-execute-notebooks
```

from the root of the repository. This will not only convert but also execute all of the notebooks -- for that to work, make sure you went through our [Quick Start](https://nvidia-nemo.github.io/DataDesigner/quick-start/) and have API keys set. A new folder `docs/notebooks` will be created, including `README.md` and `pyproject.toml` files.

Alternatively, you can use Jupytext directly

```bash
uv run --group notebooks --group docs jupytext --to ipynb *.py
```

## 🔄 Converting Jupyter notebooks to `.py`

If you want to contribute with your own notebook, you can use the following command to generate `.py` files in the same format as the ones in this folder:

```bash
uv run jupytext --to py [notebook-name].ipynb -o [notebook-name].py
```
