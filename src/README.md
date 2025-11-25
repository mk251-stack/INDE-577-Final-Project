# Rice ML Source Code

The `rice_ml` package implements machine learning utilities and models used in
INDE 577. The code favors readability over performance so students can follow
end-to-end workflows without heavy external dependencies.

## Modules

- `supervised_learning/`: k-NN models, decision trees, and linear regression
- `preprocessing/`: scaling, imputation, and feature engineering helpers
- `post_processing/`: analysis helpers for model outputs
- `data/`: datasets used in examples and demonstrations

## Development

Install dependencies in editable mode:

```bash
pip install -e .[dev]
```

Run tests from the repository root:

```bash
pytest
```
