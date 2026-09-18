# Vision Model MLOps Pipeline

**Live demo:** [vision-model-mlops-pipeline-fw2257fsulhbfjrqhsp4nc.streamlit.app](https://vision-model-mlops-pipeline-fw2257fsulhbfjrqhsp4nc.streamlit.app)

![CI](https://github.com/Dhayalramesh/vision-model-mlops-pipeline/actions/workflows/ci.yml/badge.svg)

An end-to-end MLOps pipeline: multiple training configurations are logged and compared
with **MLflow**, validated automatically on every push via **GitHub Actions CI**, and the
best model is served through a live **Streamlit** dashboard with an inference demo.

## Pipeline

1. `train.py` trains a small CNN image classifier (CIFAR-10) across several hyperparameter
   configurations, logging params/metrics to MLflow (SQLite backend).
2. The best run (by validation accuracy) is exported as `best_model.pth`; all run results
   are exported as `runs_summary.csv`.
3. `.github/workflows/ci.yml` runs a fast smoke-test training pass on every push, so the
   pipeline is verified to work end-to-end before deployment — not just claimed.
4. `app.py` is a Streamlit dashboard that displays the experiment comparison and serves
   the current best model for live image classification.

## Results

Three configurations were trained and compared automatically:

| Run | Learning Rate | Epochs | Batch Size | Val Accuracy | Train Loss |
|---|---|---|---|---|---|
| **run_lr001_bs32_ep5** ⭐ | 0.001 | 5 | 32 | **0.5905** | 0.8558 |
| run_lr001_bs64 | 0.001 | 3 | 64 | 0.5530 | 1.2118 |
| run_lr0005_bs64 | 0.0005 | 3 | 64 | 0.5145 | 1.3683 |

The dashboard automatically identifies and serves the best-performing run (highest
validation accuracy) for live inference — no manual model selection required.

## Run locally / in Colab

```bash
pip install -r requirements.txt mlflow scikit-learn
python train.py
```

This produces `best_model.pth` and `runs_summary.csv` in the project root.

## Deploy

1. Push this repo to GitHub (include `best_model.pth` and `runs_summary.csv`).
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub.
3. **New app** → select this repo → main file path `app.py` → **Deploy**.

## Continuous Integration

Every push to `main` triggers a GitHub Actions workflow that:
- Installs dependencies (CPU-only PyTorch build, for speed)
- Runs a fast smoke-test training pass (`python train.py --fast`)
- Verifies `runs_summary.csv` is generated correctly
- Checks `app.py` for syntax errors

This ensures the pipeline stays functional as the codebase changes, rather than
relying on manual verification before each deploy.

## Stack

Python · PyTorch · MLflow · GitHub Actions (CI) · Streamlit · Streamlit Community Cloud

## Project Structure

```
vision-model-mlops-pipeline/
├── app.py                          # Streamlit dashboard + inference UI
├── train.py                        # Training script (used locally, in Colab, and in CI)
├── requirements.txt
├── best_model.pth                  # Best model artifact (auto-selected by val_accuracy)
├── runs_summary.csv                # Exported MLflow run comparison data
├── README.md
└── .github/
    └── workflows/
        └── ci.yml                  # Automated pipeline validation on every push
```
