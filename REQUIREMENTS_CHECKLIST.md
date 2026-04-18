# AIPI 540 Final Project — Rubric Tracker

Tracks the published rubric line-by-line. `[x]` = demonstrated in the repo; `[ ]` = outstanding. Linked to the file or section that evidences each item.

## Written Report

- [ ] Problem Statement — `report/report.md#problem-statement`
- [ ] Data Sources — `report/report.md#data-sources`
- [ ] Related Work — `report/report.md#related-work`
- [ ] Evaluation Strategy & Metrics (with explicit rationale) — `report/report.md#evaluation`
- [ ] Modeling Approach
  - [ ] Data Processing Pipeline (with rationale per step)
  - [ ] Hyperparameter Tuning Strategy
  - [ ] Naive baseline (majority + keyword heuristic)
  - [ ] Classical ML model (XGBoost + TF-IDF + tool one-hot)
  - [ ] Deep Learning model (DistilBERT fine-tune)
- [ ] Results
  - [ ] Quantitative comparison across all three models
  - [ ] Per-tool recall@1%FPR leaderboard
  - [ ] Calibration curves
  - [ ] Confusion matrices where appropriate
- [ ] Error Analysis (5 specific mispredictions, root causes, mitigations)
- [ ] Experiment Write-Up (tool-conditional recall experiment)
  - [ ] Experimental plan
  - [ ] Results
  - [ ] Interpretation
  - [ ] Recommendations
- [ ] Conclusions + Future Work (another semester)
- [ ] Commercial Viability Statement
- [ ] Ethics Statement

## Demo Day Pitch (5 minutes max, hard stop)

- [ ] Problem & Motivation slide
- [ ] Approach Overview slide
- [ ] Live Demo (tool selector + paste + score)
- [ ] Results / Insights slide (per-tool leaderboard)
- [ ] Rehearsed to fit within 5 minutes

## Code & Deployment

- [ ] GitHub repo link
- [ ] Deployed live app link (GitHub Pages + ONNX Runtime Web)
- [ ] App runs inference on a trained model (not training)
- [ ] App runs when graded (zero if it doesn't)
- [ ] UX passes the "not a basic Streamlit" bar

## Git Best Practices

- [ ] `develop` branch created
- [ ] Feature branches with PRs into `develop`
- [ ] No direct commits to `main`
- [ ] PR review history visible
- [ ] `main` holds the submission state

## Code Quality

- [ ] All code modularized into classes and functions
- [ ] No loose executable code outside functions or `if __name__ == "__main__"`
- [ ] Descriptive variable names, docstrings, and comments where WHY is non-obvious
- [ ] External code attributed at the top of each file with a link
- [ ] Jupyter notebooks confined to `notebooks/` (not graded per rubric)

## Repository Structure

- [ ] `README.md`
- [ ] `requirements.txt`
- [ ] `setup.py`
- [ ] `app.py`
- [ ] `scripts/` with pipeline scripts
- [ ] `models/` for trained model artifacts
- [ ] `data/raw`, `data/processed`, `data/outputs`
- [ ] `notebooks/`
- [ ] `.gitignore`
