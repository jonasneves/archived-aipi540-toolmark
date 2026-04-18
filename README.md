# Toolmark

**Production prompt-injection guardrails drop from ~98% aggregate accuracy to 7–37% detection on agent-targeted indirect attacks** ([Fomin 2026](https://arxiv.org/abs/2602.14161)). Nobody has cleanly measured *which tools* cause the collapse. Toolmark is a per-tool operating-point analysis of indirect-prompt-injection detection on agent trajectories, plus a lightweight tool-conditioned classifier that exposes what aggregate benchmarks hide.

Final project for Duke **AIPI 540: Deep Learning Applications** (Spring 2026). Due 2026-04-21.

## Status

Scaffolding (2026-04-17). See [`SCOPE.md`](SCOPE.md) for the plan and [`REQUIREMENTS_CHECKLIST.md`](REQUIREMENTS_CHECKLIST.md) for the rubric tracker.

## What this project is not

Anthropic Claude Code auto-mode (launched 2026-03-24) ships a server-side probe that scans tool outputs before they enter the agent context — the same architectural layer this project operates in. Toolmark does **not** claim to beat that probe on capability. What it contributes is a measurement artifact: per-tool recall at a fixed 1% FPR on public injection benchmarks with tool identity preserved, plus a tool-conditioned baseline that quantifies the lift from conditioning.

Aggregate benchmarks hide which tools are under-sampled or adversarially fragile. Toolmark exposes that.

## Structure

```
.
├── README.md
├── SCOPE.md
├── REQUIREMENTS_CHECKLIST.md
├── requirements.txt
├── setup.py
├── app.py                  # local dev server wrapper; real app is docs/
├── scripts/
│   ├── make_dataset.py     # fetch + join InjecAgent / BIPIA / WASP with tool metadata
│   ├── build_features.py   # TF-IDF + tool one-hot; DistilBERT tokenization
│   └── model.py            # train + eval naive / classical / DL
├── data/
│   ├── raw/                # downloaded benchmarks (gitignored)
│   ├── processed/          # joined dataset with tool labels
│   └── outputs/            # predictions, calibration curves
├── models/                 # serialized artifacts (gitignored except metadata)
├── notebooks/              # exploration only; not graded
├── results/                # scores.json, plots, confusion matrices
├── report/                 # written report drafts, figures for the writeup
├── public/                 # static app source: ONNX Runtime Web + WebGPU
├── docs -> public          # symlink so GH Pages serves main/docs → public/
└── .gitignore
```

## Team

Jonas Neves · Duke University · AIPI 540 · Spring 2026
