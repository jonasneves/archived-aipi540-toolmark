# Toolmark Scope

## Thesis

Production prompt-injection guardrails report ~98% aggregate accuracy and collapse to 7–37% detection on agent-targeted indirect attacks (Fomin 2026, *When Benchmarks Lie*). The aggregate number hides **which tools** cause the collapse. Toolmark contributes a per-tool operating-point analysis on public injection benchmarks with tool identity preserved, and a lightweight tool-conditioned classifier that quantifies the lift from conditioning.

This is a measurement contribution. No capability claim vs. Anthropic Claude Code auto-mode's server-side probe.

## Why this is not a me-too project

| Competitor | What they ship | What Toolmark adds |
|---|---|---|
| Llama PromptGuard 2 / Lakera Guard / Qualifire Sentinel | Binary injection classifier, tool-agnostic, server-side | Tool-stratified measurement; surfaces which tools production classifiers miss |
| Claude Code auto-mode (Anthropic) | Production server-side probe on tool outputs | Public, reproducible per-tool leaderboard; no internal data required |
| *When Benchmarks Lie* (Fomin 2026) | Aggregate Leave-One-Dataset-Out AUC drop across 18 datasets | Tool-conditional drill-down: *which tools* cause the drop, not just *that* it drops |
| MindGuard (arXiv 2508.20412) | Attention-based Decision Dependence Graphs on metadata poisoning | Text-content layer, not metadata; orthogonal |
| WAInjectBench (arXiv 2510.01354) | Systematized text + image detectors for web agents | Cross-tool surface (not just web); operating-point-constrained |
| CourtGuard (arXiv 2510.19844) | Local multi-agent injection classifier | Tool-identity conditioning is the axis, not locality |

## Research questions

1. **Measurement**: For each tool class (shell, web fetch, file read, email, MCP metadata), what is recall at 1% FPR for binary injection detection on agent trajectories? Where is the gap largest?
2. **Lift**: Does conditioning on tool identity (as a feature) improve recall@1%FPR over a tool-agnostic baseline? By how much and on which tools?
3. **Honesty**: Does deduplication by attack *template* (not string) change the picture vs. published numbers?

## Data

| Source | Role | Notes |
|---|---|---|
| **InjecAgent** (ACL 2024) | Primary attack population | 1,054 cases × 17 user tools × 62 attacker tools; tool identity preserved |
| **BIPIA** (Yi et al.) | Training volume augmentation | ~2K examples across email / web / chat |
| **WASP** (arXiv 2504.18575) | Distribution-shift test set | VisualWebArena-based, 84 user+injection combos |
| Benign synthesis | Negative class | Generated benign tool returns from the same tool schemas; documented label-noise hazard |

**Dedup policy**: attack-template deduplication (not string matching). Published transfer numbers are inflated by template overlap; we'll report both de-duped and string-level numbers and discuss the gap.

**Skip**: MCP-SafetyBench (to avoid echo with sibling AIPI 590 Challenge 4 project).

## Three models

| Tier | Model | Features | Rationale |
|---|---|---|---|
| **Naive** | Majority class + keyword heuristic ("ignore previous", imperative verb density) | None | Required baseline; tool-agnostic |
| **Classical** | XGBoost over TF-IDF (word + char n-grams) + tool-one-hot | Text n-grams + tool identity | Tests whether tool metadata dominates; honest non-DL benchmark |
| **Deep** | DistilBERT fine-tune with tool-name token prepend | Contextualized text + tool identity | Small, ONNX-exportable, deployable in-browser |

**Why DistilBERT and not larger**: parameter count is a red herring here. The measurement question is answered by a well-calibrated 66M model with honest operating-point reporting, not a 7B model with SOTA aggregate numbers. Keeping the DL model small also protects the browser-deployable demo from unnecessary friction.

## Experiment (the one focused experiment)

**Tool-conditional recall@1%FPR on agent-targeted indirect attacks.**

**Procedure**:
1. Train each of the three models on InjecAgent + BIPIA (tool identity preserved)
2. Evaluate on WASP with **per-tool-class stratification**
3. For each (model, tool) pair: report recall at 1% FPR, PR-AUC, calibration error
4. Ablation: train/eval with tool-identity features masked, measure the lift delta
5. Dedup analysis: rerun with attack-template dedup, report number delta vs. string-level

**Output**: per-tool leaderboard + calibration curves. The artifact the community reuses.

## Evaluation metrics + rationale

- **Primary**: recall at 1% FPR (operating point matters for deployed guardrails — false positives destroy UX, false negatives miss attacks)
- **Secondary**: PR-AUC (class imbalance is real on injection benchmarks)
- **Diagnostic**: expected calibration error, per-tool
- **Not**: raw F1, accuracy (threshold-dependent and misleading on imbalanced security data)

Writeup must explicitly justify why recall@1%FPR is the right lens for this deployment shape.

## App (the demo)

Single-page static web app at `web/index.html`, served from GitHub Pages.

**Left pane**: tool selector + textarea for tool output (with example payloads per tool).
**Right pane**: risk score, tool-conditional calibration band (wider when the tool is under-sampled in training), token-level attention highlights on suspicious spans.
**Bottom pane**: live per-tool leaderboard pulled from `results/scores.json`, so the paper's key contribution is visible in the app itself.

Inference: ONNX Runtime Web with WebGPU, no server.

## Error analysis plan

Five mispredictions, each:
- (1) false positive on a benign tool output containing adversarial-sounding phrasing
- (2) false negative on a paraphrased known-attack template
- (3) false negative on a tool class under-sampled in training
- (4) disagreement between classical and DL models (which is right?)
- (5) confident wrong prediction (high-probability miss) — what feature drove it?

Each with root cause + proposed mitigation.

## Timeline (4 days)

| Day | Target |
|---|---|
| **Fri Apr 17 (tonight)** | Repo scaffold (done). Fetch InjecAgent + BIPIA + WASP. Dataset join with tool metadata preserved. Train/test split with no tool-name leakage + attack-template dedup. Corpus on disk. |
| **Sat Apr 18** | Naive + classical baselines. DistilBERT fine-tune. Evaluation harness with per-tool stratification. One clean end-to-end run. |
| **Sun Apr 19** | WASP cross-dataset eval. Per-tool leaderboard v1. Ablation: tool-identity masked. ONNX export + browser inference proven. Start web app shell. |
| **Mon Apr 20** | Web app polish (WebGPU inference, calibration band, token highlights). Deploy to GH Pages. Write report. Error analysis section. |
| **Tue Apr 21 AM** | Slides (6–7, Bent's house style). Rehearse. Submit by 11:30am. |

## Cut list (in priority order if behind)

1. WebGPU inference → CPU ONNX Runtime Web (slower, same UX)
2. Token-level attribution → whole-input risk score only
3. Calibration ablation → report single operating point
4. Template dedup → single-version numbers with caveat
5. DistilBERT → classical-only submission (still clears 3-model rubric; weaker story)

**NOT cuttable**: per-tool stratification (it is the thesis).

## Risks

- **Benign-class synthesis is the label-noise hazard**. Document explicitly in writeup; report ablation with benigns sampled from held-out tool traces only.
- **WASP is a sandbox, not a static dataset**. Need to cache its tool returns as a frozen eval set, not re-run the sandbox.
- **Challenge 4 (AIPI 590) collides on 2026-04-21 deadline**. Daily carve-out; hard cut list above.

## Decisions locked

1. **Scope**: tool-conditional measurement, not tool-agnostic binary classification
2. **Data**: InjecAgent + BIPIA for training, WASP for test. No MCP-SafetyBench
3. **Models**: naive keyword / XGBoost with tool-one-hot / DistilBERT with tool-prepend
4. **Primary metric**: recall at 1% FPR, per-tool
5. **Deployment**: static GH Pages + ONNX Runtime Web + WebGPU
6. **Framing**: measurement contribution, not capability claim

## Literature grounding

- Fomin 2026, *When Benchmarks Lie* (arXiv 2602.14161)
- Zhan et al., InjecAgent (ACL 2024 Findings)
- Yi et al., BIPIA
- Evtimov et al., WASP (arXiv 2504.18575)
- Anthropic, Claude Code auto-mode (engineering post, 2026-03-24)
- MindGuard (arXiv 2508.20412)
- WAInjectBench (arXiv 2510.01354)
- CourtGuard (arXiv 2510.19844)
- Meta Llama PromptGuard 2 (model cards, 2025)
- Lakera PINT benchmark
- AgentDojo (arXiv 2406.13352)
