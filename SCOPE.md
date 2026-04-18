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
| **InjecAgent** (ACL 2024) | Primary population | 1,054 cases × 17 user tools × 8 tool classes; matched (benign, malicious) pairs synthesized per case |
| Benign synthesis | Negative class | `Tool Response Template` with `<Attacker Instruction>` slot filled from a deterministic 30-phrase pool; label-noise hazard documented |

**Tool-class taxonomy (8 classes)**: shopping, note, code, email, calendar, healthcare, messaging, web_fetch. See `scripts/make_dataset.py` for the per-tool mapping.

**Built dataset**: 2,108 rows, perfectly balanced (1,054 malicious + 1,054 benign), 1,054 unique templates. Class counts per tool_class range from 124 (healthcare, web_fetch — one tool each) to 496 (messaging — four tools). The under-represented classes become the hardest LOTCO targets.

**Dedup policy**: attack-template deduplication on `(template_id, variant)`. InjecAgent contained no cross-split duplicates (dedup no-op); kept as a guard for future data additions.

**Skipped, with reasons documented in the report**:

- **BIPIA**: schema is `context + question + answer` (task benchmark, not span-labeled injection). Span-level label extraction exceeds the 4-day budget.
- **WASP**: requires a live web-agent sandbox for trace generation.
- **MCP-SafetyBench**: avoided to prevent echo with the sibling AIPI 590 Challenge 4 project.

## Three models

| Tier | Model | Features | Rationale |
|---|---|---|---|
| **Naive** | Majority class + keyword heuristic ("ignore previous", imperative verb density) | None | Required baseline; tool-agnostic |
| **Classical** | XGBoost over TF-IDF (word + char n-grams) + tool-one-hot | Text n-grams + tool identity | Tests whether tool metadata dominates; honest non-DL benchmark |
| **Deep** | DistilBERT fine-tune with tool-name token prepend | Contextualized text + tool identity | Small, ONNX-exportable, deployable in-browser |

**Why DistilBERT and not larger**: parameter count is a red herring here. The measurement question is answered by a well-calibrated 66M model with honest operating-point reporting, not a 7B model with SOTA aggregate numbers. Keeping the DL model small also protects the browser-deployable demo from unnecessary friction.

## Experiment (the one focused experiment)

**Leave-One-Tool-Class-Out (LOTCO) recall@1%FPR on InjecAgent.**

A finer-grained version of *When Benchmarks Lie* (Fomin 2026)'s Leave-One-Dataset-Out: within a single benchmark, we localize *which tool classes* collapse when held out of training vs. which generalize. Directly answers "which tools cause production classifiers to drop from ~98% aggregate to 7–37% on agent-indirect attacks."

**Procedure**:
1. For each of 8 tool classes, hold it out of training; train on the remaining 7 classes
2. Train each of the three models (naive, classical, DL) on the 7-class pool
3. Evaluate on the held-out class: recall at 1% FPR, PR-AUC, expected calibration error
4. Ablation: rerun the classical model with tool-identity features masked; measure the lift delta
5. Compare held-out performance to a matched in-distribution split

**Output**: 3 models × 8 tool classes × 3 metrics leaderboard + per-class calibration curves. The artifact the reader takes away.

## Evaluation metrics + rationale

- **Primary**: recall at 1% FPR (operating point matters for deployed guardrails — false positives destroy UX, false negatives miss attacks)
- **Secondary**: PR-AUC (class imbalance is real on injection benchmarks)
- **Diagnostic**: expected calibration error, per-tool
- **Not**: raw F1, accuracy (threshold-dependent and misleading on imbalanced security data)

Writeup must explicitly justify why recall@1%FPR is the right lens for this deployment shape.

## App (the demo)

Single-page static web app at `public/index.html` (served from GitHub Pages via the `docs → public` symlink).

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
| **Fri Apr 17** | Repo scaffold + GH Pages live (done). |
| **Sat Apr 18** | Dataset built (done). Naive + classical baselines. DistilBERT fine-tune. Eval harness with LOTCO stratification. One clean end-to-end run. |
| **Sun Apr 19** | LOTCO evaluation over 8 tool classes × 3 models. Per-class leaderboard v1. Tool-identity ablation. ONNX export + browser inference proven. Start web app shell. |
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
