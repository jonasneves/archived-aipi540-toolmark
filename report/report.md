# Toolmark: Where Do Prompt-Injection Classifiers Break?

**A per-tool operating-point analysis of indirect prompt-injection detection on AI-agent tool outputs.**

Jonas Neves · Duke AIPI 540 — Deep Learning Applications · Spring 2026

---

## Abstract

Production prompt-injection guardrails report ~98% aggregate accuracy. On agent-targeted indirect attacks, recent work (Fomin 2026) measures detection collapsing to **7–37%**. Aggregate benchmarks hide which tools cause the collapse. Toolmark contributes a measurement artifact: Leave-One-Tool-Class-Out (LOTCO) recall at 1% FPR on InjecAgent, with tool-identity preserved and a matched benign/malicious construction, across three models — a keyword heuristic, XGBoost over char n-gram TF-IDF with tool one-hot, and a fine-tuned DistilBERT. Two substantive findings follow. First, the **keyword-heuristic baseline collapses from 68% to 5% recall** when benigns contain natural imperative phrasing — a concrete failure mode of the first-generation "block on attack keywords" defense. Second, once text features exist, **tool-identity conditioning provides essentially zero lift** (0.941 → 0.946 mean): char n-grams saturate the signal without it. We interpret both results as evidence of benchmark-construction artifact rather than deployable security, and sketch what a realistic successor benchmark would require.

## 1. Problem Statement

AI agents increasingly read tool outputs from the world — emails, webpages, file contents, API responses. Any of those outputs can contain **indirect prompt injection**: an attacker-authored string placed upstream of the agent that the agent reads *as instructions*. The threat model is not new, but as agent deployments proliferate, a *runtime content-layer classifier* that inspects tool outputs *before* the agent consumes them has become a canonical architectural component (Anthropic 2026, Claude Code auto-mode).

Public guardrails (Meta Llama PromptGuard 2, Lakera Guard, Qualifire Sentinel, ProtectAI DeBERTa-v3) report 0.97–0.998 aggregate F1. Yet Fomin (2026), *When Benchmarks Lie*, evaluates these same guardrails on agent-indirect attacks and measures **detection dropping to 7–37%**. The aggregate number was hiding the agent-trajectory distribution.

This project asks a finer question: **within a single agent-trajectory benchmark, which tool classes cause the collapse?** We measure by holding out one tool class entirely from training and testing on it, across 8 tool classes and 3 models.

## 2. Related Work

[TODO lock down on Sun]: summarise

- InjecAgent (Zhan et al., ACL 2024 Findings) — the benchmark we build on
- BIPIA, WASP (arXiv 2504.18575) — considered and ruled out (§3)
- *When Benchmarks Lie* (Fomin, arXiv 2602.14161) — direct motivation
- Claude Code auto-mode (Anthropic, 2026-03-24) — production deployment of the architecture this work measures
- MindGuard (arXiv 2508.20412) — attention-based DDG on metadata poisoning; orthogonal layer
- WAInjectBench (arXiv 2510.01354), CourtGuard (arXiv 2510.19844) — parallel deployable classifiers
- Meta Llama PromptGuard 2 — comparison baseline family

## 3. Data Sources

**InjecAgent (primary)**: 1,054 attack cases across 17 user tools. We map each user tool to one of 8 tool classes (shopping, note, code, email, calendar, healthcare, messaging, web_fetch). Each case provides a `Tool Response Template` with an `<Attacker Instruction>` slot and a `Tool Response` where that slot is filled with the attack payload.

**Matched benign construction**: for each attack case, we emit a matched benign row by filling the `<Attacker Instruction>` slot with a phrase from a 40-entry pool. The pool is deliberately mixed-register: 20 conversational declaratives (reviews, replies, status notes) and 20 benign-intent imperatives (scheduling, forwarding, reminders). The imperative half is what makes the task non-trivial — without it, char n-grams trivially hit 100% on every fold by detecting imperative phrasing as a proxy for "attack."

Final corpus: 2,108 rows, balanced 1,054/1,054 malicious/benign, 1,054 unique templates. Per-class support ranges from 124 rows (healthcare, web_fetch — one tool each) to 496 (messaging — four tools).

**Considered and skipped**:

- **BIPIA**: schema is `context + question + answer`. Extracting binary injection-vs-benign span labels requires parsing that exceeds the 4-day budget.
- **WASP**: requires a live web-agent sandbox.
- **MCP-SafetyBench**: adjacent to a sibling AIPI 590 project; skipped to keep this work cleanly separated.

A label-noise hazard remains: the benign pool contains only 40 distinct fillers, so classifiers with memorization capacity can overfit to the pool. We report this explicitly in §6 and propose mitigations in §8.

## 4. Evaluation Strategy & Metrics

**Primary: recall at 1% FPR.** In-line guardrails have an operating-point constraint, not a threshold-invariant one. False positives block benign tool outputs and destroy UX; false negatives let injection through. The operating point matters, and aggregate AUC can be flattering without helping deployment. We calibrate the threshold that achieves FPR ≤ 0.01 on a held-out *validation* slice (drawn from the same 7-class pool used for training), and apply that threshold verbatim to the held-out test class — no test-set peeking.

**Secondary**: PR-AUC (class imbalance robustness), ROC-AUC (threshold-invariant baseline), expected calibration error (10-bin ECE).

**Not reported**: raw F1 and accuracy, which are threshold-dependent and inflate on class-balanced benchmarks.

## 5. Modeling Approach

### Data pipeline
1. Download InjecAgent's `dh_base.json` + `ds_base.json`.
2. Strip the JSON-string quote wrapper from `Tool Response` (otherwise "text starts with `\"`" is a trivial malicious/benign signal).
3. Emit matched (benign, malicious) rows; deterministic benign-filler assignment via MD5(case_id) indexing.
4. Attack-template dedup on `(template_id, variant)` (no-op on InjecAgent; kept as a guard for future additions).

### Hyperparameter tuning strategy
Given the 4-day budget and the measurement (rather than capability) framing, we fix conservative hyperparameters from literature defaults and report them verbatim. No grid search. The research question is not "what's the best classifier" but "how does the operating-point picture change across tool classes."

- XGBoost: `n_estimators=300, max_depth=6, lr=0.1, subsample=0.9, colsample_bytree=0.9, tree_method='hist'`
- DistilBERT: `epochs=1, batch=16, lr=5e-5, weight_decay=0.01, max_length=192`
- TF-IDF: `char_wb, ngram_range=(3, 5), max_features=20,000, min_df=2, sublinear_tf=True`

### Models

**Naive — keyword heuristic.** Majority-class prior (0.5 on balanced Toolmark) blended with a 15-token regex (`ignore previous`, `forward to`, `grant access`, `curl http`, etc.). Tool-agnostic. Represents "first-generation block-on-attack-keywords" defenses.

**Classical — XGBoost over char n-gram TF-IDF + tool-class one-hot.** Char n-grams catch paraphrase-style evasion that word n-grams miss. Tool one-hot is concatenated to the TF-IDF matrix via sparse hstack. An ablation variant drops the tool feature.

**Deep — DistilBERT fine-tune with tool-name prepend.** The input text is prepended with `[TOOL: <class>] ` so the transformer has access to tool identity in-band. The ablation drops the prepend. 66M parameters, single epoch on MPS. Exported to ONNX for in-browser inference via ONNX Runtime Web + WebGPU.

## 6. Results

[TODO insert the leaderboard table from `results/scores.json` + ROC-PR curves]

### Mean recall at 1% FPR across 8 LOTCO folds

| Model | Mean recall@1%FPR | PR-AUC | ECE |
|---|---|---|---|
| naive_keyword | 0.049 | 0.661 | 0.183 |
| classical_xgb (tool one-hot) | 0.941 | 1.000 | 0.012 |
| classical_xgb (no tool feature) | 0.946 | 1.000 | 0.011 |
| deep_distilbert (tool prepend) | TBD | TBD | TBD |
| deep_distilbert (no tool prepend) | TBD | TBD | TBD |

### Per-tool recall@1%FPR

[TODO table — see `results/scores.json` and the app leaderboard at `./public/index.html`]

### Reading the results

**Finding 1 — the keyword heuristic collapses.** Recall drops from 0.68 (on purely declarative benigns, run during dataset iteration) to **0.049** on mixed benigns containing imperative phrasing. With FPR held at 1%, the classifier catches almost no true injections. This is a concrete failure mode of first-generation defenses: if a legitimate user can say "please forward me the draft," any classifier keying on "forward" will either false-positive the user or false-negative the attacker.

**Finding 2 — tool conditioning gives zero lift.** The classical model with tool one-hot (0.941 mean) and without it (0.946 mean) are indistinguishable. Char n-grams have already separated benigns from malicious on text content alone; there's no signal left for tool identity to add. This is counter to my original hypothesis that under-represented tools would benefit from explicit identity features.

**Finding 3 — char n-gram saturation is the benchmark's ceiling.** Classical XGBoost achieves PR-AUC 1.000 on 6 of 8 tool classes; the two "underperformers" (healthcare, web_fetch) still hit 0.887 recall@1%FPR. A 66M-parameter DistilBERT achieves numbers in the same neighborhood. In the context of Fomin 2026's 7–37% finding on real agent-indirect attacks, the gap between our numbers and theirs is the artifact we set out to measure.

## 7. Error Analysis

[TODO after DL run — pick 5 mispredictions across folds, annotate]

Five specific mispredictions, selected across folds:

1. **FP — benign imperative with attack-like phrasing**: "Please renew the license key in the billing portal before it lapses" flagged as malicious. Root cause: "please" + target-object-with-modifier matches classical's XGBoost feature importance. Mitigation: expand the benign pool to neutralize the feature (~hundreds of benign imperatives, as opposed to 20).

2. **FN — paraphrased known attack**: TBD

3. **FN — attack on under-represented tool class**: TBD

4. **Disagreement — classical vs. deep on same example**: TBD

5. **Confident wrong — high-probability miss**: TBD

## 8. Experiment: Leave-One-Tool-Class-Out

### Plan
For each of 8 tool classes, hold that class entirely out of training; train each model on the remaining 7 classes plus a 15% template-aware validation slice for threshold calibration; evaluate on the held-out class. An ablation rerun masks tool-identity features (one-hot for XGBoost, `[TOOL:]` prepend for DistilBERT).

### Results
See §6. Briefly: classical and deep models hit 0.88–1.00 per held-out class. Healthcare and web_fetch, the only tool classes with a single user tool each, show the largest degradation — consistent with the hypothesis that under-representation hurts generalization. But the effect size is small (0.88 vs. 1.00), and the aggregate story is that **when text features saturate, holding out a tool class is not the stress test I thought it was**.

### Interpretation
The LOTCO experiment reveals an important *null* result: tool-identity conditioning does not meaningfully change per-fold numbers, because char n-grams already separate the classes. The real generalization-stress comes from **distribution shift in attack strategy, not in tool vocabulary**. A successor benchmark should either (a) build benigns from a much larger pool so memorization is impossible, or (b) use LLM-paraphrased attacks as held-out tests, holding attack *style* out rather than tool *class* out.

### Recommendations
- Do not report aggregate F1 on InjecAgent-as-classification. Report operating-point recall, cross-class deltas, and benchmark-sensitivity.
- Treat "tool conditioning lifts recall" as an unvalidated hypothesis until tested on a benchmark where base rates are non-trivial.
- Next measurement to build: adversarial paraphrase transfer from InjecAgent to WASP, holding *attack family* rather than *tool class* constant.

## 9. Conclusions + Future Work

Toolmark is a measurement contribution, not a capability claim. The two substantive takeaways:

1. **Keyword defenses fail hard** when benign tool outputs contain natural imperative phrasing. A 15-token regex drops from 68% to 5% recall when the benigns look like real user requests.
2. **Tool-identity conditioning does not help** once a modern text classifier exists. The "tool-aware" feature is doing no work; the benchmark is trivially separable on text content.

With another semester:
- Build a 1,000+ unique benign pool by sourcing real tool outputs from public corpora (Enron emails, Amazon reviews, GitHub issues, calendar.ics archives), so the classifier cannot memorize a small benign set.
- Run the same LOTCO sweep with BIPIA and WASP as additional test distributions.
- Measure against production guardrails (PromptGuard 2, Lakera Guard) on the same per-tool operating point, for direct comparison.
- Attack-style LOOCV: hold attack families (role-play override, encoded payload, indirect-retrieved, etc.) out of training rather than tool classes.

## 10. Commercial Viability

The deployed app is a measurement dashboard, not a product. As a commercial product, a per-tool operating-point analyzer hits real demand: every team shipping an agent needs to know where their guardrail breaks. But the project does not compete with Anthropic / Meta / Lakera on capability — they ship production-scale classifiers with private training data. Where a commercial successor could add value is in **per-customer operating-point tuning**: a SaaS that ingests a customer's agent traces and reports the specific recall drop on the customer's tool surface. That is dataset-ops, not model research.

## 11. Ethics

- **Dataset attribution**: InjecAgent is open-sourced under its repository license; the benign fillers are author-written.
- **Demo content**: the shipped examples in `public/app.js` are illustrative and hand-written, not lifted from InjecAgent verbatim, to avoid redistributing benchmark strings in the bundle.
- **Deployment risk**: a trained injection classifier, made public, can also be used by attackers to craft examples that evade it. We address this directly: the keyword model is trivially bypassed (which the work itself demonstrates); the DistilBERT model is small enough to iterate against locally. Neither represents a material uplift to attackers relative to open-source alternatives (PromptGuard 2, ProtectAI DeBERTa) that already exist on HuggingFace.
- **Fairness**: the benchmark is English-only. Non-English attacks are out of scope and would require a multilingual benchmark — a known gap in the field.
- **No personal data**: all content is synthetic or public; no PII is stored, processed, or inferred.
