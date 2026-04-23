# Toolmark: When Does Deep Learning Actually Help on Prompt-Injection Detection?

**A per-tool operating-point analysis of indirect prompt-injection detection on AI-agent tool outputs.**

Jonas Neves · Duke AIPI 540 — Deep Learning Applications · Spring 2026

---

## Abstract

Production prompt-injection guardrails report ~98% aggregate accuracy; Fomin (2026) finds that on agent-targeted indirect attacks, detection collapses to 7–37%. Toolmark contributes a measurement artifact under a cleaner, reproducible construction: Leave-One-Tool-Class-Out (LOTCO) recall at 1% FPR on InjecAgent, with a matched-pair benign/malicious construction, across three models — a keyword heuristic, XGBoost over char n-gram TF-IDF with tool one-hot, and a fine-tuned DistilBERT with tool-class prepend. Three findings. **(1)** The keyword-heuristic baseline collapses from 68% to **5%** recall when benigns contain natural imperative phrasing — a concrete failure mode of first-generation "block on attack keywords" defenses. **(2)** DistilBERT beats XGBoost by **~4 points** mean (0.990 vs. 0.946), but the gap is not uniform: DL wins on natural-language tool outputs (email, code, note, messaging) while classical matches or beats DL on structured-JSON outputs (calendar, shopping). **(3)** Tool-identity conditioning is a **null result** for both models — adding the tool feature changes mean recall by less than 0.5 points and sometimes hurts per-class. Text features saturate the signal; the tool-as-feature hypothesis does not earn its complexity.

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

### Mean recall at 1% FPR across 8 LOTCO folds

| Model | Mean recall@1%FPR | PR-AUC | ECE |
|---|---|---|---|
| naive_keyword | 0.049 | 0.661 | 0.183 |
| classical_xgb (tool one-hot) | 0.941 | 1.000 | 0.012 |
| classical_xgb (no tool feature) | 0.946 | 1.000 | 0.011 |
| **deep_distilbert (tool prepend)** | **0.986** | **1.000** | **0.031** |
| deep_distilbert (no tool prepend) | 0.990 | 1.000 | 0.032 |

### Per-tool recall@1%FPR

| Tool class | Support | Naive | Classical | XGBoost − | DistilBERT | DistilBERT − |
|---|---:|---:|---:|---:|---:|---:|
| calendar | 248 | 0.016 | **1.000** | 1.000 | 0.960 | 0.976 |
| code | 372 | 0.016 | 0.984 | 0.984 | **1.000** | 1.000 |
| email | 248 | 0.000 | 0.903 | 0.911 | **1.000** | 1.000 |
| healthcare | 124 | 0.032 | 0.887 | 0.903 | **0.984** | 0.984 |
| messaging | 496 | 0.012 | 0.956 | 0.960 | **1.000** | 1.000 |
| note | 248 | 0.008 | 0.952 | 0.952 | **1.000** | 1.000 |
| shopping | 248 | 0.000 | **0.960** | 0.952 | 0.944 | 0.992 |
| web_fetch | 124 | 0.306 | 0.887 | 0.903 | **1.000** | 0.968 |

*Bold = best recall in the row. "−" columns are the ablations without the tool feature.*

### Reading the results

**Finding 1 — the keyword heuristic collapses.** Recall drops from 0.68 (measured during dataset iteration on purely declarative benigns) to **0.049** on mixed benigns that contain imperative phrasing. At FPR ≤ 1%, the classifier catches almost no true injections. This is a concrete failure mode of first-generation defenses: if a legitimate user says "please forward me the draft," a classifier keying on "forward" must either false-positive the user or false-negative the attacker. There is no threshold that cleanly separates intent from surface form.

**Finding 2 — DistilBERT beats XGBoost by ~4 points, but the gap is non-uniform.** DL mean recall is 0.986 vs. 0.941 for XGBoost (+4.5 points). Per-tool, the picture is sharper: DL wins on natural-language tool outputs (email, code, note, messaging, healthcare, web_fetch) and classical wins on structured-JSON outputs (calendar, shopping). The natural-vs-structured split is intuitive: tree-based classifiers over char n-grams exploit exact structural tokens in JSON payloads — `'product_details'`, `'GoogleCalendarEventID'` — that DistilBERT's subword tokenization dilutes into context. Conversely, on free-form email or note text, contextual representations pay off.

**Finding 3 — tool-identity conditioning is a null result.** Adding a tool-class one-hot to XGBoost: +0.5 points mean (0.946 vs. 0.941). Adding a `[TOOL: <class>]` prepend to DistilBERT: +0.4 points in the **wrong** direction (0.986 vs. 0.990). Across 16 ablation cells, the feature changes recall by > 2 points in only three (shopping-deep: -0.048, web_fetch-deep: +0.032, healthcare-xgb: -0.016), and those deltas average out in aggregate. The original "tool-aware is better than tool-agnostic" hypothesis is not what this data says. Once text features exist, tool identity is redundant.

**Reading across the three findings together.** The benchmark does not stress-test tool identity because it does not need to: the text content of an indirect-injection attack is so structurally different from benign tool output that any classifier with reasonable representation capacity separates them cleanly. Fomin 2026's 7–37% finding on real agent-indirect attacks must therefore be attributable to (a) adversarial attack-style distribution shift that InjecAgent does not capture, or (b) benign distributions in the wild that are much messier than the 40-entry filler pool we could construct in four days.

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
The LOTCO experiment reveals two stable signals and one null. **Stable:** a 4.5-point DistilBERT lift over XGBoost mean, and a cleaner split by tool-output format than by tool identity (structured-JSON vs. natural-language). **Null:** tool-identity conditioning does not meaningfully change per-fold numbers, in either direction, for either model. The real generalization stress is distribution shift in *attack strategy*, not in *tool vocabulary*. A successor benchmark should either (a) build benigns from a much larger, real-world-sourced pool so memorization is impossible, or (b) LOOCV on attack *style* (role-play override, encoded payload, indirect-retrieved, multi-turn) rather than tool *class*.

### Recommendations
- Do not report aggregate F1 on InjecAgent-as-classification. Report operating-point recall, per-tool deltas, and benchmark-sensitivity.
- When choosing between XGBoost and DistilBERT for an injection-detection deployment, the decision should follow tool-output *format*: classical for structured JSON, deep for natural language. The 4-point aggregate gap is an average across two opposite effects, not a uniform win.
- Treat "tool conditioning lifts recall" as an unvalidated hypothesis. The feature earned no complexity on this benchmark.

## 9. Conclusions + Future Work

Toolmark is a measurement contribution, not a capability claim. Three substantive takeaways:

1. **Keyword defenses fail hard** when benign tool outputs contain natural imperative phrasing. A 15-token regex drops from 68% to 5% recall when benigns look like real user requests.
2. **DistilBERT beats XGBoost by ~4 points mean, but the gap is non-uniform**: DL wins on natural-language tool outputs (email, code, note, messaging); classical matches or beats DL on structured-JSON outputs (calendar, shopping). Deployment should pick classifier by tool-output format, not by absolute accuracy.
3. **Tool-identity conditioning is a null result**. Neither XGBoost's one-hot nor DistilBERT's `[TOOL:]` prepend earned its complexity. Text features saturate the signal; identity features are redundant.

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
