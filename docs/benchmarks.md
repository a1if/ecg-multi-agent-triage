# Benchmarks

## Baseline: receiver inference, before any optimisation (2026-10-08)

Setup: RTX 5070 12 GB (Windows 11), Gemma 4 E4B, bitsandbytes NF4, Hugging Face Transformers `generate`, batch size
1, schema-constrained greedy decoding, no prefix cache. 15 windows from the bundled MIT-BIH test records (5 per size:
2 urgent, 2 priority, 1 routine by the rule) × 3 channels = 45 calls. Script: `backend/scripts/bench_inference.py`;
raw data: `backend/results/bench_baseline_5070_seed101.json`. Medians per cell (5 calls each).

**Cold start:** imports 7.9 s; model load + token table + warm-up 59.4 s; 9.1 GB GPU memory after load.

| Channel / beats | Prompt tokens | Time to first token | Time to decision (tier) | Full answer | Answer tokens | Decode tok/s | Energy per call | Peak GPU memory | Tier = rule |
|---|---|---|---|---|---|---|---|---|---|
| compact / 10 | 1,105 | 243 ms | 1.43 s | 14.5 s | 91 | 6.0 | 869 J | 9.3 GB | 3/5 |
| compact / 20 | 1,694 | 362 ms | 1.52 s | 14.2 s | 85 | 6.1 | 852 J | 9.4 GB | 2/5 |
| compact / 50 | 3,455 | 835 ms | 1.96 s | 13.7 s | 80 | 6.3 | 894 J | 9.7 GB | 3/5 |
| filtered / 10 | 749 | 204 ms | 1.37 s | 10.8 s | 64 | 6.1 | 629 J | 9.3 GB | 5/5 |
| filtered / 20 | 750 | 202 ms | 1.36 s | 9.9 s | 61 | 6.2 | 600 J | 9.3 GB | 5/5 |
| filtered / 50 | 751 | 191 ms | 1.30 s | 12.5 s | 79 | 6.3 | 741 J | 9.3 GB | 4/5 |
| adapter / 10 | 553 | 196 ms | 1.31 s | 9.9 s | 60 | 6.1 | 575 J | 9.2 GB | 4/5 |
| adapter / 20 | 593 | 191 ms | 1.29 s | 11.0 s | 67 | 6.1 | 633 J | 9.2 GB | 4/5 |
| adapter / 50 | 713 | 189 ms | 1.29 s | 8.4 s | 51 | 6.3 | 491 J | 9.3 GB | 3/5 |

### What it says

1. **Writing the answer is ~90% of the time.** Reading the prompt takes 0.19-0.84 s; the tier exists after 1.3-2.0 s;
   the justification takes the remaining 7-13 s at ~6 tokens/s. "Decide early, explain later" (design §6d, lever 2)
   cuts a ranking call from ~10-14 s to ~1.3-2.0 s, without changing the tier.
2. **Decoding is slow for this model size: ~6 tokens/s (~160 ms/token)**, the same as the paper's measurement. This
   points at the runtime (bitsandbytes 4-bit kernels + `generate` overhead), not the GPU, so levers 8-9 (faster
   runtime, FP8/bf16 on the 24 GB L4) are worth a spike — behind the accuracy gate.
3. **Prompt size matters little at batch 1**: compact text at 50 beats reads 3,455 tokens in 0.84 s vs 0.19 s for the
   adapter (4.4×), but that is under a second either way. The channel's cost advantage shows up in memory and under
   batching (the paper's serving results), not in single-call latency.
4. **Energy per call is ~0.5-0.9 kJ**, dominated by answer length; at full-answer length a 6-window review is ~4-5 kJ
   (~1.3 Wh).
5. **Agreement with the rule (small sample, 15 windows per channel, not an accuracy estimate):** filtered 14/15,
   adapter 11/15, compact 8/15, the paper's ordering. Compact text under-triaged 5 of its 6 urgent windows to
   priority, the failure the paper reports for long lists; adapter errors were 3 over-triages and 1 under-triage.
   **Every under-triage (7 calls) is the case the guardrail exists for:** the agent would reject it and ask for a
   resend over filtered text.

## Optimisation 1: decide early, explain later (2026-10-08)

Same machine, model, windows and channels as the baseline. The agent's reviews now stop at the tier (7 answer tokens:
the forced `{"urgency_tier":"` literal plus the tier word); the justification is generated only on request, with
those 7 tokens placed in the prompt. Raw data: `backend/results/bench_decide_explain_5070_seed101.json`.

| Channel | Decide (new review cost) | Baseline full answer | Speed-up | Energy per review: new vs baseline | Explain (on demand) |
|---|---|---|---|---|---|
| compact | 1.35 s | 14.2 s | 10.6× | 124 J vs 869 J | 13.2 s |
| filtered | 1.18 s | 10.8 s | 9.2× | 89 J vs 629 J | 9.2 s |
| adapter | 1.17 s | 9.9 s | 8.5× | 85 J vs 575 J | 8.5 s |

* **Tier identical in 45/45 calls** (required: the optimisation must not change any decision).
* All 45 calls together: **57.5 s instead of 510 s** of GPU time; median energy per review 90 J instead of 638 J.
* Decide + explain together (median 10.4 s) is slightly below the old full answer (11.0 s): the 7 forced tokens are
  read in one parallel pass instead of 7 decode steps. So explaining later never costs more than explaining always.
* **Explanation text identical in 39/45.** The 6 differences are floating-point divergence between reading the forced
  tokens in parallel and generating them one by one; once one token differs, greedy decoding follows a different path.
  One is a capital letter. The tier is never affected.

### Finding: adapter explanations are often not usable

Comparing the texts exposed a problem that is not about speed. Over the adapter channel the receiver chooses the
right tier but often writes a degenerate justification, in the baseline and in the new path alike, e.g.
*"Beat 12 of 20 is the most urgent based on the pattern of 3-15/25:19999999999999!!"* or
*"Second beat of 20:100: This is the second of 32:32:2032:urgent:20:20:20:2."* The adapter was trained to carry what
the tier needs, not what a sentence about specific beats needs. Text channels write coherent justifications.
Consequence for the design: **a channel that is good for deciding is not automatically good for explaining** — see
design §6d, "explanation channel".

## Signal quality: when the sender can be trusted (2026-10-08)

Script: the experiment in this section was run inline; the faults are `backend/src/ecg_agent/signal/stress.py` and
the expected outcomes are pinned in `backend/tests/test_quality.py`.

**Finding 1: the inherited quality index is blind to noise.** It penalises flat and clipped signal only; adding noise
*raised* it (record 100: 0.82 clean, 0.99 with noise) while the classifier degraded.

**Finding 2: moderate noise makes the classifier miss, not alarm.** Classifier on the annotated beats of the first
5 minutes, with white noise added; noise measure = share of beat-window energy above 40 Hz (median per record):

| SNR | Noise measure | Accuracy vs annotations | Note |
|---|---|---|---|
| clean | 0.000-0.019 | 0.97-1.00 | |
| 20 dB | 0.011-0.041 | 0.90-1.00 | record 233: abnormal predicted 19% vs 28% true |
| 10 dB | 0.095-0.190 | 0.59-0.99 | record 233: **3% called abnormal vs 28% true** (silent under-triage) |
| 5 dB | 0.23-0.38 | 0.39-0.64 | |
| 0 dB | 0.44-0.57 | 0.33-0.48 | |

Thresholds chosen from this: noise above 0.08 per window = unreadable (not triaged, manual review); recording
unreadable if more than half its beats are; 0.03-0.08 = warning. Residual risk, stated: 20 dB on record 233 loses
some ventricular beats below the warning line; clinician sign-off remains the control.

**Outcome per fault** (records 100 and 233, 5 min, detector path):

| Fault | Outcome |
|---|---|
| clean, inverted lead, baseline wander | passes, no warnings |
| noise bursts (30% of time) | passes; the 3-4 affected windows marked unreadable, never sent for triage |
| 36% flat (electrode off) | passes with "36% flat, not assessed" warning |
| noise 10 dB and worse, clipping, 54% flat | stopped: run status `unreadable`, no tier |

All seven bundled full (30-min) records pass cleanly in the sender's training setting.

Also found by the experiment: a clipped signal produced two detected peaks 166 ms apart (360 bpm), the event schema
rejected it and the exception crashed the intake loop. Fixed twice: the detector keeps no two beats closer than
200 ms, and a failed strategy is now a failed attempt, not a failed agent.

## System evaluation (2026-10-08)

Scripts: `backend/scripts/eval_system.py` (suites `invariants`, `stress`, `gemma`) and `backend/scripts/chaos_test.py`
(`load`, `chaos`). Raw results: `backend/results/eval_*.json`, `backend/results/chaos_*.json`.

**Invariants checked on every run:** I1 the run completes; I2 no finding ends below its screening tier; I3 every
readable urgent window is reviewed within budget (otherwise only urgent windows are); I4 review and attempt budgets
hold; I5 the overall tier is at least the highest readable screening tier; I6 the policy refused nothing.

| Suite | Scope | Result |
|---|---|---|
| Invariants under injected faults (CPU) | 7 records × 10 min × 3 modes × fault rate 0 / 0.3 / 1.0 (one-tier under-triage, plus 5% unparseable when faults are on) = 63 runs | **0 violations.** 343 faults injected, **343 rejected** and resent; at fault rate 1.0, 99 windows ended at their screening tier, flagged for human review. ~0.65 s per run |
| Signal faults (CPU) | 7 records × 10 scenarios = 70 runs | **70/70 expected outcomes**, 0 invariant violations. The noise thresholds were set on records 100 and 233 only; the other five records are held-out validation |
| Real Gemma (RTX 5070, adapter seed 101) | 7 records × 5 min, balanced mode, 6 reviews each; every finding explained; 12-question battery on record 233 | **0 violations.** 7/7 summaries grounded; 39/42 explanations by Gemma, 3 by the rule (filtered text disagreed); **12/12 questions answered as expected, 6/6 hostile ones refused**; median run 20.2 s incl. summary; model load 58 s |
| Load (API, offline receiver) | 8 runs of 10 min posted at once, concurrency limit 2 | 8/8 complete, never more than 2 running, 5.9 s total |
| Chaos (API + GPU service, two processes) | GPU service killed 3.8 s into a live 10-min run (after 2 windows) | Run **completed**: 2 answers from `gemma` before the kill, 8 from the labelled `offline-rule` after, summary from the template; none below screening |

Receiver verdicts in the Gemma suite (46 calls): adapter 18/18 agreed with screening (median 1.17 s), filtered 24/26
(2 under-triage), compact 0/2 (both under-triage; it is only used as the second escalation step). Every under-triage
was caught and resent. These counts are too small to rank channels and come from different windows (the router sends
abnormal-heavy windows to the adapter), so they do not contradict the paper's accuracy ordering.

**After promoting adapter seed 303** ([mlops.md](mlops.md#promoting-a-model-the-gate)), the real-Gemma suite was
re-run (`eval_gemma.json`; the seed 101 run is kept as `eval_gemma_seed101.json`). Same outcome on every safety and
quality measure: 0 violations, adapter 18/18 agreeing with screening, filtered 24/26, 39/42 explanations by Gemma,
7/7 summaries grounded, 12/12 questions as expected. The latency benchmarks were re-run too
(`bench_*_5070_wsl_seed303.json`), but on the same GPU inside the Linux container rather than natively on Windows
(native loading crashed that day in transformers' memory-mapped loader). The container is slower overall: the text
channels, which never touch the adapter, were 27-32% slower there as well, so the tables above (native, seed 101) stay
the latency reference. Like for like, nothing moved: the adapter's time to the tier decision relative to filtered text
is 0.98 (was 0.97), decide-early still gave the full answer's tier on 45/45 calls, and the adapter's tier matched the
rule on 14/15 benchmark windows (was 11/15).

Observed weakness: the question "Is there a run of abnormal beats near the start?" was routed to `evidence` in an
earlier run and to `unclear` here (safe, but less useful). The classifier is sensitive to the window list in its
prompt; giving it the time layout explicitly is on the task list.
