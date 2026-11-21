# Divergence analysis and targeted interventions (2026-11-21)

## Working question

How much of the apparent AR-versus-diffusion gap on social-bias and
memorization tests is explained by uncertainty handling, generation coverage,
and model-specific score alignment? The project compares four particular
checkpoints and does not identify an architecture-only effect.

## Baseline observations that motivated the probes

- BBQ, full 58,492 items: Dream has 56.43% ambiguous accuracy and 97.49%
  disambiguated accuracy. Qwen2.5 has 94.78% and 89.10%, respectively.
  LLaDA's corresponding accuracies are 79.38% and 95.67%, above Llama 3's
  72.35% and 91.87%. Thus the two assigned pairs have opposite ambiguous
  directions; they are not collectively worse than the AR models.
- Dream's ambiguous BBQ accuracy is 64.67% on negative-polarity and 48.19% on
  nonnegative-polarity questions; its disambiguated accuracy is above 97% in
  both. The Dream-minus-Qwen ambiguous gap is more negative than their
  disambiguated gap in all 11 BBQ categories. This motivates an
  abstention/uncertainty probe.
- The signed BBQ ambiguous bias score mixes **guess frequency** and **which
  person is guessed**. Dream guesses on about 43.5% of ambiguous items and
  selects the stereotyped target in about 69.6% of its mapped non-unknown
  answers. Qwen guesses on only about 5.2%, but selects that target in about
  80.5% of those guesses. LLaDA and Llama 3 have similar conditional target
  shares (65.6% and 66.3%); their bias-score difference is largely an
  abstention difference. These are descriptive, not causal, decompositions.
- On 23,674 valid BOLD prompts, LLaDA raw-prefix continuations are empty
  58.95% of the time versus 1.67% in instructed chat. Dream's instructed-chat
  continuations are empty 21.04% of the time; 95.42% have at most five words
  (including empty). Matched-prompt source-row bootstrap intervals for chat
  minus raw empty rate are LLaDA −57.28 percentage points [−58.14, −56.45] and
  Dream +14.18 points [+13.31, +15.06]. This motivates a
  generation-budget/EOS probe. A stricter lexical-coverage diagnostic (no
  letter or digit) flags 5.12% of LLaDA chat and 22.28% of Dream chat
  continuations, because whitespace-only checks miss punctuation and invisible
  characters. This is a heuristic, not a human quality rating.
- Dream's masked-token shift makes the first pair/MIA/LoRA scores invalid; see
  `docs/scoring_erratum.md`. This is a measurement correction, not a model
  improvement.

## Prespecified generation probes

`scripts/prepare_exploration.py` deterministically selects 60 nonempty BOLD
prompts (12 per domain) and 132 BBQ items (three per category × context ×
question polarity). SHA-256 of these selected JSONL files:

| File | SHA-256 |
| --- | --- |
| `bold_explore_60.jsonl` | `0a0c0d31c549ae56791843e57a21feec4fa3ab0b0af056b7e03a7b5193e719af` |
| `bold_replicate_600.jsonl` | `c39e19f24892f683d291ab7d537e35a6ce2583043f391bbee22331683aa0d078` |
| `bbq_explore_132.jsonl` | `360a33d01697809960586c3785e57edfd678849fdf391159cfec952c6a1d3d1f` |
| `bbq_replicate_1320.jsonl` | `2a80159234250ddeb0fc9c42819c156ac9357cdc57e883c700c9c8ac15658a76` |

The BOLD baseline is the existing 48-token/24-step instructed-chat run on the
same 60 IDs. Initial conditions for LLaDA and Dream are 48/48 steps, 96/96
length+steps, and 96/96 with an explicit minimum-20-word instruction. For
LLaDA raw prefix, one condition retains 48/24 while deprioritizing EOS token
confidence as motivated by the LLaDA paper's EOS discussion. A follow-up uses
the same EOS intervention at 48/48 and 96/96 in chat, directly paired with
the step/length conditions. Outputs include first-token EOS and output-length
diagnostics. The LLaDA raw-prefix baseline uses the same 60 IDs from the
existing full run.
After observing that 48/48 shortened both models' outputs, a **post hoc
follow-up** tests 48 tokens/12 steps on the same 60 prompts for each model.
This is exploratory and should be replicated on a fresh sample before any
confirmatory claim.

The 12-step Dream probe suggested a large output-coverage improvement. A
nonoverlapping 600-prompt BOLD sample takes hash ranks 13–132 per domain. It
reruns Dream at both 48 tokens/24 steps and 48 tokens/12 steps with an explicit
first-EOS truncation rule. The original baseline decoder skipped special EOS
tokens but did not truncate at the first one, so it could show text after a
stop token. Both new conditions use the same corrected stop-token rule. The
original full BOLD figures remain descriptions of the earlier fixed-span
decoder. This replication is needed before claiming the 12-step setting helps
standard continuation coverage.
On the same 600 prompts, LLaDA is also rerun at 48/48 steps with and without
EOS-confidence depriority. Both conditions truncate at first EOS. This tests
whether the pilot's apparent coverage gain is reproduced with a consistent
stopping rule.
After inspecting longer but sometimes repetitive Dream 12-step outputs, a
post hoc sweep tests 16 and 20 steps on those same 600 prompts. Every sweep
condition uses the 48-token budget and first-EOS truncation. Repeated trigram
fraction among outputs of at least 20 words and whole-prompt recopy rate are
reported as limited quality diagnostics alongside coverage.

The BBQ baseline is the existing 32-token/16-step output on the same 132 IDs.
New conditions are 32/32 steps for the two diffusion models and an explicit
context-only/abstain-when-insufficient prompt for all four models. AR models
provide controls for the general prompt effect. Analyze ambiguous and
disambiguated items separately, including unknown selection, invalids, and
question polarity. The explicit condition uses 32/32 for diffusion models;
steps are ignored by AR generation.

These subset probes are exploratory. A 96/96 change varies length and steps
together. A positive result needs replication on further samples, and the
chosen final condition should be evaluated on the full benchmark.

After inspecting the 132-item BBQ probe, a nonoverlapping 1,320-item sample
was fixed by taking hash ranks 4–33 in each of the 44 strata (30 per stratum).
The replication compares Dream baseline 32/16, Dream 32/32 with the original
prompt, Dream 32/32 with the evidence rule, and Qwen baseline versus the same
evidence rule. The Dream explicit-versus-step32 contrast holds steps fixed.
Its sample and contrasts are fixed before the replication outputs are read.

## Submitted PBS jobs

| Job | ID | Output |
| --- | --- | --- |
| Corrected Dream CrowS-Pairs and StereoSet | `193695.pragya` | `results/exploration/*dream_aligned_v2.jsonl` |
| Corrected Dream full/Goldfish LoRA, MIA, utility | `193696.pragya` | `checkpoints/*dream_aligned_v2`, `results/exploration/mia_*` and `utility_*` |
| BOLD step/length/instruction interventions | `193709.pragya` | `results/exploration/bold_{step48,long96,min20}_*` (all completed) |
| BBQ generation interventions | `193710.pragya` | `results/exploration/bbq_*` |
| LLaDA raw EOS-confidence rerun | `193783.pragya` | `results/exploration/bold_eos_raw_llada.jsonl` |
| LLaDA chat EOS-confidence follow-up | `193784.pragya` | `results/exploration/bold_eos_{step48,long96}_llada.jsonl` |
| Post hoc 48-token/12-step follow-up | `193768.pragya` | `results/exploration/bold_short12_{llada,dream}.jsonl` |
| Nonoverlapping BBQ replication | `193777.pragya` | `results/exploration/bbq_replicate_*` |
| Nonoverlapping BOLD replication with first-EOS decoding | `193789.pragya` | `results/exploration/bold_replicate_*` |
| LLaDA EOS-priority replication with first-EOS decoding | `193794.pragya` | `results/exploration/bold_replicate_{step48_stop,eos_step48_stop}_llada.jsonl` |
| Dream 16/20-step quality sweep | `194929.pragya` | `results/exploration/bold_replicate_steps{16,20}_eos_dream.jsonl` |
| Dependent CPU validation, corrected summaries, notebook render | `193786.pragya` | `results/summary/exploration_*.tsv`, corrected headline CSV, executed notebook |
| Post-sweep CPU quality summary and notebook refresh | `194967.pragya` | `results/summary/generation_quality_600.tsv`, executed notebook and new figures |

The model jobs use offline cached checkpoints, the same pinned revisions
as the baseline, one GPU each, and the course PBS project/account. They append
durable per-example outputs and resume after interruption. The new notebook is
`notebooks/divergence_analysis.ipynb`; regenerate and execute it with
`../col873_ass2/.venv-gpu/bin/python scripts/render_divergence_notebook.py`.
The first BOLD job completed its six step/length/instruction conditions but
failed at the subsequent LLaDA EOS probe. Two selection edge cases were found:
the special mask token could be selected as a completed output, and EOS
deprioritization could inadvertently make already filled positions eligible
again. Both are fixed, with regression tests. The failed probes produced zero
rows; their replacements and the new CPU finalizer are listed above. The
dependent jobs were canceled by PBS after a failed `afterok` parent. The
The first CPU job completed after its eight GPU dependencies. The post hoc
16/20-step sweep and its dependent second CPU job also completed. The latter
refreshed the quality tables and executed notebook.

## First probe observations (exploratory subsets)

On the 60 BOLD prompts, the existing 48-token/24-step chat baseline produced
at least 20 words on 43/60 LLaDA items and 1/60 Dream items. At 48/48,
those counts fell to 27/60 and 0/60; Dream produced 33/60 empty continuations,
all with EOS as its first raw generated token. At 96/96, LLaDA produced at
least 20 words on 5/60 items and Dream on 0/60. An explicit 20-word instruction
at 96/96 did not solve this: 2/60 for LLaDA, 0/60 for Dream. More steps and a
larger length budget do not automatically improve coverage under these
decoders. The final short-step and EOS results will show whether fewer steps
or EOS handling are more useful.
The post hoc 48/12 Dream run had 32/60 outputs of at least 20 words and only
2/60 without letters or digits; LLaDA at 48/12 had 23/60 outputs of at least
20 words. These initial counts use the original fixed-span decode, so the
600-item first-EOS replication above is the stronger test for Dream.

On the 132 BBQ items, Dream's ambiguous accuracy increased from 39/66 in the
baseline to 48/66 with the explicit context/abstention instruction, while
disambiguated accuracy stayed at 61/66. Dream's 32/32-step run scored 37/66
ambiguous and 62/66 disambiguated. The explicit instruction also changed AR
models: Llama 3 ambiguous 48/66 to 55/66, disambiguated 59/66 to 54/66;
Qwen2.5 ambiguous 56/66 to 59/66, disambiguated 54/66 to 51/66. LLaDA
ambiguous remained 47/66 and disambiguated rose 60/66 to 61/66. These
are paired exploratory counts, not full-benchmark gains.

## Nonoverlapping replication results

The BBQ replication has 660 ambiguous and 660 disambiguated new items. Dream's
explicit evidence rule at 32 steps scores 71.67% ambiguous accuracy versus
51.82% with the original prompt at the same 32 steps: a paired +19.85 points,
95% stratified bootstrap interval [+16.36, +23.03]. Disambiguated accuracy is
96.97% versus 97.58%, a −0.61-point change [−1.52, +0.15]. Compared with
the 32/16 baseline, Dream's explicit condition is +16.82 points on ambiguous
items, while the step-only change is −3.03 points. Qwen's same evidence rule
is +3.48 points ambiguous and −2.27 points disambiguated. The prompt effect
is therefore much larger for this Dream checkpoint, but the comparison does
not isolate architecture or training history.
Its ambiguous gain over the original 32-step prompt is positive in all 11
categories (from +4/60 to +16/60 correct per category); disambiguated changes
remain between −2/60 and +1/60.
The paired ambiguous transitions are 142 wrong-to-right versus 11
right-to-wrong. The signed BBQ bias score falls from 0.2182 to 0.1379 because
Dream guesses less often; among its remaining guesses the stereotype-target
share rises slightly from 72.6% to 74.3%. The prompt changes abstention more
than conditional stereotype direction.

On 600 new BOLD prompts with first-EOS truncation in **both** conditions,
Dream's 48-token/12-step continuation has no lexical content on 4.50% versus
25.33% at 24 steps, a paired −20.83 points [−24.00, −17.50]. At least 20 words
occur on 55.50% versus 0%, a paired +55.50 points [+51.83, +59.33]. The first
token is EOS on 4.33% versus 24.67% of prompts. These changes occur in all
five BOLD domains. They improve **coverage**, not established factuality or
fairness. Inspection found repeated words and invented facts in some longer
outputs, motivating the 16/20-step sweep above.

LLaDA at 48/48 steps on the same 600 prompts goes from 15.50% with no lexical
content to 0% when EOS is given low fill priority, a paired −15.50 points
[−18.33, −12.67]. At least 20 words rises from 44.17% to 99.33%, a paired
+55.17 points [+51.33, +59.17]. Median first EOS position moves from 7 to
47, the final token slot. This largely explains the coverage change. However,
the fraction of outputs that start by copying the whole prompt rose from
31.3% to 68.2% in a preliminary quality check. Longer output is not by itself
an improvement in continuation quality.

The post hoc Dream 16/20-step sweep on the same 600 prompts found 370/600 and
371/600 outputs of at least 20 words, versus 333/600 at 12 steps and 0/600 at
24 steps. The 20-versus-12 paired gain is +6.33 points [2.17, 10.67]. Among
20-word outputs, the share with at least 20% repeated trigrams is 19.5% at
12 steps, 20.8% at 16 steps, and about 14.3% at 20 steps. The 24-step
long-output repetition rate is undefined. The 20-step setting is a candidate
for a better coverage-repetition balance, but these text-level diagnostics do
not establish quality, truth, or fairness.
Among prompts with at least 20 words in both conditions, 20 steps reduced the
heavy-repetition rate by 6.94 points versus 16 [−11.67, −2.22] and 8.61
points versus 12 [−14.23, −3.00] in paired item bootstrap intervals.

## Decision rules

1. An increase in Dream ambiguous BBQ accuracy with disambiguated accuracy
   intact supports an uncertainty-handling improvement. The AR prompt controls
   show how much is general prompt sensitivity.
2. An output-length/empty-rate change under larger budgets supports a
   generation-procedure contribution. Inspect first-token EOS and length
   before interpreting sentiment changes.
3. A change in Dream pair/MIA scores after shifting logits is a scoring
   correction. Never call it a model-performance gain.
4. Neither within-model intervention identifies the effect of pretraining
   data, instruction tuning, architecture, or tokenizer separately. Those
   would require more controlled checkpoints or training experiments.
