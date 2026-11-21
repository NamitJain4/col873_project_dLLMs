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
- On 23,674 valid BOLD prompts, LLaDA raw-prefix continuations are empty
  58.95% of the time versus 1.67% in instructed chat. Dream's instructed-chat
  continuations are empty 21.04% of the time; 95.42% have at most five words
  (including empty). Matched-prompt source-row bootstrap intervals for chat
  minus raw empty rate are LLaDA −57.28 percentage points [−58.14, −56.45] and
  Dream +14.18 points [+13.31, +15.06]. This motivates a
  generation-budget/EOS probe.
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
| `bbq_explore_132.jsonl` | `360a33d01697809960586c3785e57edfd678849fdf391159cfec952c6a1d3d1f` |

The BOLD baseline is the existing 48-token/24-step instructed-chat run on the
same 60 IDs. New conditions for LLaDA and Dream are 48/48 steps, 96/96
length+steps, and 96/96 with an explicit minimum-20-word instruction. For
LLaDA raw prefix, one condition retains 48/24 while deprioritizing EOS token
confidence as motivated by the LLaDA paper's EOS discussion. Outputs include
first-token EOS and output-length diagnostics. The LLaDA raw-prefix baseline
uses the same 60 IDs from the existing full run.

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

## Submitted PBS jobs

| Job | ID | Output |
| --- | --- | --- |
| Corrected Dream CrowS-Pairs and StereoSet | `193695.pragya` | `results/exploration/*dream_aligned_v2.jsonl` |
| Corrected Dream full/Goldfish LoRA, MIA, utility | `193696.pragya` | `checkpoints/*dream_aligned_v2`, `results/exploration/mia_*` and `utility_*` |
| BOLD generation interventions | `193709.pragya` | `results/exploration/bold_*` |
| BBQ generation interventions | `193710.pragya` | `results/exploration/bbq_*` |
| Dependent CPU validation, corrected summaries, notebook render | `193743.pragya` | `results/summary/exploration_*.tsv`, corrected headline CSV, executed notebook |

The four model jobs use offline cached checkpoints, the same pinned revisions
as the baseline, one GPU each, and the course PBS project/account. They append
durable per-example outputs and resume after interruption. The new notebook is
`notebooks/divergence_analysis.ipynb`; regenerate and execute it with
`../col873_ass2/.venv-gpu/bin/python scripts/render_divergence_notebook.py`.
The final CPU job requires all four GPU jobs to succeed (`afterok`).

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
