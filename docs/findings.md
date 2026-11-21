# Research memo: why the assigned diffusion models diverge

This is an experiment memo for the team, not the course report. The executed
plots and underlying calculations are in
[`notebooks/divergence_analysis.ipynb`](../notebooks/divergence_analysis.ipynb).
All confidence intervals below are 95% bootstrap intervals on the specified
benchmark extract. They measure sampling variation among these items, not
uncertainty from checkpoint training or model selection.

## Main question

**How much of the observed AR-versus-diffusion difference is explained by
missing-evidence decisions, stop-token behavior and decoding schedule, and
model-specific masked-token score alignment?** The project compares four
particular checkpoints. LLaDA and Dream also differ in pretraining history,
instruction tuning, tokenizers, and decoding code, so these comparisons cannot
identify a pure diffusion-architecture effect.

## What the complete baselines establish

All four assigned models completed the 58,492-item BBQ baseline and the
23,674-valid-prompt BOLD baseline. Dream versus Qwen2.5 changes sign across
BBQ conditions: ambiguous accuracy is 56.43% versus 94.78%, while
disambiguated accuracy is 97.49% versus 89.10%. LLaDA versus Llama 3 is
79.38% versus 72.35% ambiguous and 95.67% versus 91.87% disambiguated. The
diffusion models are therefore **not collectively worse** on BBQ. Dream's
ambiguous deficit is present in every category relative to its disambiguated
comparison, which points to a missing-evidence decision problem.

The BBQ signed bias score needs decomposition. On ambiguous items, Dream
guesses a person about 43.5% of the time and Qwen about 5.2%; among mapped
guesses, the stereotype-target shares are about 69.6% and 80.5%, respectively.
Thus Dream's worse signed score is driven largely by guessing more often, not
by a larger stereotype share conditional on guessing. This is a descriptive
decomposition, not a proof about its internal reasoning.

On BOLD, the original full decoder's raw versus instructed-chat condition
strongly changes coverage. LLaDA's raw continuation is empty 58.95% of the
time versus 1.67% in chat. Dream's instructed-chat continuation is empty
21.04%, and 95.42% of its chat outputs contain at most five words, including
empty outputs. The original decoder removed EOS special tokens without
truncating at the first EOS, so its visible output can include text after a
stop token. The new paired BOLD replications apply first-EOS truncation in
both conditions.

The original Dream CrowS-Pairs, StereoSet, MIA, Goldfish and utility scores
were invalid because Dream predicts a masked token from the preceding logit
position. Corrected runs are complete. Dream's corrected CrowS-Pairs
stereotype preference is 0.6320 and StereoSet preference is 0.6481; Qwen's
corresponding baseline preferences are 0.6174 and 0.6558. Those mixed
directions do not support a general pairwise bias ranking. Corrected Dream
full-adapter MIA AUROC is 0.5291; its Goldfish variant is 0.5245, paired
change −0.0046 with interval [−0.0133, +0.0041]. The Goldfish difference is
inconclusive on this one-seed synthetic experiment. The score correction is
a measurement fix, not an improvement in Dream itself.

## Interventions with independent item samples

**Dream evidence rule, BBQ.** The first 132-item exploratory probe suggested
that an explicit instruction to select the unknown answer when the context
does not establish an answer would help Dream. On 1,320 different items, the
32-step Dream run scored 71.67% ambiguous accuracy with that rule versus
51.82% with the original prompt at the same 32 steps: paired +19.85
percentage points [16.36, 23.03]. Disambiguated accuracy changed from 97.58%
to 96.97%, −0.61 points [−1.52, 0.15]. Qwen also gained 3.48 points on
ambiguous items under the rule but lost 2.27 points on disambiguated items.
The rule is a useful **checkpoint-specific** intervention for Dream's
missing-evidence behavior; its effect is larger than Qwen's here. More
diffusion steps alone reduced Dream ambiguous accuracy by 3.03 points versus
its 16-step baseline. A future full-BBQ prompt run would establish the exact
complete-extract gain.
The explicit-versus-original 32-step ambiguous gain is positive in all 11 BBQ
categories on the replication sample (category changes range from +4/60 to
+16/60 correct). Disambiguated changes are between −2/60 and +1/60.
The ambiguous gain also appears when the correct unknown answer is at A, B,
or C (respectively +40/205, +36/254, and +55/201 correct), so a single
answer-letter preference does not explain it.
Among ambiguous items, 142 Dream answers changed from incorrect to correct
under the explicit rule and 11 changed from correct to incorrect. Its signed
BBQ bias score fell from 0.2182 to 0.1379. The conditional share of guesses
that selected the stereotyped target actually rose slightly (72.6% to 74.3%);
the lower signed score came from guessing less often. This is why both
abstention and stereotype direction must be shown when interpreting BBQ bias.

**Dream diffusion schedule, BOLD.** On 600 new prompts, both 48-token runs
used first-EOS truncation. Reducing steps from 24 to 12 changed no-lexical-
content rate from 25.33% to 4.50%: paired −20.83 points [−24.00, −17.50].
The share with at least 20 words rose from 0% to 55.50%: +55.50 points
[51.83, 59.33]. First-token EOS fell from 24.67% to 4.33%. This happens
across all five BOLD domains. It demonstrates that the step schedule can
explain a substantial part of Dream's very short baseline outputs.
The completed Llama 3 and Qwen2.5 chat baselines on the same 600 prompt IDs
reach at least 20 words on 98.33% and 98.83%, respectively. Dream's selected
12-step setting therefore still has a large coverage gap to those AR models.

**LLaDA EOS priority, BOLD.** On the same 600 prompts, LLaDA was run at 48
tokens and 48 steps with and without putting predicted EOS last in fill
priority. Both outputs were truncated at first EOS. No-lexical-content rate
fell from 15.50% to 0%: paired −15.50 points [−18.33, −12.67]. The share with
at least 20 words rose from 44.17% to 99.33%: +55.17 points [51.33, 59.17].
Median first EOS position moved from 7 to 47, the last token slot. This
directly supports an EOS-placement mechanism for output coverage in our
LLaDA adapter.
LLaDA's selected setting reaches a comparable *length frequency* to the AR
baselines, though it copies the source prompt more often and has not been
judged for content quality.

## Quality limits and next decision

The BOLD interventions improve **coverage**, not established factuality,
continuation quality, or fairness. Manual inspection found repeated words and
invented facts in some Dream 12-step outputs. LLaDA's EOS intervention raised
the fraction of outputs that start by copying the full prompt from about
31.3% to 68.2%. In the post hoc Dream sweep on the same 600 prompts, at least
20 words appeared on 55.5%, 61.7%, 61.8%, and 0% of outputs at 12, 16, 20,
and 24 steps, respectively. Among those long outputs, the share with at least
20% repeated trigrams was about 19.5%, 20.8%, and 14.3% at 12, 16, and 20
steps. It is undefined at 24 steps because none was long. Thus 20 steps is
the most promising tested setting by these two *limited* diagnostics. Its
coverage advantage over 12 steps is a paired +6.33 points [2.17, 10.67].
On prompts where both outputs had at least 20 words, the 20-step setting's
heavy-repetition rate was 6.94 points lower than 16 steps [−11.67, −2.22]
and 8.61 points lower than 12 steps [−14.23, −3.00] by paired item bootstrap.
There is no evidence here that 20 steps produces more factual or relevant
text. A small blinded human assessment of factuality,
coherence, continuation relevance, and stereotype content would still be
needed before calling either BOLD setting a quality or fairness improvement.

If the team pursues one clean central claim, the Dream evidence-rule result
is currently the strongest: it improves ambiguous BBQ accuracy on a second,
nonoverlapping sample while leaving disambiguated accuracy nearly unchanged.
The BOLD decoding results explain why naive sentiment comparisons are
unreliable: a model can score favorably simply because it emits almost no
lexical content, and a procedure that emits much more text can have quality
costs. The Dream masked-score correction adds a third lesson about measuring
diffusion checkpoints with their actual prediction heads.
