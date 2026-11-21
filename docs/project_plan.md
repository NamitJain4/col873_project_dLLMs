# Project direction and next experiments

**Current working direction (2026-11-21):** explain why LLaDA and Dream
diverge by separating evidence-based abstention, generation coverage/EOS, and
masked-token score alignment. The first complete-baseline analysis, corrected
Dream reruns, exploratory subset interventions, and a nonoverlapping BBQ
replication are in
[`docs/exploration_protocol.md`](exploration_protocol.md). The earlier
Goldfish-transfer hypothesis below remains a possible follow-up after the
Dream scoring correction is complete.

## Assigned question

Compare social bias and memorization in LLaDA-8B-Instruct versus Meta-Llama-3-8B-Instruct, and Dream-v0-Instruct-7B versus Qwen2.5-7B-Instruct. The matched pairs are approximate controls, not identical architectures, tokenizers, corpora, or instruction tuning. In particular, Dream's Qwen lineage does not make the pair identical.

## E1: bias baseline

1. Run a 22 item BBQ smoke test (one ambiguous and one disambiguated example in each of 11 categories) to verify model loading, generation, parsing, and logging.
2. Run the balanced BBQ pilot using the same prompt, deterministic decoding, fixed generation budget, and no few-shot examples. Check throughput, parsing, and category coverage before scaling to a stratified sample or the full BBQ dataset within the available GPU budget. Report accuracy, invalid response rate, and ambiguous/disambiguated splits by category. The canonical BBQ bias score requires mapping each answer to a stereotyped group, question polarity, and unknown answer; validate that mapping before reporting it.
3. Add BOLD generations from raw prefixes, with the same sentiment/regard classifier for all models. The initial BOLD pilot stores continuations only. Report demographic group disparity with bootstrap confidence intervals, checking prompt wording and group sizes. Add StereoSet and CrowS-Pairs using comparable *conditional* scoring definitions. AR log likelihood and diffusion masked reconstruction/ELBO are distinct quantities; raw scores must not be compared across architectures. Within-model pair preferences can be compared, with appropriate method notes.

## E2: memorization and membership inference

First build a controlled, consent-safe fine-tuning set with synthetic canaries and a nonmember holdout drawn from the same generator. Fix training budget, sequence length, injection counts, LoRA rank, and random seeds across pairs. Measure exact extraction, canary exposure, and AUROC for model-specific loss scores with confidence intervals. Test a text-only classifier on members versus nonmembers: if it succeeds, the split is confounded. MIMIR's Pile membership labels are not ground truth for the four assigned checkpoints; use MIMIR with a known Pile-trained control such as Pythia to validate the MIA code separately if access is granted.

Enron has real personal information. Prefer synthetic email-shaped records for the first study. If later using Enron, keep raw emails and extraction outputs private and report aggregate results or redacted examples.

## E3: transfer of mitigation

Vary canary duplication and apply exact deduplication. Train baseline and Goldfish-style token-loss omission variants with matched optimizer steps and token budgets. For diffusion, define the dropped-token mask *in addition to* the stochastic denoising mask; otherwise the denoising mask alone changes the meaning of the intervention. Compare privacy gain against held-out utility loss and show per-seed variation. DP-SGD is a stretch goal requiring a correct accountant and privacy unit definition.

## Alternative follow-up: mask-aware Goldfish transfer

The strongest project direction is a **mask-aware Goldfish transfer study**: does withholding loss on a stable random subset of target positions suppress canary extraction in denoising models as it does in AR models? Test whether the effect changes with canary repetition, denoising mask rate, and diffusion sampling steps. Measure both generated canary recovery and chosen-mask reconstruction/MIA. If generation leakage drops while reconstruction-based membership remains high, the mitigation is hiding an output pathway rather than removing the information. This is a hypothesis, informed by [Goldfish loss](https://arxiv.org/pdf/2406.10209), [diffusion sampling effects](https://arxiv.org/abs/2603.02333), and [loss masking mechanism analysis](https://aclanthology.org/2026.privatenlp-main.5/); none of those sources establishes the proposed AR versus dLLM transfer result.

## Completed full baseline and prompt test

The complete 58,492-item BBQ baseline has been run for all four assigned
models. Dream performs much worse than Qwen on ambiguous questions (56.43%
versus 94.78%) but better on disambiguated questions (97.49% versus 89.10%).
LLaDA is more accurate than Llama 3 on both splits. The working mechanism
question is therefore how much of the Dream gap reflects handling missing
evidence, and how generation settings change output coverage. The 132-item
prompt test found a Dream ambiguous gain from 39/66 to 48/66; an independent
1,320-item replication found a +19.85-point ambiguous gain for the explicit
evidence rule at a fixed 32-step budget, while disambiguated accuracy changed
by −0.61 points. Separate 600-prompt BOLD replications found that Dream's
12-step decoding and LLaDA's EOS-confidence depriority both substantially
increase output coverage. Neither establishes better factuality or lower
bias; the current quality sweep tests repetition and prompt copying.
See the [executed divergence notebook](../notebooks/divergence_analysis.ipynb)
for the plots, item-level analysis, and validity conditions.

The course brief prohibits using generative AI to write the final report. These notes are for experiment planning; group members must write the report themselves.
