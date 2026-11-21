# Dream masked-token alignment correction (2026-11-21)

The pinned Dream checkpoint predicts target token at position `i` from the
logits at position `i-1`. This is explicit in the [Dream paper](https://arxiv.org/html/2508.15487v1)
(Section 4.1, Shift Operation) and the cached checkpoint's `generation_utils.py`:
its `diffusion_generate` method shifts logits right before looking up masked
tokens. LLaDA predicts a masked token from the logits at the same position.

Our first masked-token scorers and Dream LoRA fine-tuning used logits at the
masked position for **both** diffusion checkpoints. This was correct for LLaDA
and wrong for Dream. The original Dream CrowS-Pairs, StereoSet, base/tuned MIA,
Goldfish, and utility scores are therefore invalid as measurements of the
intended Dream objective. The old Dream full and Goldfish adapters were trained
with the wrong objective and must not be compared to other adapters.

The Dream BBQ and BOLD runs use the checkpoint's official `diffusion_generate`
implementation and remain valid. All AR and LLaDA runs remain valid.

The correction is implemented in `src/dllm_eval/scoring.py`. Corrected outputs
use `aligned_v2` in both their file names and `score_method` provenance. All
corrected Dream pair, base/full/Goldfish MIA, and utility files passed the
complete-row, source-hash, and score-method validation in
`scripts/validate_corrected_dream.py`. The corrected headline values are in
`results/summary/dream_corrected_headline.tsv` and the current
`results/summary/final_comparison.csv`. The executed
`notebooks/divergence_analysis.ipynb` uses the aligned-v2 Dream outputs.

The **older** `notebooks/final_comparison.ipynb` and its Dream pair, MIA,
Goldfish, and utility plots remain invalid historical artifacts. Do not cite
them. The old raw outputs are retained locally for audit and excluded from the
current notebook. The alignment change is a measurement correction, not a
model-performance improvement.
