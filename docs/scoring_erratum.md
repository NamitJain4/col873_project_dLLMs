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
use `aligned_v2` in both their file names and `score_method` provenance. The
old outputs are retained locally for audit, but excluded from the new
`notebooks/divergence_analysis.ipynb`. Corrected jobs and progress are tracked
in `docs/exploration_protocol.md`.

Until the corrected runs complete, **do not use the Dream CrowS-Pairs,
StereoSet, MIA, Goldfish, or utility values** in the older
`notebooks/final_comparison.ipynb`, `results/summary/final_comparison.csv`, or
their associated plots. When the reruns finish, regenerate the comparison
notebook and summaries using only aligned-v2 Dream outputs.
