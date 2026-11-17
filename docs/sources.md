# Source audit

Brief: `PROJECTS-Trustworthy_LLMs.pdf`, pages 8–10. Linked references checked 2026-11-14.

| Source | Use / caution |
| --- | --- |
| [MIMIR dataset](https://huggingface.co/datasets/iamgroot42/mimir) | Gated. Card says it is for models trained on The Pile; it is not direct membership ground truth for our four models. |
| [CheckMIABench](https://arxiv.org/html/2606.17464v1) | Matched member/nonmember distributions and blind text-only checks are essential. |
| [Automatic Calibration MIA](https://arxiv.org/pdf/2505.03392) | Candidate score calibration; compare only after validating split construction. |
| [Tag&Tab](https://arxiv.org/pdf/2501.08454) | Keyword-based MIA attack baseline. |
| [Membership Inference Attacks on Discrete Diffusion Language Models](https://arxiv.org/abs/2605.16445) | Withdrawn; no PDF is available. Do not rely on its reported results. |
| [Dissimilarity Learned Policy](https://arxiv.org/pdf/2305.01550) | Approximate memorization mitigation via learned generation policy. |
| [Goldfish loss](https://arxiv.org/pdf/2406.10209) | Token loss omission intervention to transfer to diffusion. |
| [Planting and Mitigating Memorized Content](https://arxiv.org/pdf/2212.08619) | Canary controls and mitigation comparison. |
| [Undesirable Memorization survey](https://arxiv.org/pdf/2410.02650) | Taxonomy of memorization metrics and failure modes. |
| [Fine-tuned LLM memorization risk](https://arxiv.org/html/2508.14062) | Additional background; claims need independent validation. |
| [Diffusion memorization and sampling effects](https://arxiv.org/abs/2603.02333) | Existing 2026 result links sampling resolution to extraction; include sampling steps as an experimental axis rather than presenting that link as new. |
| [Loss Masking Under the Hood](https://aclanthology.org/2026.privatenlp-main.5/) | AR study reports suppressed token generation alongside retained internal information; motivates checking reconstruction/MIA separately from verbatim extraction. |
| [LLaDA official code](https://github.com/ML-GSAI/LLaDA) / [checkpoint](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct) | Official masked diffusion generation method. |
| [Dream official code](https://github.com/DreamLM/Dream) / [checkpoint](https://huggingface.co/Dream-org/Dream-v0-Instruct-7B) | `diffusion_generate` API; official code tests Transformers 4.46.2 / Torch 2.5.1. |
| [Llama checkpoint](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) / [Qwen checkpoint](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) | AR comparators. Meta requires license access. |
| [BBQ copy](https://huggingface.co/datasets/heegyu/bbq) / [official data](https://github.com/nyu-mll/BBQ/tree/main/data) | The copy's viewer API supplied the small pilots; the full extract uses the official 11 JSONL files at pinned commit `bea11bd97d79217245b5871acd247b9d6eb24598` because the viewer rate limited large downloads. Cite original BBQ work in the final report. |
| [BOLD](https://huggingface.co/datasets/AmazonScience/bold) / [official repository](https://github.com/amazon-science/bold) | The official repository lists 23,679 prompts across five domains. Full runs use every prompt, not every source row. |
| [BOLD paper](https://arxiv.org/abs/2101.11718) / [NLTK VADER data](https://github.com/nltk/nltk_data) | VADER compound sentiment is one BOLD evaluation axis. It is a lexical proxy, not a regard or toxicity measure; report empty generations separately. |
| [CrowS-Pairs](https://huggingface.co/datasets/nyu-mll/crows_pairs) | Pairwise benchmark; native HF loader uses arbitrary Python code. |
| [StereoSet](https://github.com/moinnadeem/StereoSet) / [HF card](https://huggingface.co/datasets/McGill-NLP/stereoset) | Original intrasentence dev split, with stereotype, anti-stereotype, and unrelated choices. |
| [BBQ original score script](https://github.com/nyu-mll/BBQ/blob/main/analysis_scripts/BBQ_calculate_bias_score.R) | Uses `target_loc` metadata, computes signed preference among non-unknown answers, and scales ambiguous bias by error rate. |

The LLaDA adapter in this repo follows the official low-confidence remasking approach without copying code. Cite the official repository and model paper if using it in a publication or report.
