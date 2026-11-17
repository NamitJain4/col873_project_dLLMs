# Submitted runs (2026-11-14)

The first eight-hour jobs were cancelled during checkpoint download. The Hugging Face cache retained some partial downloads, and the replacement 24-hour jobs below were submitted. Status changes over time; query PBS and inspect `logs/` and `results/` before treating a run as complete.

| Model | BBQ smoke job | Dependent follow-up job | Checkpoint |
| --- | --- | --- | --- |
| Qwen2.5-7B-Instruct | `179899.pragya` | `179916.pragya` | `a09a35458c702b33eeacc393d103063234e8bc28` |
| Meta-Llama-3-8B-Instruct (brief's exact model) | `180080.pragya` | `180081.pragya` | `8afb486c1db24fe5011ec46dfbe5b5dccdb575c2` |
| Llama-3.1-8B-Instruct (provisional substitute) | `179900.pragya` | `179917.pragya` | `0e9e39f249a16976918f6564b8830bc894c89659` |
| LLaDA-8B-Instruct | `179901.pragya` | `179918.pragya` | `08b83a6feb34df1a6011b80c3c00c7563e963b07` |
| Dream-v0-Instruct-7B | `179902.pragya` | `179919.pragya` | `05334cb9faaf763692dcf9d8737c642be2b2a6ae` |

Each follow-up job depends on its corresponding BBQ job succeeding and then runs balanced BBQ, BOLD, CrowS-Pairs, and StereoSet pilots. The initial exact `Meta-Llama-3-8B-Instruct` access check returned HTTP 403. After the user requested access, both the supplied token and the existing Hub login could read its config, so its jobs were added. Treat Llama 3.1 as an extra sensitivity comparator, not as the brief's matched pair.

Input SHA-256 values:

| Extract | SHA-256 |
| --- | --- |
| `bbq_pilot.jsonl` | `bf016bf808e6b8b4a83c5613bfb84a334a9e875f20e0ec3e86c2f116ea5a7168` |
| `bbq_balanced.jsonl` | `9d5dbb472915275b30a5ece8b7c29c44221ca803f872b18485f2ad7f96dc1a74` |
| `bold_pilot.jsonl` | `373547d796289463b5531b176c61a2c2fc75a4ad21dac140d763c68a72d90b30` |
| `crows_pilot.jsonl` | `dec5fe2b1ed91d5bc9245729df2fbb6cf33ab2b24708c508c3b8e1a1bb0f0e9a` |
| `stereoset_pilot.jsonl` | `db7e16cd9c5b3cdab134f8d5d6f634d17eb85cb5bd2c08298a2589c8a94f6f2e` |
| `bbq_additional_metadata.csv` | `f36708416b0e7adb81b47ad1926f9c39c2beb611702b73b01e06c9b6c9ffbd3d` |

No model scores are reported here until output JSONL files pass count and validity checks.

## Recovery on 2026-11-17

The four cached models completed their 22-row BBQ smoke tests: exact Llama 3, Llama 3.1, LLaDA, and Dream. Every prediction parsed as A/B/C. The extract contains only negative-polarity questions and is too small to estimate bias. Accuracy in the ambiguous/disambiguated halves was Llama 3 8/11 and 10/11, Llama 3.1 8/11 and 9/11, LLaDA 10/11 and 10/11, Dream 9/11 and 10/11. These values validate the pipeline only.

The original four follow-up jobs failed their Hugging Face connectivity preflight during a compute-node proxy outage. Since their checkpoints are fully cached, the PBS script now accepts `OFFLINE=1` and skips that preflight. Replacement follow-up jobs are exact Llama 3 `187209.pragya`, Llama 3.1 `187210.pragya`, LLaDA `187211.pragya`, and Dream `187212.pragya`. Each runs balanced BBQ, BOLD, CrowS-Pairs, and StereoSet in sequence.

Qwen's first download failed after caching two of four weight shards. CPU-only prefetch job `187214.pragya` downloads the two remaining shards with Xet disabled and retries. Its dependent offline BBQ job is `187215.pragya`; its dependent follow-up job is `187216.pragya`. Check `qstat -u "$USER"` and `python3 scripts/status.py` before interpreting these as complete.

## Full baseline submitted on 2026-11-17

All five models completed BBQ smoke, balanced BBQ, raw BOLD, CrowS-Pairs, and StereoSet pilots. The full input extracts are:

| Dataset | Rows | SHA-256 |
| --- | ---: | --- |
| BBQ, official 11 categories at commit `bea11bd97d79217245b5871acd247b9d6eb24598` | 58,492 | `57e0efe6a9c3745f27b952b0238e21befc27643cc2992d5984014aa9ad294717` |
| BOLD, all prompts including multiple prompts per source row | 23,679 | `993c95f31b91a7b3784d6edb036b5f31159de5d7fc1108f2dd2d0c63cacf5efa` |
| CrowS-Pairs | 1,508 | `31b31be11bc7bfd94757e562db27a58bf5bfea70cde4f5cfd16532e5468729a4` |
| StereoSet intrasentence dev | 2,106 | `2520aae000db19809f3c66915fa4ba33a5184f738b83ceaa565d7c5496a8628a` |

Each `DATASET=all_full,OFFLINE=1` job runs the four inputs in that order, resuming completed rows: Qwen `188416.pragya`, exact Llama 3 `188417.pragya`, LLaDA `188418.pragya`, Dream `188419.pragya`, and extra Llama 3.1 `188420.pragya`. CPU validation and metric job `188434.pragya` depends on all five succeeding. It writes to `results/summary/`. The BBQ input is exactly balanced between ambiguous/disambiguated and negative/nonnegative questions. Sixteen Gender Identity rows lack an author-supplied stereotype target; count them in accuracy and exclude them from target-based bias scores.

BOLD raw-prefix generation produced 24 empty continuations out of 40 for LLaDA. An instructed chat continuation pilot `188443.pragya` produced 0 empty outputs out of 40. The instructed full BOLD runs, using the same prompt format for every model, are Qwen `188457.pragya`, Llama 3 `188458.pragya`, LLaDA `188459.pragya`, Dream `188460.pragya`, and Llama 3.1 `188461.pragya`. CPU validation and VADER sentiment job `188462.pragya` depends on all five. Keep raw and instructed conditions separate. The VADER lexicon is pinned to NLTK data commit `550b6625bcef1f2abff2ff770a5a0d272c9c6b2a` with ZIP SHA-256 `8adba4294eef3964d820bf655e37e61bdc3a341994356af59b74fb3b4a36ce5c`.

Controlled synthetic canary LoRA training checks are Qwen `188454.pragya` and LLaDA `188455.pragya`, 16 examples each. Their dependent extraction and likelihood evaluations are `188469.pragya` and `188470.pragya`. These are engineering checks, not the full memorization experiment.

The queue rejected an additional LLaDA full-training submission on 2026-11-17 with its per-user queued-job limit. The single-model Qwen full job `188473.pragya` was cancelled and replaced by consolidated job `188475.pragya`, which depends on both pilot extraction checks succeeding. It sequentially trains full-corpus LoRA adapters for all four assigned models, evaluates all 192 candidates against both base and tuned checkpoints, validates ids and scores, and writes `results/summary/mia_full.tsv`. This is a controlled synthetic fine-tuning baseline, not pretraining membership inference and not a mitigation result. If a stage fails, durable model outputs remain for diagnosis and the pipeline can resume completed stages.

Goldfish-style stable 25% response-token loss omission is queued as `188478.pragya`, dependent on the full baseline. It uses the same 1,184 training rows, seed, optimizer steps, LoRA configuration, and instruction; the omitted target positions are deterministic per record id and therefore fixed across copies. It evaluates the same 192 candidates and writes `results/summary/mia_goldfish.tsv`. A separate 128-record fictional routine-email holdout, SHA-256 `5932ac88e16ef6fb1afcc3323579c3433ebfacdb6aef608463c4b4285b50a45a`, is scored by job `188479.pragya` after Goldfish for a within-model utility comparison. All downstream jobs are held until their dependencies succeed. These runs are one-seed experiments; do not claim a robust mitigation effect before inspecting validated outputs and running uncertainty checks.

CPU job `188488.pragya` depends on Goldfish and computes paired candidate bootstrap intervals for the change in extraction rate and membership AUROC, overall and by canary repetition group. It writes `results/summary/mia_mitigation_paired.tsv`.

## Full-run recovery and comparison notebook, 2026-11-17

All five full BBQ runs (58,492 rows each) and all five instructed BOLD runs (23,679 rows each) have completed. The original `all_full` jobs for Qwen (`188416.pragya`), LLaDA (`188418.pragya`), and Dream (`188419.pragya`) stopped at raw BOLD because the source contains five empty prompts. `scripts/eval_bold.py` now records an explicit empty continuation for these rows. Replacement, resumable jobs `189426.pragya`, `189427.pragya`, and `189428.pragya` continue raw BOLD, CrowS-Pairs, and StereoSet for Qwen, LLaDA, and Dream respectively. The exact Llama 3 and extra Llama 3.1 completed all four full benchmarks in the original jobs.

The five empty BOLD source prompts are excluded from sentiment denominators, leaving 23,674 valid prompts per model and prompt mode. The instructed-condition sentiment rescore `189430.pragya` completed. CPU job `189429.pragya` depends on the three resumed GPU jobs; it will validate all 20 full output files and write BBQ, raw BOLD sentiment, and pair summaries. Notebook job `189432.pragya` depends on `189429.pragya`; it will execute `notebooks/final_comparison.ipynb` and save aggregate plots under `results/figures/` and a comparison table at `results/summary/final_comparison.csv`.

The full synthetic fine-tuning, Goldfish-style omission, MIA, paired bootstrap, and utility jobs are complete and validated. `notebooks/completed_runs_so_far.ipynb` is an executed progress artifact. Regenerate it with `scripts/build_progress_notebook.py`. The final notebook refuses to plot a full comparison until the source validators pass. After the dependent jobs finish, check `qstat -u "$USER"`, `logs/189429.pragya.OU`, `logs/189432.pragya.OU`, and notebook execution counts.

## Status on 2026-11-18

LLaDA `189427.pragya` and Dream `189428.pragya` finished with exit status 0. Their full raw BOLD, CrowS-Pairs, and StereoSet files pass `scripts/validate_full.py --models llama llama31 llada dream` together with the completed Llama runs. Only Qwen `189426.pragya` remains active. Four-model raw BOLD sentiment and pair summaries are in `results/summary/bold_raw_completed_four_sentiment.tsv` and `results/summary/pair_completed_four.tsv`; `notebooks/completed_runs_so_far.ipynb` was re-executed to include those results. The dependent all-model validation and notebook jobs remain queued.

Qwen pair jobs `189724.pragya` (CrowS-Pairs) and `189725.pragya` (StereoSet) were submitted independently so they can run while Qwen raw BOLD finishes. Both pair evaluators now acquire a per-output file lock before checking already completed rows; if the original all-full job reaches a pair output at the same time, it waits and then resumes without duplicate writes. The original all-full Qwen job remains a fallback for both pair benchmarks.

## Complete baseline on 2026-11-18

Qwen `189426.pragya`, pair jobs `189724.pragya` and `189725.pragya`, full validation/summarization `189429.pragya`, and notebook render `189432.pragya` all finished with exit status 0. `scripts/status.py` reports complete output for all five models on BBQ (58,492 each), raw BOLD (23,679 each), CrowS-Pairs (1,508 each), and StereoSet (2,106 each). The instructed BOLD condition is also complete for all five models. `results/summary/validation.tsv` records all 20 full-output validations. `notebooks/final_comparison.ipynb` has 12 executed code cells and no errors; it saves `results/summary/final_comparison.csv` and the figures under `results/figures/`. The four assigned models also have complete synthetic MIA, Goldfish-style mitigation, and held-out utility results. Llama 3.1 is an additional comparator only.
