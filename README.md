# AR versus diffusion LLMs: bias, memorization, and decoding

COL873 Trustworthy LLMs project. The assignment is on pages 8–10 of
[PROJECTS-Trustworthy_LLMs.pdf](PROJECTS-Trustworthy_LLMs.pdf). We compare
LLaDA-8B-Instruct with Meta-Llama-3-8B-Instruct, and
Dream-v0-Instruct-7B with Qwen2.5-7B-Instruct. Llama-3.1-8B-Instruct is an
additional historical comparator; the exact assigned Llama 3 checkpoint was
subsequently run.

## Start here

Open **[project_walkthrough.ipynb](project_walkthrough.ipynb)** in the repository
root. It is already executed and contains the project question, model and metric
explanations, a run ledger, complete benchmark results, the corrected Dream
scores, diagnostic interventions, 16 plots, and interpretations. Reading the
notebook requires no GPU, Hugging Face access, or local raw outputs. Its code
cells read the small committed tables in `results/summary/` if you want to
rerun the plots.

The headline finding is that there is no uniform diffusion penalty: LLaDA
beats Llama 3 on BBQ accuracy, while Dream struggles especially when BBQ
questions lack sufficient evidence. An explicit evidence rule improves Dream's
ambiguous accuracy on a separate 1,320-item replication. BOLD generation
coverage is sensitive to denoising steps and EOS handling; greater length alone
does not establish better content. The notebook gives the numbers and caveats.

**Measurement correction:** Dream predicts masked token $i$ from logit position
$i-1$. The original Dream CrowS-Pairs, StereoSet, synthetic MIA, Goldfish, and
utility measurements used the wrong alignment. Use only the corrected
`aligned_v2` Dream rows in the root notebook and current summaries. See
[docs/scoring_erratum.md](docs/scoring_erratum.md). Original full BOLD decoding
also used a fixed span that could expose text after EOS; the later 600-prompt
paired replication truncates at the first EOS in both conditions.

## Directory guide

| Path | What is there | Where to start |
| --- | --- | --- |
| [`project_walkthrough.ipynb`](project_walkthrough.ipynb) | Main, executed, explanation-first notebook | **Start here for results and plots** |
| [`PROJECTS-Trustworthy_LLMs.pdf`](PROJECTS-Trustworthy_LLMs.pdf) | Course brief; assigned project on pages 8–10 | Check the original requirements |
| [`docs/`](docs/) | Protocols, research memo, run history, source audit, correction notice | Use the document map below |
| [`src/dllm_eval/`](src/dllm_eval/) | Model adapters, BBQ prompts/parsing, masked-token scoring | Read when checking implementation |
| [`scripts/`](scripts/) | Dataset preparation, evaluators, validation, summarization, notebook builders | Run or audit an experiment |
| [`scripts/pbs/`](scripts/pbs/) | PBS submission files for GPU and CPU jobs | Check job resources and commands |
| `data/processed/` | Prepared benchmark and synthetic extracts | Local only; ignored by Git |
| `results/*.jsonl`, `results/exploration/`, `results/mia/`, `results/utility/` | Per-example predictions and diagnostic outputs | Local audit; ignored by Git |
| [`results/summary/`](results/summary/) | Compact TSV/CSV metric tables, including corrected Dream and notebook inputs | Shared numerical results |
| [`results/figures/`](results/figures/) | Exported figures from earlier analyses | Supplementary plots |
| `checkpoints/` | Local LoRA adapters and cached training artifacts | Ignored by Git; large files |
| `logs/` | PBS stdout/stderr and job logs | Ignored by Git; use for failures |
| [`notebooks/`](notebooks/) | Earlier technical and historical notebooks | Supplementary or legacy only |
| [`tests/`](tests/) | Evaluator and scoring regression tests | Check implementation changes |
| [`requirements-analysis.txt`](requirements-analysis.txt), [`requirements-eval.txt`](requirements-eval.txt) | Analysis and inference dependencies | Set up an environment |

### Which document answers which question?

| Question | File |
| --- | --- |
| What happened and what does it mean? | [Main notebook](project_walkthrough.ipynb), then [research memo](docs/findings.md) |
| Which jobs, checkpoints, and input hashes were used? | [Run history](docs/runs.md), [output inventory](results/summary/walkthrough_output_inventory.tsv) |
| How were the BBQ/BOLD diagnostics designed? | [Exploration protocol](docs/exploration_protocol.md) |
| Why were Dream scores rerun? | [Scoring erratum](docs/scoring_erratum.md) |
| How was synthetic membership tested? | [MIA protocol](docs/mia_protocol.md) |
| Where did datasets and papers come from? | [Source audit](docs/sources.md) |
| What project direction was considered? | [Project plan](docs/project_plan.md) |

The technical [divergence notebook](notebooks/divergence_analysis.ipynb) has
additional plots and item-level diagnostics. The older
[final comparison notebook](notebooks/final_comparison.ipynb) is **historical**:
its Dream pair, MIA, Goldfish, and utility plots are invalid. Do not cite those
plots. The root notebook supersedes it for the current comparison.

## What has run?

All five checkpoints completed full BBQ (58,492 items), raw BOLD and instructed
BOLD (23,679 source prompts each), CrowS-Pairs (1,508 pairs), and StereoSet
intrasentence (2,106 examples). The four assigned models also completed the
controlled synthetic canary baseline, Goldfish-style loss omission, membership
attack, and 128-record held-out utility check. Corrected Dream score-based
reruns and the BBQ/BOLD diagnostic replications are complete. The notebook's
inventory checks expected IDs and source hashes for 83 output files. Five BOLD
source prompts are blank; aggregate analysis uses 23,674 nonblank prompts.

The pairwise CrowS-Pairs/StereoSet summaries are **within-model preferences**,
not the benchmarks' official aggregate scores. Synthetic membership results
come from one-seed LoRA fine-tuning on fictional records, not inferred
pretraining membership. These limits are explained alongside the plots.

## Reading and reproducing the analysis

From the repository root, open `project_walkthrough.ipynb` in Jupyter. The
saved outputs already include the figures. To regenerate the notebook from
the **committed aggregate tables**, run:

```bash
../col873_ass2/.venv-gpu/bin/python scripts/render_project_walkthrough.py
```

That path is the working Python environment on this cluster. Elsewhere,
install [`requirements-analysis.txt`](requirements-analysis.txt), register an
`ipykernel` named `dllm-eval`, and use that Python interpreter with the same
script. The renderer rebuilds and executes the notebook at the **repository
root**; it does not run models or submit PBS jobs. Running the notebook cells
directly also works from the root directory.

If the large raw outputs are available locally, first re-audit them and rebuild
the compact notebook inputs:

```bash
../col873_ass2/.venv-gpu/bin/python scripts/build_walkthrough_tables.py
../col873_ass2/.venv-gpu/bin/python scripts/render_project_walkthrough.py
```

`build_walkthrough_tables.py` requires ignored `data/processed/` and raw
`results/` JSONL files, so a fresh Git clone can **read and rerun the plots**
but cannot reconstruct raw outputs without those data and model runs. The
checkpoint revisions, source hashes, PBS job IDs, and original data links are
recorded under `docs/`.

## Running or checking cluster jobs

Evaluation uses Python 3.11, PyTorch/CUDA, Transformers, Accelerate, NumPy,
and scikit-learn. Hugging Face credentials come from the user's existing Hub
login; no token is stored in this repository. Access to the gated Meta Llama
checkpoint is required to reproduce that model.

From the repository root, check output counts and PBS state with:

```bash
python3 scripts/status.py
qstat -u "$USER"
```

The full baseline PBS entry point is [`scripts/pbs/eval.pbs`](scripts/pbs/eval.pbs).
For example, `MODEL=llada,DATASET=all_full,OFFLINE=1` runs the four full
baseline extracts sequentially and resumes already completed JSONL rows:

```bash
qsub -v MODEL=llada,DATASET=all_full,OFFLINE=1 scripts/pbs/eval.pbs
```

The completed jobs do **not** need to be resubmitted to view the results.
Read [docs/runs.md](docs/runs.md) for the actual submitted job IDs and
[docs/exploration_protocol.md](docs/exploration_protocol.md) for diagnostic
jobs. The scripts in `scripts/validate_*.py` check output completeness and
provenance before summaries are interpreted. For a small pipeline smoke test,
see the pilot commands and history in those documents rather than treating a
22-item pilot as a benchmark estimate.

The course brief prohibits using generative AI to write the final submitted
report. This repository's notebook and notes are internal analysis materials
for teammates to examine and write from independently.
