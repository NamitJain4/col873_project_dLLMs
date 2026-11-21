# AR versus diffusion LLMs: bias and memorization

**Scoring correction (2026-11-21):** Dream's masked-token head predicts each
token from the previous logit position. The original Dream CrowS-Pairs,
StereoSet, MIA, Goldfish, and utility numbers in the older final comparison
are invalid. Their aligned-v2 reruns are complete and used in the current
notebook and summary. The original BOLD generation used a fixed-span decoder
that could show text after EOS; the new 600-prompt paired replications use
first-EOS truncation for both conditions.
Read [the correction](docs/scoring_erratum.md) and the new
[divergence analysis](notebooks/divergence_analysis.ipynb) before using figures.
The current [exploration protocol](docs/exploration_protocol.md) tests
abstention, generation coverage/EOS, and correct masked-token scoring.
The [research memo](docs/findings.md) summarizes the completed results and
their limits for teammates.

Course project for COL873. The assigned brief is pages 8–10 of [PROJECTS-Trustworthy_LLMs.pdf](PROJECTS-Trustworthy_LLMs.pdf). The full baseline compares four assigned instruction models on BBQ, BOLD, CrowS-Pairs, and StereoSet. See [docs/project_plan.md](docs/project_plan.md) for the research plan and [docs/sources.md](docs/sources.md) for the source audit.

## Layout

- `src/dllm_eval/`: model adapters and scoring logic
- `scripts/`: dataset preparation, evaluation, and PBS jobs
- `data/processed/`: benchmark extracts (ignored by Git)
- `results/`: raw JSONL predictions stay local; compact summaries and final PNG figures are committed
- `logs/`: PBS logs (ignored by Git)
- `docs/`: experiment protocol and literature notes
- `notebooks/divergence_analysis.ipynb`: current executed analysis and plots
- `notebooks/final_comparison.ipynb`: historical analysis with invalid Dream score plots

## First run

The scripts use Python 3.11, PyTorch with CUDA, Transformers, Accelerate, NumPy, and scikit-learn. On this cluster, `../col873_ass2/.venv-gpu/bin/python` has these packages. Set `PROJECT_PYTHON` to another compatible interpreter if preferred. Hugging Face credentials are read from the existing user login; no token is stored in this repo.

```bash
python3 scripts/prepare_bbq.py --per-stratum 1 --output data/processed/bbq_pilot.jsonl
mkdir -p logs results
for model in qwen llama llada dream; do
  qsub -v MODEL="$model",DATASET=bbq_pilot scripts/pbs/eval.pbs
done
```

The pilot uses one ambiguous and one disambiguated example from each of BBQ's 11 categories (22 items). It is a **smoke test**, not a publishable estimate. The job resumes completed predictions. A full extract can be created with `python3 scripts/prepare_bbq.py --full --output data/processed/bbq_full.jsonl`; submit with `DATASET=bbq_full` after reviewing pilot outputs.

The first four jobs were launched before access to the brief's exact `Meta-Llama-3-8B-Instruct` checkpoint was granted. They include `llama31` as a clearly labeled provisional comparator. Access was granted later, and `MODEL=llama` is now the required matched-scale AR run; keep the two Llama versions separate in every table.

Each job writes `results/bbq_<dataset>_<model>.jsonl`. Summarize completed runs with:

```bash
python3 scripts/summarize_bbq.py results/bbq_bbq_pilot_*.jsonl
```

After the balanced BBQ results exist, run `python3 scripts/score_bbq_bias.py --data data/processed/bbq_balanced.jsonl results/bbq_bbq_balanced_*.jsonl` for accuracy, invalid/unknown rates, and the authors' signed bias score. The metadata CSV is stored under `data/processed/` and is kept out of Git.

Do not present this pilot as a completed four model result until all four output files exist and contain 22 valid predictions.

Use `python3 scripts/status.py` to check completed row counts across the full pilot matrix, and `qstat -u "$USER"` for PBS state. A held follow-up job is expected while its `afterok` dependency is still running.

For a second dataset, `python3 scripts/prepare_bold.py --per-domain 8 --output data/processed/bold_pilot.jsonl` builds a BOLD continuation pilot. Submit it with `DATASET=bold_pilot` using the same PBS script. The BOLD outputs are raw continuations; sentiment/regard scoring requires a separately specified classifier and has not been included in the pilot.

For a third dataset, `python3 scripts/prepare_crows.py --per-type 2 --output data/processed/crows_pilot.jsonl` builds a CrowS-Pairs pilot. Submit it with `DATASET=crows_pilot`. This pilot uses mean token likelihood for AR and repeated masked-token likelihood for diffusion, so it measures within-model pair preference and is **not** the canonical CrowS-Pairs score. Do not compare raw score magnitudes across model types.

For StereoSet, `python3 scripts/prepare_stereoset.py --per-type 4 --output data/processed/stereoset_pilot.jsonl` builds an intrasentence dev pilot. Submit it with `DATASET=stereoset_pilot`. The pair preference and meaningful-versus-unrelated checks are preliminary; they do not constitute the official StereoSet composite score.

The smoke BBQ extract contains only negative-polarity questions. `python3 scripts/prepare_bbq_balanced.py --output data/processed/bbq_balanced.jsonl` builds a 44 item pilot with both question polarities and both context conditions in every category.

To schedule all four follow-up pilots after a model's BBQ job succeeds, submit a single `DATASET=followups` job with `qsub -W depend=afterok:BBQ_JOBID -v MODEL=qwen,DATASET=followups scripts/pbs/eval.pbs`. This avoids the queue's per-user pending-job limit. It runs balanced BBQ, BOLD, CrowS-Pairs, and StereoSet in that order, with resumable outputs for each.

For the memorization experiment, `python3 scripts/prepare_synthetic_mia.py` creates a controlled fictional corpus and matched member/nonmember candidates. See [docs/mia_protocol.md](docs/mia_protocol.md). The queued short LoRA checks are engineering validation, not privacy results.

## Full runs, November 2026

The pilot matrix and all five full BBQ, raw BOLD, instructed BOLD, CrowS-Pairs, and StereoSet runs are complete and validated. The final comparison notebook has executed successfully. BOLD has a separate instructed continuation condition because raw-prefix LLaDA often returns an empty continuation. Read [docs/runs.md](docs/runs.md) for job IDs and input hashes. `python3 scripts/status.py` reports row counts.

The controlled synthetic canary fine-tuning, base and tuned MIA, Goldfish-style token-loss omission, and separate 128-record held-out utility comparison have completed for all four assigned models. These are one-seed fine-tuning membership experiments with fictional data; see [docs/mia_protocol.md](docs/mia_protocol.md) for interpretation limits.

## Shared results

The [executed divergence notebook](notebooks/divergence_analysis.ipynb) contains the current analysis and 30 figures, including full BBQ comparisons, BOLD coverage and sentiment diagnostics, corrected Dream pair/MIA results, and nonoverlapping BBQ and BOLD replications. The [consolidated comparison](results/summary/final_comparison.csv) now includes the corrected Dream values. The older [full comparison notebook](notebooks/final_comparison.ipynb) is retained as a historical artifact; its Dream pair, MIA, Goldfish, and utility plots are invalid. See [the erratum](docs/scoring_erratum.md) before using old figures or detailed TSVs.

Raw benchmark extracts, per-example outputs, adapters, and PBS logs remain local because they are large or contain generated text. The pinned data sources and input hashes in `docs/` support reruns. Teammates need their own Hugging Face access to the gated Meta Llama checkpoint to reproduce it.

Regenerate and execute the current notebook with `../col873_ass2/.venv-gpu/bin/python scripts/render_divergence_notebook.py`. The analysis environment is pinned in `requirements-analysis.txt`. The legacy final-comparison renderer remains disabled to prevent reuse of its invalid Dream plots.
