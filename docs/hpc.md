# HPC runbook

This run uses the `mausam.inlp.pragya.spons` project, the `standard` queue, one GPU, four CPUs, 40 GB host memory, and a 24-hour walltime. The resource and proxy setup follows the working `col873_ass1/assignment1.pbs` and the earlier cluster conversations: model inference runs only on GPU nodes, and the compute node uses the IITD proxy login helper before Hugging Face access. Dataset downloads and syntax checks can run on the login node. The PBS script starts the proxy helper only when the first connectivity check fails, and cleans it up when the job exits.

Submit from this repository root so `PBS_O_WORKDIR` points to the right directory:

```bash
mkdir -p logs results
qsub -v MODEL=qwen,DATASET=bbq_pilot scripts/pbs/eval.pbs
qstat -u "$USER"
qstat -f JOBID | grep -E 'job_state|comment|exec_host|project|Resource_List'
tail -f logs/JOBID.OU
```

Supported `MODEL`: `qwen`, `llama`, `llama31`, `llada`, and `dream`. `llama` is the brief's exact checkpoint; `llama31` is a provisional extra comparator from before access was granted. Supported `DATASET`: `bbq_pilot`, `bbq_balanced`, `bbq_full`, `bold_pilot`, `bold_full`, `crows_pilot`, `crows_full`, `stereoset_pilot`, `stereoset_full`, `followups`, or `all_full`. `all_full` runs full BBQ, BOLD, CrowS-Pairs, and StereoSet in sequence. Outputs resume at the example level, but a changed input file hash is rejected to prevent accidental mixing.

To run another benchmark only after its BBQ job succeeds:

```bash
qsub -W depend=afterok:BBQ_JOBID -v MODEL=qwen,DATASET=followups scripts/pbs/eval.pbs
```

The queue rejected the final two separate StereoSet follow-up submissions on 2026-11-14 because of its per-user queued-job limit. Those first submissions were cancelled while extending the pilot walltime. The replacement uses one combined follow-up job per model, so all five model runs have balanced BBQ, BOLD, CrowS-Pairs, and StereoSet scheduled. Check `qstat` first; do not duplicate an already queued or running benchmark job.

The four initially queued checkpoints total roughly 60 GB; the exact Meta Llama 3 checkpoint adds about 16 GB. First startup downloads weights to the shared Hugging Face cache, which can take hours through the campus proxy. A direct range test measured about 0.4 MB/s on 2026-11-14; the first eight-hour pilot jobs were replaced with 24-hour jobs for this reason. GPU allocation starts the PBS walltime clock, so inspect cache growth and logs. Do not interpret missing result JSONL during download as a model failure.

No Hugging Face token is embedded in the script. The connected account's existing Hugging Face login is read by the Hub library. Access to `meta-llama/Meta-Llama-3-8B-Instruct` was granted after the initial provisional jobs were submitted; both the supplied token and the existing login can now read its config.

After a checkpoint is fully cached, submit with `OFFLINE=1` in the `-v` list. This sets the Hub and Transformers offline flags and skips the proxy preflight, which failed for the first follow-up jobs on 2026-11-15 even though their model weights were available. Do not use offline mode for Qwen until `scripts/prefetch_qwen.py` has cached all four shards. Its CPU-only PBS wrapper is `scripts/pbs/prefetch_qwen.pbs`; the dependent evaluation can then run offline.

`scripts/pbs/eval_bold_chat.pbs` runs a separate instructed continuation condition for BOLD; keep it separate from raw-prefix results. `scripts/pbs/train_canary.pbs` and `scripts/pbs/eval_canary.pbs` run short engineering checks. The dependent `mia_full_pipeline.pbs`, `mia_goldfish.pbs`, and `mia_utility.pbs` scripts use one GPU each and write adapters under ignored `checkpoints/` and scores under ignored `results/`. Full job IDs and dependencies are in `docs/runs.md`.
