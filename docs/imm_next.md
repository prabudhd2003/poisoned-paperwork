# Immediate next task: accept the Donut attack smoke on CARC

The clean validation baseline is already accepted. Do not rerun it. The next
gate is the 20-receipt Donut targeted-attack smoke owned by Khalid (`user4`).
Production remains deliberately disabled in `configs/donut_attack/active.json`
until the smoke evidence below is reviewed.

## Frozen input provenance

The Donut configs require the accepted baseline artifacts exactly:

```text
run: baseline_validation_v1
prediction records: 527
inference Git commit: 5e387062d29b2a9b2b79de03c305c52d7b97f01f
baseline config SHA-256: a371655bf97c99e6c08ac5da2b288ca48bf301ad0c2bdb21e4260c588266d05a
predictions.parquet SHA-256: 483e12b88167555d94e943d0179314de25016333e409417b34a6f922c46caeb0
```

The attack stops if the saved predictions or merge manifest do not match these
values. It also reruns clean inference for every selected receipt and stops on
prediction drift.

## Step 1: update and run CPU checks

From the CARC login node:

```bash
cd /project2/yzhao010_1531/csci_699_new_arch/poisoned-paperwork
git pull origin main
git status --short
source scripts/activate_paperwork.sh
export PYTHONPATH="$PWD/src"
python -m unittest discover -s tests -v
sha256sum outputs/baseline/baseline_validation_v1/merged/predictions.parquet
python scripts/smoke_donut_attack.py \
  --config configs/donut_attack/smoke_validation.json --preflight
```

Do not submit unless the worktree is clean, all tests pass, the printed SHA-256
matches the value above, and preflight selects exactly ten clean-correct SROIE
and ten clean-correct CORD receipts.

## Step 2: submit the unattended smoke job

```bash
sbatch scripts/slurm/donut_smoke.sbatch \
  configs/donut_attack/smoke_validation.json
```

Record the returned job ID. It is safe to disconnect after submission.

```bash
squeue -j <job_id>
sacct -j <job_id> --format=JobID,State,Elapsed,ExitCode,NodeList
tail -n 100 logs/pp_donut_smoke_khalid_<job_id>.out
tail -n 100 logs/pp_donut_smoke_khalid_<job_id>.err
```

Success requires Slurm state `COMPLETED`, exit code `0:0`, and the final smoke
completion message.

## Step 3: inspect the smoke artifacts

```bash
RUN=outputs/donut_attack/donut_validation_smoke_v1
cat "$RUN/environment.json"
cat "$RUN/merged/summary.json"
cat "$RUN/merged/failures.csv"
column -s, -t < "$RUN/merged/attacks.csv" | less -S
find "$RUN/shards/worker_00/images" -type f | sort
find "$RUN/shards/worker_00/records" -type f | sort
```

Acceptance checklist:

- 20 unique documents and 20 final epsilon-8 attack records;
- no inference failures or clean-prediction drift;
- processor parity is below the configured tolerance and is documented;
- every trace contains finite loss, target log probability, gradient norms,
  prediction, exact-target status, and observed L-infinity distance;
- every saved/reloaded image has `linf_255 <= 8.0001`;
- LPIPS and SSIM are finite and visually plausible;
- in-memory and reloaded success are reported separately;
- five clean/adversarial/magnified-difference examples are inspected; and
- unsuccessful attacks are retained and reported rather than tuned away.

## Step 4: prove checkpoint resume

On the first smoke attempt, wait until at least one `state.pt` checkpoint exists,
then cancel the Slurm job once with `scancel <job_id>`. Resubmit the exact same
config and confirm that the corresponding final record contains
`checkpoint_resumed=true` and a continuous trace. Do not edit or delete any
checkpoint, config, baseline artifact, or completed record. The remaining
uninterrupted smoke examples provide the comparison for bounds and metrics.

## Step 5: unlock and submit validation production

After the team records smoke acceptance, edit only this field in
`configs/donut_attack/active.json`:

```json
"ready_for_production": true
```

Commit and push that change. All five teammates then pull the same commit and
submit exactly one shard:

```bash
bash scripts/submit_stage.sh donut_attack user1
bash scripts/submit_stage.sh donut_attack user2
bash scripts/submit_stage.sh donut_attack user3
bash scripts/submit_stage.sh donut_attack user4
bash scripts/submit_stage.sh donut_attack user5
```

The active validation config attacks all 121 Donut clean-correct eligible
receipts at epsilon 2, 4, 8, and 16, for 484 expected records. If a worker times
out, the same teammate submits the same command again. Do not edit the config or
run ID between retries.

## Step 6: merge and accept validation

The last worker attempts the merge automatically. It can also be checked on a
CPU login node:

```bash
python scripts/merge_donut_attack.py \
  --config configs/donut_attack/active.json
cat outputs/donut_attack/donut_validation_v1/merged/summary.json
cat outputs/donut_attack/donut_validation_v1/merged/merge_manifest.json
```

Acceptance requires 484 exact records, five correct worker markers, identical
Git/config/baseline hashes, no missing or duplicate experiment keys, no epsilon
violations, and explicit accounting of every failure.

Only after validation analysis freezes the Donut settings should a separate
test config be created with `mode="test"`, `frozen_for_test=true`, and the
accepted validation run ID.
