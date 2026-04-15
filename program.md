# autoresearch

This is an experiment to have the LLM do its own research on TS-CAN.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `apr15`). The branch `autoresearch/<tag>` must not already exist - this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current main branch.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` - repository context.
   - `prepare.py` - fixed constants, runtime utilities, data loading, and evaluation. Do not modify.
   - `train.py` - the file you modify. Model architecture, optimizer, and training loop all live here.
4. **Verify runtime/data exists**: The default experiment uses bundled `ETT-small/ETTh1.csv`. Make sure the `pytorch` conda environment works and the runtime is healthy by running `conda run -n pytorch python prepare.py`. If this fails, fix the environment before continuing.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a single GPU. The training script runs for a **fixed time budget of 5 minutes** by default (wall clock training time, excluding startup/warmup steps). You launch it simply as: `conda run -n pytorch python train.py`.

**What you CAN do:**
- Modify `train.py` - this is the only file you edit. Everything is fair game inside it: architecture, optimizer, hyperparameters, batch size, model size, training loop, and any TS-CAN block design choices.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed evaluation, data loading, runtime constants, and device/data setup.
- Install new packages or add dependencies.
- Modify the evaluation harness. The validation metric produced through `prepare.py` is the ground truth metric.

**The goal is simple: get the lowest val_mse.** Since the time budget is fixed, you don't need to worry about training time too much - it's always 5 minutes by default unless the human deliberately shortens it for a smoke test. Everything inside `train.py` is fair game: change the architecture, the optimizer, the hyperparameters, the batch size, or the model size. The only constraint is that the code runs without crashing and finishes within the time budget.

**VRAM** is a soft constraint. Some increase is acceptable for meaningful `val_mse` gains, but it should not blow up dramatically.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome - that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A tiny improvement that adds hacky complexity is probably not worth it. A tiny improvement from deleting code probably is. An improvement of ~0 with much simpler code is also often worth keeping.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

## Output format

Once the script finishes it prints a summary like this:

```text
---
val_mse:          0.366418
val_mae:          0.395711
test_mse:         0.366418
test_mae:         0.395711
training_seconds: 300.0
total_seconds:    318.2
peak_vram_mb:     4096.0
num_steps:        240
num_params_M:     0.321
batch_size:       8
optimizer:        adam
```

Note that the script is configured to stop after the fixed wall-clock budget, so depending on the computer the numbers may differ. You can extract the key metric from the log file:

```powershell
Select-String -Pattern "^val_mse:|^peak_vram_mb:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated - commas break in descriptions).

The TSV has a header row and 5 columns:

```tsv
commit	val_mse	memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. `val_mse` achieved (e.g. `0.366418`) - use `0.000000` for crashes
3. peak memory in GB, round to `.1f` (divide `peak_vram_mb` by `1024`) - use `0.0` for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

Example:

```tsv
commit	val_mse	memory_gb	status	description
a1b2c3d	0.366418	4.0	keep	baseline
b2c3d4e	0.360500	4.2	keep	raise model width
c3d4e5f	0.372000	4.0	discard	switch off temporal interaction
d4e5f6g	0.000000	0.0	crash	double width caused OOM
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/apr15` or `autoresearch/apr15-gpu0`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on.
2. Tune `train.py` with an experimental idea by directly hacking the code.
3. git commit.
4. Run the experiment: `conda run -n pytorch python train.py > run.log 2>&1` (redirect everything - do NOT let output flood your context).
5. Read out the results: `Select-String -Pattern "^val_mse:|^peak_vram_mb:" run.log`.
6. If the command output is empty, the run crashed. Run `Get-Content run.log -Tail 50` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the results in the TSV (NOTE: do not commit the `results.tsv` file, leave it untracked by git).
8. If `val_mse` improved (lower), you "advance" the branch, keeping the git commit.
9. If `val_mse` is equal or worse, you git reset back to where you started.

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Timeout**: Each experiment should take about 5 minutes total, plus a bit of startup and eval overhead. If a run exceeds 10 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, a bug, etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log `crash` as the status in the TSV, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, away from the computer, or expecting you to continue working indefinitely until manually stopped. You are autonomous. If you run out of ideas, think harder - reread the in-scope files, combine previous near-misses, and try more radical architectural changes inside `train.py`. The loop runs until the human interrupts you, period.
