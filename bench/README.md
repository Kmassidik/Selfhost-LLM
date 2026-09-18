# bench/ — the measuring instrument

Every engine in Part I is measured by `run.py`, with the same prompts, the same
settings and the same six metrics. Nothing is measured by hand.

    python3 bench/run.py --endpoint http://10.0.0.20:11434/v1 \
                         --model smollm2:360m --level L0 --label "L0 ollama bf16"

    python3 bench/table.py          rebuild the comparison from results/

## Why it is shaped this way

**One JSON per run, and the table is generated.** A hand-maintained comparison
drifts from the measurements within a week. `table.py` reads `results/*.json`
and nothing else, so a number that was never measured cannot appear.

**Temperature 0 everywhere.** Without it two runs cannot be compared — any
difference might be chance — and the output hash is meaningless. With it, the
hash column answers the only question that matters at the end of Part I: did
these engines actually compute the same thing?

**Standard library only.** The harness must never fail to run because an
engine's environment installed an incompatible version of something.

**`prompts.json` is frozen.** Changing a prompt invalidates every measurement
taken before the change. To add one, add a new id and keep the old.

## The honest caveats

- **Token counts are estimated** from character length divided by four. Engines
  disagree about what they report and the harness trusts none of them, so the
  tok/s figures are marked `est_`. They are comparable *between engines* because
  the same estimate is applied to all — but they are not exact token counts.
- **VRAM is sampled at 5 Hz** by polling `nvidia-smi`. A spike shorter than
  200 ms can be missed.
- **Context ceiling is not measured here.** Finding it means driving an engine
  until it fails, which belongs in its own script rather than in a run that is
  supposed to be repeatable.
