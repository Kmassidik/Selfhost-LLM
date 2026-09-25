# Raw captures

Unprocessed stdout and JSON from benchmark runs, kept because the summarised
numbers in `bench/results/` and the knowledge base are worth nothing without
the output they came from. Nothing here is read by any program.

    gpusample_*.txt    nvidia-smi sampled during a run, one file per split:
                       c1 one card, c3layer pipeline, c3tens tensor,
                       p3/p3n/v1 the vLLM attempts including the failures
    out_*.txt          the run's own stdout, same suffixes
    agent-*.json       agent task results from BEFORE the harness was fixed
    ours.json, r.json  intermediate hash comparisons
    tie.py, run_*.sh   the scripts that produced some of the above

The agent numbers here predate three harness bugs: a model's hallucinated tool
output being fed back into its own transcript, WRITE with no path writing to a
file called "def", and macOS resource forks appearing in every prompt. They are
kept as a record of what was measured at the time, not as a result, and must
not be compared with anything measured after those fixes.
