# configs/ — architecture only, no weights

Each folder here holds a model's `config.json` and nothing else. A few kilobytes
instead of gigabytes.

**Why this exists.** A model's parameter count, memory footprint and speed ceiling
are all derivable from its configuration alone — that method was validated against
a real file (SmolLM2-360M) and came out exact, to the byte. So architectures can be
studied, compared and budgeted **without downloading the weights.**

    models/
      hf/        full models — weights present, can actually run
      configs/   architecture only — cannot run, only reasoned about
      gguf/      llama.cpp format

Never put weights in here, and never assume a folder in here can be loaded.

Fetch another with:

    ./fetch.sh <org>/<model>
