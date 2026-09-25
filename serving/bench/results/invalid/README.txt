Runs kept out of the table because the engine was wrong when they were taken.
1789368948  decode recorded over a cache that still moved; the recording read a
            stale cache. 159.2 tok/s and the wrong text.
1789369015  cache preallocated but the fill pointer never advanced, so every
            decode step overwrote column 0 and saw a one-token context.
Both are kept rather than deleted — they are what a fast wrong answer looks like.
