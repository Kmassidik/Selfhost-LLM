# R18 + R19 (fresh measurements)

## R18 — KV-cache quantization sweep (extraction, hard set)
| KV precision | EXACT | note |
|---|---|---|
| f16 | 99.5% | baseline |
| q8_0 | 96.8% | nearly free, halves KV memory (2x context) |
| q4_0 | 1.0% | COLLAPSE |
KV cache is far more quant-sensitive than weights: q8 is safe (-2.7pt, doubles
context capacity), q4 destroys the model. (Weights survived Q2 in R1 — KV does not.)

## R19 — tensor (row) vs layer split, 2 cards, Llama-3-8B Q4
- layer split: prefill 2217 tok/s, decode 79.6 tok/s. Works.
- row (tensor) split: FAILS TO LOAD. On this PCIe-only box (no NVLink, cards
  refuse peer access) tensor/row split is not viable — layer split is the only
  option. Confirms Part IV.
