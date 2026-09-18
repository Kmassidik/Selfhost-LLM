"""The formula predicted more saving than appeared. Why?

Chapter 07's cache formula multiplies by "bytes per value". A quantized block
is not purely values: llama.cpp stores 32 numbers per block plus a scale factor
to turn them back into real numbers. The scale is overhead the formula ignores.
"""
L, KVH, HD, CTX = 32, 8, 128, 8192
slots = 2 * L * KVH * HD * CTX          # keys and values, every layer, every token

BLOCKS = {                               # llama.cpp block layouts
    "f16":  ("2 bytes per value, no block structure",        2.0,    2.0),
    "q8_0": ("32 values in 32 bytes + one 2-byte scale",     1.0,    34/32),
    "q4_0": ("32 values in 16 bytes + one 2-byte scale",     0.5,    18/32),
}
print(f"cache slots: 2 x {L} layers x {KVH} kv heads x {HD} head dim x {CTX:,} tokens")
print(f"           = {slots:,} values\n")
print(f"{'type':<7}{'layout':<42}{'naive MB':>10}{'real MB':>10}{'overhead':>10}")
print("-" * 80)
sizes = {}
for t, (desc, naive, real) in BLOCKS.items():
    n_mb, r_mb = slots*naive/1e6, slots*real/1e6
    sizes[t] = r_mb
    print(f"{t:<7}{desc:<42}{n_mb:>10.0f}{r_mb:>10.0f}{(real/naive-1)*100:>9.1f}%")

print()
print(f"{'change':<16}{'predicted naive':>17}{'predicted real':>16}{'measured':>11}{'error':>9}")
print("-" * 70)
for t, meas in (("q8_0", 480), ("q4_0", 736)):
    naive_save = slots*(BLOCKS['f16'][1]-BLOCKS[t][1])/1e6
    real_save  = sizes['f16'] - sizes[t]
    print(f"{'f16 -> '+t:<16}{naive_save:>16.0f}M{real_save:>15.0f}M{meas:>10}M"
          f"{(meas-real_save)/real_save*100:>8.1f}%")
print(f"""
  Including the scale factors the prediction lands within a few percent.
  Chapter 07's formula is right about the values and silent about the block
  overhead, which is {(34/32/1.0-1)*100:.1f}% for q8_0 and {(18/32/0.5-1)*100:.1f}% for q4_0 —
  small, and the reason a 4-bit cache is not exactly half of an 8-bit one.""")
