#!/usr/bin/env python3
# GGUF tensor probe: names, shapes, types. Usage: python3 gguf_probe.py <model.gguf> [prefix]
# prefix filters tensor names (e.g. "blk.64." or blank for all), case-sensitive.
from gguf import GGUFReader
from collections import Counter
import sys

P = sys.argv[1] if len(sys.argv) > 1 else "model.gguf"
PREFIX = sys.argv[2] if len(sys.argv) > 2 else ""
g = GGUFReader(P)
infos = [(t.name, t.shape, t.tensor_type) for t in g.tensors]
print("total tensors:", len(infos))
print("type histogram:", dict(Counter(x[2] for x in infos)))
sc = [n for n, _, _ in infos if n.endswith("_s") or "_scale" in n]
print(f"scale-like tensors: {len(sc)}")
for n in sc[:20]:
    print("  ", n)
print()
for name, sh, t in sorted(x for x in infos if x[0].startswith(PREFIX)):
    print(f"  {name:40s} shape={sh} type={t}")
