---
name: gguf-inspect
description: Inspect a GGUF model's tensor names, shapes, and quant types from WSL/python. Use when you need the model's tensor layout, quant histogram, presence of scale (`_s`) tensors, or the MoE-vs-dense FFN structure, before touching graph/fusion code.
---

# Inspect a GGUF from python

Use the `gguf` python library. Do not hand-roll a header parser - the GGUF binary layout is
fiddly (header is magic(4)+version(u32)+tensor_count(u64)+kv_count(u64), then the kv table,
then per-tensor name/shapes/offset) and a wrong offset table or kv-type table silently walks
off the end of the file. The library handles it.

## Install (PEP 668 / externally-managed python)

`pip install gguf` and `python -m venv` both fail out of the box on a default Ubuntu/WSL
python (externally-managed env, and `ensurepip` absent for venv). Install into the user site:

```
python3 -m pip install --user --break-system-packages gguf
```

Verify: `python3 -c "import gguf"`.

## Reader API (verified)

`GGUFReader(path)` ->
- `.tensors` : list of `ReaderTensor`
- `ReaderTensor` fields: `.name`, `.shape` (tuple of ints, ne order), `.tensor_type`
  (a `GGMLQuantizationType`), `.n_elements`, `.n_bytes`, `.data_offset`, `.field`, `.index`
- There is NO `.dtype` and NO `.gguf_type` - those attribute names raise AttributeError.

Tensor type integer codes seen: F32=0, Q8_0=8, Q6_K=14.

## Probe script

`gguf_probe.py` in this skill dir does the whole job:

```
python3 gguf_probe.py /path/to/model.gguf [name_prefix]
```

Prefix (e.g. `blk.64.`) filters to one block/MTP slice; empty lists everything. It prints the
total count, a type histogram (Counter of tensor_type), the scale-like tensors, then
`name shape type` for the matching tensors.

## What to look for (why you're inspecting)

- **Scale tensors** (names ending `_s` or containing `_scale`): their presence makes
  `build_lora_mm(weight, x, scale)` insert an extra `MUL` right after the GEMV in the graph,
  which breaks naive `{MUL_MAT, <op>}` fusion patterns because the GEMV is no longer directly
  linked to its consumer. A zero count means the GEMV is the last op in its subgraph.
- **MoE vs dense FFN**: per block, dense FFN = `ffn_gate`/`ffn_up`/`ffn_down`; MoE =
  `ffn_gate_inp`/`ffn_gate_exps`/`ffn_up_exps`/`ffn_down_exps` (plus optional `*_shexp`).
  This decides which `build_layer_ffn` branch runs.
- **Quant histogram**: tells you which GEMV path fires (Q6_K -> DMMV, Q4_K -> MMVQ, Q8_0 ->
  its own path) and which fusions are even eligible.
- **MTP block**: the last `blk.N` (N = n_layer) has `nextn.*` tensors; confirm whether its FFN
  is MoE or dense, since `graph_mtp` asserts `ffn_gate_inp` (MoE) and a dense MTP block will
  trip it.
