# Simulator Architecture: Detailed Reference

Source file: `problem.py` — class `Machine`.

---

## Overview

The simulator models a **VLIW SIMD processor** — a custom architecture that is
neither x86 nor ARM. You write programs for it by hand (as Python lists of dicts)
and it executes them cycle-by-cycle.

Two key concepts:

| Concept | Meaning |
|---------|---------|
| **VLIW** | Every cycle issues one "bundle" containing slots for ALL engines simultaneously |
| **SIMD** | Vector instructions operate on 8 elements (VLEN=8) in a single slot |

---

## Memory Model

### What is `mem`?

`mem` is a flat Python list of 32-bit integers. Its size for the benchmark test
(`forest_height=10, batch_size=256`) is calculated in `build_mem_image()`:

```python
header     = 7
n_nodes    = 2^(10+1) - 1 = 2047   (tree nodes)
batch_size = 256
extra_room = 2047 + 256*2 + 8*2 + 32 = 2607

mem size = 7 + 2047 + 256 + 256 + 2607 = 5173 words = ~20 KB
```

Layout of `mem`:

```
Index  0  : rounds           = 16
Index  1  : n_nodes          = 2047
Index  2  : batch_size       = 256
Index  3  : forest_height    = 10
Index  4  : forest_values_p  = 7       (pointer → tree data starts here)
Index  5  : inp_indices_p    = 2054    (pointer → indices array)
Index  6  : inp_values_p     = 2310    (pointer → values array)
─────────────────────────────────────────
Index  7  .. 2053 : tree node values   (2047 words)
Index  2054.. 2309 : input indices     (256 words)
Index  2310.. 2565 : input values      (256 words)
Index  2566.. 5172 : extra room        (scratch/output space)
```

- You **never index mem directly by a constant** in your program. You always load the
  pointer from scratch (e.g. `scratch[forest_values_p]` = 7) and add to it.
- Results are read back from `mem` by the test harness at the end.

### What is `scratch`?

`scratch` is a second flat list of 1536 32-bit integers — your register file.
You allocate named slots with `alloc_scratch()`. All operand addresses in
instructions are indices into `scratch`, not into `mem`.

```
scratch[0]  = tmp1          ← allocated by alloc_scratch("tmp1")
scratch[1]  = tmp2
scratch[2]  = tmp3
scratch[3]  = rounds_var    ← loaded from mem[0] at init
scratch[4]  = n_nodes_var   ← loaded from mem[1] at init
scratch[5]  = forest_height_var
scratch[6]  = forest_values_p  ← value = 7 (a pointer into mem)
scratch[7]  = inp_indices_p    ← value = 2054
scratch[8]  = inp_values_p     ← value = 2310
scratch[9]  = const_0       ← value = 0
scratch[10] = const_1       ← value = 1
scratch[11] = const_2       ← value = 2
...
```

---

## What is `dest` in a Slot?

Every slot format is: `(op, dest, src1, src2)`

**`dest`, `src1`, `src2` are all indices into the `scratch` array — not memory
addresses and not literal values.**

```
scratch = [0, 0, 0, 0, 0, ...]
           ^  ^  ^  ^
           0  1  2  3  ...   ← these numbers are the "addresses"
```

Concrete example — the instruction `("^", 2, 0, 1)`:

```
op   = "^"     (XOR operation)
dest = 2       (write result to scratch[2])
src1 = 0       (read first operand from scratch[0])
src2 = 1       (read second operand from scratch[1])

Effect: scratch[2] = (scratch[0] ^ scratch[1]) % 2**32
```

Another example — `("+", 5, 6, 9)`:

```
op   = "+"    (add)
dest = 5      (scratch[5] gets the result)
src1 = 6      (scratch[6] holds forest_values_p = 7)
src2 = 9      (scratch[9] holds const_0 = 0)

Effect: scratch[5] = (7 + 0) % 2**32 = 7
```

> Think of scratch addresses like variable names.
> `dest=5` means "store the result in the variable at slot 5".
> `src1=6` means "read the value from the variable at slot 6".

---

## Execution Model: The Cycle

Every program is a list of **instruction bundles**. Each bundle is a Python dict:

```python
{
    "alu":   [slot, slot, ...],    # up to 12 slots
    "valu":  [slot, slot, ...],    # up to 6 slots
    "load":  [slot, slot],         # up to 2 slots
    "store": [slot, slot],         # up to 2 slots
    "flow":  [slot],               # up to 1 slot
    "debug": [slot, ...],          # unlimited, ignored in submission
}
```

**In a single cycle:**
- ALL engines execute ALL their slots **simultaneously**
- Reads happen first (from current scratch/mem values)
- Writes commit at **end of cycle** (scratch_write / mem_write buffers)
- A slot in one engine cannot see a write from another slot in the **same cycle**

```
Cycle N:
  ┌────────────────────────────────────────────────────────┐
  │ Read phase: ALL engines read from scratch & mem        │
  │   alu[0]  reads scratch[a], scratch[b]                 │
  │   valu[0] reads scratch[x..x+7]                        │
  │   load[0] reads mem[scratch[addr]]                     │
  │   (all simultaneously, in parallel)                    │
  ├────────────────────────────────────────────────────────┤
  │ Write phase: ALL results commit to scratch & mem       │
  │   scratch[dest]      ← alu result                      │
  │   scratch[dest..+7]  ← valu result (8 words)           │
  │   scratch[dest]      ← loaded value                    │
  └────────────────────────────────────────────────────────┘
       ↓
  Cycle N+1 sees the new scratch values
```

The cycle counter increments only if the bundle has at least one non-debug slot.

---

## The Five Engines and Their Slot Limits

```
SLOT_LIMITS = {
    "alu":   12,   ← scalar math
    "valu":  6,    ← vector math (8-wide)
    "load":  2,    ← memory reads
    "store": 2,    ← memory writes
    "flow":  1,    ← control flow
    "debug": 64,   ← assertions (ignored in submission)
}
```

---

### 1. ALU Engine (12 slots/cycle) — Scalar Math

Slot format: `(op, dest, src1, src2)` — all three numbers are scratch addresses.

Result is always `% 2**32` (wraps to 32-bit unsigned).

| Op | Meaning |
|----|---------|
| `+` | add |
| `-` | subtract |
| `*` | multiply |
| `//` | integer divide |
| `cdiv` | ceiling divide |
| `^` | bitwise XOR |
| `&` | bitwise AND |
| `\|` | bitwise OR |
| `<<` | left shift |
| `>>` | right shift |
| `%` | modulo |
| `<` | less-than → 0 or 1 |
| `==` | equal → 0 or 1 |

**Example — 12 ALU slots in one cycle:**

```python
# scratch[10] = inp_indices_p  (holds value 2054)
# scratch[11] = const_i_0      (holds 0)
# scratch[12] = const_i_1      (holds 1)
# scratch[13] = const_i_2      (holds 2)
# ...

{
  "alu": [
    ("+", 20, 10, 11),   # scratch[20] = 2054 + 0 = 2054  (addr for element 0)
    ("+", 21, 10, 12),   # scratch[21] = 2054 + 1 = 2055  (addr for element 1)
    ("+", 22, 10, 13),   # scratch[22] = 2054 + 2 = 2056  (addr for element 2)
    ("+", 23, 10, 14),   # scratch[23] = 2054 + 3 = 2057  (addr for element 3)
    ("+", 24, 10, 15),   # scratch[24] = 2054 + 4 = 2058  (addr for element 4)
    ("+", 25, 10, 16),   # scratch[25] = 2054 + 5 = 2059  (addr for element 5)
    ("+", 26, 10, 17),   # scratch[26] = 2054 + 6 = 2060  (addr for element 6)
    ("+", 27, 10, 18),   # scratch[27] = 2054 + 7 = 2061  (addr for element 7)
    ("+", 28, 10, 19),   # scratch[28] = 2054 + 8 = 2062  (addr for element 8)
    ("+", 29, 10, 30),   # ...
    ("+", 31, 10, 32),
    ("+", 33, 10, 34),
  ]
}
# All 12 address computations happen IN PARALLEL in 1 cycle.
```

---

### 2. VALU Engine (6 slots/cycle) — Vector Math (VLEN=8)

#### What is VALU?

VALU is the **Vector ALU** — it does the same operation as ALU but on **8 numbers
at once** instead of 1. Each VALU slot occupies one block of 8 consecutive scratch
addresses.

If ALU is like processing one box, VALU is like processing a pallet of 8 boxes
simultaneously.

```
ALU  slot:  scratch[dest]   = op(scratch[src1],   scratch[src2])
                               ← 1 result

VALU slot:  scratch[dest+0] = op(scratch[src1+0], scratch[src2+0])
            scratch[dest+1] = op(scratch[src1+1], scratch[src2+1])
            scratch[dest+2] = op(scratch[src1+2], scratch[src2+2])
            scratch[dest+3] = op(scratch[src1+3], scratch[src2+3])
            scratch[dest+4] = op(scratch[src1+4], scratch[src2+4])
            scratch[dest+5] = op(scratch[src1+5], scratch[src2+5])
            scratch[dest+6] = op(scratch[src1+6], scratch[src2+6])
            scratch[dest+7] = op(scratch[src1+7], scratch[src2+7])
                               ← 8 results from 1 slot
```

#### What does "6 slots per cycle" mean?

It means in ONE cycle you can issue up to **6 VALU slots simultaneously**.
Since each slot processes 8 elements, you get **6 × 8 = 48 element-operations per cycle**.

**Concrete example: hashing 48 elements in 1 cycle**

Say you have 6 groups of 8 elements, each needing `val = val ^ node_val`:

```
scratch layout:
  val_group0  = scratch[100..107]   ← 8 values, group 0
  val_group1  = scratch[108..115]   ← 8 values, group 1
  val_group2  = scratch[116..123]   ← 8 values, group 2
  val_group3  = scratch[124..131]   ← 8 values, group 3
  val_group4  = scratch[132..139]   ← 8 values, group 4
  val_group5  = scratch[140..147]   ← 8 values, group 5

  node_group0 = scratch[200..207]   ← 8 node values, group 0
  node_group1 = scratch[208..215]
  ... etc
```

One VALU bundle:

```python
{
  "valu": [
    ("^", 100, 100, 200),  # val_group0[0..7] ^= node_group0[0..7]  (8 XORs)
    ("^", 108, 108, 208),  # val_group1[0..7] ^= node_group1[0..7]  (8 XORs)
    ("^", 116, 116, 216),  # val_group2[0..7] ^= node_group2[0..7]  (8 XORs)
    ("^", 124, 124, 224),  # val_group3[0..7] ^= node_group3[0..7]  (8 XORs)
    ("^", 132, 132, 232),  # val_group4[0..7] ^= node_group4[0..7]  (8 XORs)
    ("^", 140, 140, 240),  # val_group5[0..7] ^= node_group5[0..7]  (8 XORs)
  ]
}
# Total: 6 slots × 8 lanes = 48 XOR operations in 1 cycle
```

#### Why have 6 slots instead of 1?

Because within a single hash computation the stages are **serial** (each stage depends
on the previous). You cannot put 6 hash steps for the SAME group in one cycle.
But you CAN process 6 **different** groups simultaneously. This is called
**software pipelining / interleaving**:

```
Cycle 1:  group0-stage1  group1-stage1  group2-stage1  group3-stage1  group4-stage1  group5-stage1
Cycle 2:  group0-stage2  group1-stage2  group2-stage2  group3-stage2  group4-stage2  group5-stage2
Cycle 3:  group0-stage3  group1-stage3  group2-stage3  group3-stage3  group4-stage3  group5-stage3
...
```

Every cycle all 6 VALU slots are busy with a different group.
Without interleaving (1 group at a time) you waste 5 of the 6 slots every cycle.

#### Special VALU ops

| Op | Format | Meaning |
|----|--------|---------|
| `vbroadcast` | `(dest, scalar_src)` | Copy 1 scalar → 8 identical values in `dest..dest+7` |
| `multiply_add` | `(dest, a, b, c)` | `(a[i] * b[i] + c[i]) % 2**32` per lane — fused multiply-add |

`vbroadcast` example:
```python
# scratch[5] = 12345  (a scalar constant)
# After: ("vbroadcast", 100, 5)
# scratch[100] = scratch[101] = ... = scratch[107] = 12345
```

`multiply_add` example — hash stage simplification `a*4097 + C`:
```python
# scratch[C_4097]     = scalar 4097
# scratch[C_hash1]    = scalar 0x7ED55D16
# vbroadcast first:
("vbroadcast", 50, C_4097),     # scratch[50..57] = all 4097
("vbroadcast", 60, C_hash1),    # scratch[60..67] = all 0x7ED55D16
# Then fused multiply-add on all 8 lanes:
("multiply_add", 100, 100, 50, 60)
# scratch[100+i] = (scratch[100+i] * 4097 + 0x7ED55D16) % 2**32  for i=0..7
# = 1 VALU slot replaces 3 ALU ops per element = 24 ALU ops saved for 8 elements
```

---

### 3. Load Engine (2 slots/cycle) — Memory Reads

| Slot | Format | What it does |
|------|--------|--------------|
| `load` | `(dest, addr)` | `scratch[dest] = mem[scratch[addr]]` — scalar load |
| `load_offset` | `(dest, addr, offset)` | `scratch[dest+offset] = mem[scratch[addr+offset]]` |
| `vload` | `(dest, addr)` | Load 8 contiguous words: `scratch[dest..dest+7] = mem[scratch[addr]..+7]` |
| `const` | `(dest, val)` | `scratch[dest] = val % 2**32` — load immediate |

> `vload` loads 8 **contiguous** memory words. There is no scatter/gather — to load 8
> non-contiguous addresses you need 8 separate scalar `load` ops = 4 load cycles.

---

### 4. Store Engine (2 slots/cycle) — Memory Writes

| Slot | Format | What it does |
|------|--------|--------------|
| `store` | `(addr, src)` | `mem[scratch[addr]] = scratch[src]` |
| `vstore` | `(addr, src)` | `mem[scratch[addr]..+7] = scratch[src..src+7]` |

---

### 5. Flow Engine (1 slot/cycle) — Control

| Slot | Format | What it does |
|------|--------|--------------|
| `select` | `(dest, cond, a, b)` | `scratch[dest] = scratch[a] if scratch[cond]!=0 else scratch[b]` |
| `vselect` | `(dest, cond, a, b)` | Same but 8-wide, lane-by-lane |
| `add_imm` | `(dest, a, imm)` | `scratch[dest] = (scratch[a] + imm) % 2**32` |
| `cond_jump` | `(cond, addr)` | if `scratch[cond] != 0`: jump to absolute PC |
| `cond_jump_rel` | `(cond, offset)` | if `scratch[cond] != 0`: `pc += offset` |
| `jump` | `(addr,)` | unconditional jump |
| `jump_indirect` | `(addr,)` | `pc = scratch[addr]` |
| `halt` | `()` | stop the core |
| `pause` | `()` | pause (used to sync with `yield` in reference) |

---

### 6. Debug Engine (unlimited, free)

```python
("compare",  loc, key)   # assert scratch[loc] == value_trace[key]
("vcompare", loc, keys)  # assert scratch[loc..+7] == [value_trace[k] for k in keys]
```

Zero cycles in submission mode. Use freely during development.

---

## Cycle Counting Rules

```python
if any(name != "debug" for name in instr.keys()):
    self.cycle += 1
```

- Debug-only bundle: **0 cycles**
- Any real engine present: **1 cycle**

---

## Maximum Throughput Per Cycle

| Engine | Slots/cycle | Elements | Scalar-equiv ops |
|--------|-------------|----------|-----------------|
| ALU    | 12          | 1 each   | 12              |
| VALU   | 6           | 8 each   | **48**          |
| Load   | 2           | —        | 2               |
| Store  | 2           | —        | 2               |
| Flow   | 1           | —        | 1               |
| **Total** |          |          | **~65**         |

---

## Key Constraints That Limit Performance

### 1. No forwarding within a cycle
If slot A writes `scratch[5]` and slot B reads `scratch[5]` in the **same bundle**,
slot B reads the **old** value. The write only takes effect next cycle.

### 2. Load latency = 1 cycle
```
Cycle N:   load("load", dest=10, addr=5)    ← result NOT yet in scratch[10]
Cycle N+1: alu("+", dest=11, src1=10, ...)  ← NOW scratch[10] is valid
```

### 3. Scatter/gather = multiple load cycles
8 non-contiguous loads need 8 scalar `load` slots = 4 cycles minimum.

### 4. Scratch space = 1536 words only
For the benchmark: 256 indices + 256 values = 512 words just for element state.
Only ~1024 words left for temporaries and constants.

### 5. Flow engine = 1 slot/cycle
Only one branch or select per cycle, even `vselect` which operates on 8 lanes.

---

## Summary Table

| Property | Value |
|----------|-------|
| Word size | 32-bit unsigned |
| `mem` size (benchmark) | 5173 words (~20 KB) |
| `scratch` size | 1536 words |
| VLEN (vector width) | 8 lanes |
| ALU slots/cycle | 12 (scalar, 1 element/slot) |
| VALU slots/cycle | 6 (vector, **8 elements/slot**) |
| Load slots/cycle | 2 |
| Store slots/cycle | 2 |
| Flow slots/cycle | 1 |
| Max ops/cycle | ~65 scalar-equivalent |
| Write visible | next cycle (no same-cycle forwarding) |
| Cores | 1 (N_CORES=1) |