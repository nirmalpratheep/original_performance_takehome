# Basic Iteration: Operation Count in `perf_takehome.py`

Reference: baseline scalar `build_kernel` + `build_hash` in `perf_takehome.py`.

---

## Per Single Element, Per Round

### Phase 1 — Load data (7 ops)

| # | Code | Engine |
|---|------|--------|
| 1 | `addr = inp_indices_p + i` | alu `+` |
| 2 | `idx = mem[addr]` | load |
| 3 | `addr = inp_values_p + i` | alu `+` |
| 4 | `val = mem[addr]` | load |
| 5 | `addr = forest_values_p + idx` | alu `+` |
| 6 | `node_val = mem[addr]` | load |
| 7 | `val = val ^ node_val` | alu `^` |

### Phase 2 — Hash: `build_hash()` (18 ops)

6 stages × 3 alu each. Formula per stage: `a = op2( op1(a, val1), op3(a, val3) )`

| Stage | Instr 1 | Instr 2 | Instr 3 |
|-------|---------|---------|---------|
| 0 | `tmp1 = a + 0x7ED55D16` | `tmp2 = a << 12` | `a = tmp1 + tmp2` |
| 1 | `tmp1 = a ^ 0xC761C23C` | `tmp2 = a >> 19` | `a = tmp1 ^ tmp2` |
| 2 | `tmp1 = a + 0x165667B1` | `tmp2 = a << 5`  | `a = tmp1 + tmp2` |
| 3 | `tmp1 = a + 0xD3A2646C` | `tmp2 = a << 9`  | `a = tmp1 ^ tmp2` |
| 4 | `tmp1 = a + 0xFD7046C5` | `tmp2 = a << 3`  | `a = tmp1 + tmp2` |
| 5 | `tmp1 = a ^ 0xB55A4F09` | `tmp2 = a >> 16` | `a = tmp1 ^ tmp2` |

Within each stage: instrs 1 & 2 are independent (both read `a`), instr 3 waits for both.
Across stages: each stage waits for the previous stage's instr 3.

**Hash total: 6 × 3 = 18 alu ops**

### Phase 3 — Index update + write-back (11 ops)

| # | Code | Engine |
|---|------|--------|
| 8  | `tmp1 = val % 2`            | alu `%`    |
| 9  | `tmp1 = (tmp1 == 0)`        | alu `==`   |
| 10 | `tmp3 = select(tmp1, 1, 2)` | flow select |
| 11 | `idx = idx * 2`             | alu `*`    |
| 12 | `idx = idx + tmp3`          | alu `+`    |
| 13 | `tmp1 = (idx < n_nodes)`    | alu `<`    |
| 14 | `idx = select(tmp1, idx, 0)`| flow select |
| 15 | `addr = inp_indices_p + i`  | alu `+`    |
| 16 | `store(addr, idx)`          | store      |
| 17 | `addr = inp_values_p + i`   | alu `+`    |
| 18 | `store(addr, val)`          | store      |

---

## Total Per Element Per Round

| Engine | Count | Source |
|--------|-------|--------|
| alu    | 29    | 4 (phase 1) + 18 (hash) + 7 (phase 3) |
| load   | 3     | phase 1 |
| flow   | 2     | phase 3 |
| store  | 2     | phase 3 |
| **Total** | **36** | |

---

## Why Each Op = 1 Cycle (Scalar Baseline)

`build()` packs exactly **one slot per instruction bundle**:

```python
def build(self, slots, vliw=False):
    for engine, slot in slots:
        instrs.append({engine: [slot]})   # one op per cycle
```

No parallelism is exploited. Every op runs alone in its own cycle.

**36 ops → 36 cycles per element.**

---

## Total Cycles Calculation

```
Elements per round : 256
Rounds             : 16
Ops per element    : 36
---------------------------------
Main loop          : 256 × 16 × 36 = 147,456 cycles
Init overhead      :                      ~278 cycles
---------------------------------
Total              :               ≈ 147,734 cycles  ← matches BASELINE
```

---

## Note on `theoretical_limits.md` count of 25

That document counts only **alu ops** (18 hash + 7 address/index alu) and ignores the
3 loads, 2 flow selects, and 2 stores. The true total across all engines is **36 ops**.