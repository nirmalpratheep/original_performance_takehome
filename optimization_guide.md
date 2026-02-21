# Performance Optimization Guide: Anthropic Take-Home

This document explains the objective, the simulated hardware, and the baseline code in detail.

## 1. The Objective
We need to optimize a "kernel" (a small, performance-critical piece of code) that performs a **Tree Traversal with Hashing**.
- **Current Performance:** ~147,734 cycles (Scalar Baseline)
- **Target Performance:** < 1,500 cycles (Vectorized & Pipelined)

## 2. The Algorithm
For a batch of inputs (indices and values):
1.  **Read** the current node value from the tree using the index.
2.  **Hash** the input value combined with the node value.
3.  **Update** the index to the left or right child based on the hash result.
4.  **Repeat** for a fixed number of rounds (depth).

## 3. The Simulated Hardware (VLIW & SIMD)
The code runs on a simulated **VLIW (Very Long Instruction Word)** processor with **SIMD (Single Instruction, Multiple Data)** capabilities.

### Key Constraints (`problem.py`)
-   **VLEN = 8**: You can process 8 values at once using `valu`, `vload`, `vstore`.
-   **Parallel Slots**: In a single clock cycle, you can execute multiple operations:
    -   `alu`: 12 scalar arithmetic ops
    -   `valu`: 6 vector arithmetic ops
    -   `load`: 2 memory loads
    -   `store`: 2 memory stores
    -   `flow`: 1 control flow (jump/branch) op

### Optimization Strategy
To achieve the target speedup, we must:
1.  **Vectorize**: Process 8 items at a time instead of 1.
2.  **Pipeline**: Execute loads, hashes, and arithmetic in parallel slots every cycle.

---

## 4. Baseline Kernel Walkthrough (`perf_takehome.py`)

Here is a line-by-line explanation of the `build_kernel` function starting at line 88.

### Function Signature & Setup
**Lines 88-90**: `def build_kernel(...)`
Defines the function to build the instruction stream. It takes parameters for the tree height, number of nodes, batch size, and number of rounds.

**Lines 95-97**: Allocating Scratch Space (Registers)
Allocates temporary registers `tmp1`, `tmp2`, `tmp3` for intermediate calculations.

**Lines 99-112**: Loading Parameters
Loads the initial constants (pointers to memory arrays) into scratch registers so we can access them later.

**Lines 114-116**: Constants
Creates constants `0`, `1`, `2` in registers for frequently used values.

### The Main Loop
**Line 134**: `for round in range(rounds):`
Outer loop: iterates through the depth of traversal.

**Line 135**: `for i in range(batch_size):`
Inner loop: iterates through each item in the batch **sequentially**. This is the main bottleneck. The optimized version will process `i` in steps of 8 (0, 8, 16...).

### The Inner Loop Body (Per Item)

**Lines 136-140**: Load Current Index
```python
136: i_const = self.scratch_const(i)
138: body.append(("alu", ("+", tmp_addr, self.scratch["inp_indices_p"], i_const)))
139: body.append(("load", ("load", tmp_idx, tmp_addr)))
```
-   Calculates the address of `indices[i]`.
-   Loads the current index for item `i` into `tmp_idx`.

**Lines 141-144**: Load Current Value
```python
142: body.append(("alu", ("+", tmp_addr, self.scratch["inp_values_p"], i_const)))
143: body.append(("load", ("load", tmp_val, tmp_addr)))
```
-   Calculates the address of `values[i]`.
-   Loads the current value for item `i` into `tmp_val`.

**Lines 145-148**: Load Tree Node
```python
146: body.append(("alu", ("+", tmp_addr, self.scratch["forest_values_p"], tmp_idx)))
147: body.append(("load", ("load", tmp_node_val, tmp_addr)))
```
-   Uses `tmp_idx` to find the address of the tree node.
-   Loads the node value into `tmp_node_val`. **Latency Alert:** This load depends on the previous load of `tmp_idx`.

**Lines 149-152**: Hash Calculation
```python
150: body.append(("alu", ("^", tmp_val, tmp_val, tmp_node_val)))
151: body.extend(self.build_hash(tmp_val, tmp1, tmp2, round, i))
```
-   XORs the value with the node value.
-   Calls `build_hash` which generates a sequence of ALU operations to mix the bits. This is computationally expensive.

**Lines 153-159**: Calculate Next Index
```python
154: body.append(("alu", ("%", tmp1, tmp_val, two_const))) # Check parity (even/odd)
156: body.append(("flow", ("select", tmp3, tmp1, one_const, two_const))) # Select 1 or 2
157: body.append(("alu", ("*", tmp_idx, tmp_idx, two_const))) # idx * 2
158: body.append(("alu", ("+", tmp_idx, tmp_idx, tmp3))) # idx * 2 + (1 or 2)
```
-   Determines if we go left (add 1) or right (add 2) based on the hash.

**Lines 160-163**: Wrap Around (Modulo)
```python
161: body.append(("alu", ("<", tmp1, tmp_idx, self.scratch["n_nodes"])))
162: body.append(("flow", ("select", tmp_idx, tmp1, tmp_idx, zero_const)))
```
-   If the new index is outside the tree, wrap it back to 0.

**Lines 164-169**: Write Back Results
```python
166: body.append(("store", ("store", tmp_addr, tmp_idx)))
169: body.append(("store", ("store", tmp_addr, tmp_val)))
```
-   Stores the updated index and value back to memory.

## Summary of Inefficiencies
1.  **Sequential Execution**: It processes one item, finishes it, then moves to the next.
2.  **Underutilized Slots**: Most instructions barely use the available slots (12 ALUs, 2 loads, etc.).
3.  **Latency Stalls**: The code waits for memory loads (line 147) and hash computations (line 151) instead of doing other useful work while waiting.
