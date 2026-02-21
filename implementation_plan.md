# Implementation Plan - Vectorization

## Goal
Optimize the `build_kernel` function in `perf_takehome.py` to use SIMD instructions, processing 8 elements per iteration.

## Proposed Changes

### 1. `perf_takehome.py`

#### [MODIFY] `perf_takehome.py`
-   **Update `KernelBuilder` helper methods**:
    -   Add `alloc_scratch_vec(length=VLEN)` to allocate aligned vector registers.
    -   Add `scratch_const_vec(val)` to broadcast constants to vectors.
    -   Update `build_hash` to support vector `valu` operations instead of scalar `alu`.

-   **Rewrite `build_kernel`**:
    -   **Loop Structure**: Change the inner loop to stride by 8 (`range(0, batch_size, 8)`).
    -   **Load Operations**:
        -   Use `vload` to load 8 indices and 8 values at once.
        -   Gather support: The hardware documentation mentions `vload` for contiguous loads. For loading tree nodes based on indices (which are not contiguous), we might need to load them individually or use a gather-like pattern if available (or just 8 scalar loads if `vgather` isn't supported).
        -   *Note*: `problem.py` `load` engine has `load` (scalar), `load_offset`, and `vload` (contiguous vector). It does **not** seem to have a `vgather`. We will likely need to issue **8 scalar loads** for the tree nodes, or use `load_offset` if we can organize data. Since tree nodes are random access, we probably need 8 scalar loads. `load` engine has 2 slots, so we can do 2 loads per cycle.
    -   **Hash Calculation**:
        -   Convert the hash chain to use `valu` (vector ALU) ops (`^`, `+`, `<<`, `>>` work on vectors too).
    -   **Logic & Control Flow**:
        -   Use `vselect` to handle the conditional logic (odd/even branch) for all 8 lanes.
        -   Use `valu` comparison `<` for the wrap-around logic.
    -   **Store Operations**:
        -   Use `vstore` to write back updated indices and values (they are contiguous in the input array).

## Verification Plan

### Automated Tests
-   **Correctness**: Run `python perf_takehome.py Tests.test_kernel_correctness` (need to uncomment this test in `perf_takehome.py` or run it dynamically).
-   **Performance**: Run `python perf_takehome.py Tests.test_kernel_cycles` and observe the cycle count.
    -   **Target**: Significant reduction from 147k cycles.

### Manual Verification
-   Inspect `trace.json` if debugging is needed.
