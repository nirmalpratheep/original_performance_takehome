"""
Detailed explanation and step-by-step trace of the myhash function.
Run this file to see exactly what happens at every stage.
"""

HASH_STAGES = [
    ("+", 0x7ED55D16, "+", "<<", 12),
    ("^", 0xC761C23C, "^", ">>", 19),
    ("+", 0x165667B1, "+", "<<", 5),
    ("+", 0xD3A2646C, "^", "<<", 9),
    ("+", 0xFD7046C5, "+", "<<", 3),
    ("^", 0xB55A4F09, "^", ">>", 16),
]

fns = {
    "+": lambda x, y: x + y,
    "^": lambda x, y: x ^ y,
    "<<": lambda x, y: x << y,
    ">>": lambda x, y: x >> y,
}

fn_names = {
    "+": "ADD",
    "^": "XOR",
    "<<": "LEFT SHIFT",
    ">>": "RIGHT SHIFT",
}

def r(x):
    """Truncate to 32 bits (simulate hardware register overflow)"""
    return x % (2**32)

def myhash_traced(a: int) -> int:
    print(f"{'='*70}")
    print(f"  INPUT: a = {a} (hex: 0x{a:08X})")
    print(f"{'='*70}")
    
    for stage_num, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
        print(f"\n--- Stage {stage_num} ---")
        print(f"  Formula: a = {fn_names[op2]}( {fn_names[op1]}(a, 0x{val1:08X}), {fn_names[op3]}(a, {val3}) )")
        print(f"  a before = {a} (0x{a:08X})")
        
        # Step 1: left side = op1(a, val1)
        left = r(fns[op1](a, val1))
        print(f"    Instruction 1 (left):  {fn_names[op1]}(a={a}, 0x{val1:08X}) = {left} (0x{left:08X})")
        
        # Step 2: right side = op3(a, val3)  [CAN RUN IN PARALLEL with Step 1!]
        right = r(fns[op3](a, val3))
        print(f"    Instruction 2 (right): {fn_names[op3]}(a={a}, {val3}) = {right} (0x{right:08X})")
        
        # Step 3: combine = op2(left, right)  [MUST WAIT for Steps 1 & 2]
        a = r(fns[op2](left, right))
        print(f"    Instruction 3 (combine): {fn_names[op2]}({left}, {right}) = {a} (0x{a:08X})")
        print(f"  a after  = {a} (0x{a:08X})")
    
    print(f"\n{'='*70}")
    print(f"  OUTPUT: a = {a} (hex: 0x{a:08X})")
    print(f"{'='*70}")
    return a


def benchmark():
    """Show that different inputs produce wildly different outputs (good hash)"""
    print("\n\n" + "="*70)
    print("  BENCHMARK: Hash Distribution")
    print("="*70)
    print(f"  {'Input':>15}  ->  {'Output (hex)':>15}  {'Output (dec)':>15}")
    print(f"  {'-'*15}      {'-'*15}  {'-'*15}")
    
    test_inputs = [0, 1, 2, 3, 100, 255, 1000, 0x7FFFFFFF, 0xDEADBEEF]
    for inp in test_inputs:
        out = myhash_simple(inp)
        print(f"  {inp:>15}  ->  0x{out:08X}       {out:>15}")
    
    # Show avalanche effect: tiny change in input -> huge change in output
    print(f"\n  AVALANCHE EFFECT (1-bit difference in input):")
    print(f"  {'Input':>15}  ->  {'Output (binary, last 16 bits)':>35}")
    print(f"  {'-'*15}      {'-'*35}")
    for inp in [100, 101]:
        out = myhash_simple(inp)
        print(f"  {inp:>15}  ->  ...{out:016b}  (0x{out:08X})")
    bits_different = bin(myhash_simple(100) ^ myhash_simple(101)).count('1')
    print(f"  Bits that changed: {bits_different} out of 32")


def myhash_simple(a: int) -> int:
    """Same hash, no printing"""
    for op1, val1, op2, op3, val3 in HASH_STAGES:
        a = r(fns[op2](r(fns[op1](a, val1)), r(fns[op3](a, val3))))
    return a


if __name__ == "__main__":
    # Trace with a concrete input
    print("EXAMPLE 1: Small input")
    myhash_traced(42)
    
    print("\n\nEXAMPLE 2: Larger input (typical XOR result)")
    myhash_traced(0xDEADBEEF ^ 0x12345678)
    
    benchmark()
