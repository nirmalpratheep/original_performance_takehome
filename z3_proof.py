import z3

def check_equivalence():
    s = z3.Solver()
    
    # input variable
    a = z3.BitVec('a', 32)
    
    # --- STAGE 1 ---
    # Original: (+, 0x7ED55D16, +, <<, 12)
    # a = (a + 0x7ED55D16) + (a << 12)
    val1 = 0x7ED55D16
    orig_stage1 = (a + val1) + (a << 12)
    
    # Proposed: a * 4097 + 0x7ED55D16
    # 4097 = 2^12 + 1
    prop_stage1 = a * 4097 + val1
    
    s.add(orig_stage1 != prop_stage1)
    if s.check() == z3.unsat:
        print("Stage 1 Simplification Verified: a * 4097 + C")
    else:
        print("Stage 1 Simplification FAILED")
        print(s.model())

    # --- STAGE 3 ---
    # Original: (+, 0x165667B1, +, <<, 5)
    # a = (a + 0x165667B1) + (a << 5)
    s.reset()
    val3 = 0x165667B1
    orig_stage3 = (a + val3) + (a << 5)
    # 2^5 + 1 = 33
    prop_stage3 = a * 33 + val3
    
    s.add(orig_stage3 != prop_stage3)
    if s.check() == z3.unsat:
        print("Stage 3 Simplification Verified: a * 33 + C")
    else:
        print("Stage 3 Simplification FAILED")

    # --- STAGE 5 ---
    # Original: (+, 0xFD7046C5, +, <<, 3)
    # a = (a + 0xFD7046C5) + (a << 3)
    s.reset()
    val5 = 0xFD7046C5
    orig_stage5 = (a + val5) + (a << 3)
    # 2^3 + 1 = 9
    prop_stage5 = a * 9 + val5
    
    s.add(orig_stage5 != prop_stage5)
    if s.check() == z3.unsat:
        print("Stage 5 Simplification Verified: a * 9 + C")
    else:
        print("Stage 5 Simplification FAILED")

    # --- STAGE 4 (Mixed) ---
    # Original: (+, 0xD3A2646C, ^, <<, 9)
    # a = (a + 0xD3A2646C) ^ (a << 9)
    s.reset()
    val4 = 0xD3A2646C
    orig_stage4 = (a + val4) ^ (a << 9)
    
    # Synthesize: Try to find M, K such that orig == a * M + K
    M = z3.BitVec('M', 32)
    K = z3.BitVec('K', 32)
    equation = (orig_stage4 == a * M + K)
    
    # We want this to hold for ALL 'a'.
    # This is a quantifier problem: ForAll(a, equation).
    # Z3 can handle quantifiers.
    
    goal = z3.ForAll([a], equation)
    s.add(goal)
    
    if s.check() == z3.sat:
        print("Stage 4 CAN be simplified to Affine!")
        print(s.model())
    else:
        print("Stage 4 CANNOT be simplified to simple Affine transformation.")

    # Try Linear GF(2): a ^ (a << S) ^ K
    # We can try to synthesize Shift amount S and Constant K
    # But Stage 4 has ADD (+), which is non-linear in GF(2).
    # So it's unlikely to match pure XOR/Shift logic perfectly without carry handling.
    
if __name__ == "__main__":
    check_equivalence()
