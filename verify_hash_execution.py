import random
from problem import myhash

def r(x):
    return x % (2**32)

def myhash_optimized(a: int) -> int:
    """
    Optimized version of myhash using affine simplifications.
    """
    # Stage 1: (a + 0x7ED55D16) + (a << 12)  =>  a * 4097 + 0x7ED55D16
    a = r(a * 4097 + 0x7ED55D16)
    
    # Stage 2: (a ^ 0xC761C23C) ^ (a >> 19)
    a = (a ^ 0xC761C23C) ^ (a >> 19)
    a = r(a)

    # Stage 3: (a + 0x165667B1) + (a << 5)   =>  a * 33 + 0x165667B1
    a = r(a * 33 + 0x165667B1)

    # Stage 4: (a + 0xD3A2646C) ^ (a << 9)   =>  Mixed (No simple affine)
    # Note: problem.py uses fns[op2](r(fns[op1](a, val1)), ...)
    # op1 is +, val1 is 0xD3A2646C. op3 is <<, val3 is 9. op2 is ^.
    term1 = r(a + 0xD3A2646C)
    term2 = r(a << 9)
    a = r(term1 ^ term2)

    # Stage 5: (a + 0xFD7046C5) + (a << 3)   =>  a * 9 + 0xFD7046C5
    a = r(a * 9 + 0xFD7046C5)

    # Stage 6: (a ^ 0xB55A4F09) ^ (a >> 16)
    a = (a ^ 0xB55A4F09) ^ (a >> 16)
    a = r(a)
    
    return a

def packet_verification():
    print("Verifying hash function equivalence...")
    
    # Test specific edge cases
    test_inputs = [0, 1, 0xFFFFFFFF, 0xAAAAAAAA, 0x55555555, 123456789]
    for x in test_inputs:
        orig = myhash(x)
        opt = myhash_optimized(x)
        if orig != opt:
            print(f"MISMATCH for input {x}: Original={orig:#x}, Optimized={opt:#x}")
            return
            
    # Test random inputs
    print("Testing 100,000 random inputs...")
    for _ in range(100000):
        x = random.randint(0, 2**32 - 1)
        orig = myhash(x)
        opt = myhash_optimized(x)
        if orig != opt:
            print(f"MISMATCH for input {x}: Original={orig:#x}, Optimized={opt:#x}")
            return

    print("SUCCESS: 100,000 random inputs verified.")

if __name__ == "__main__":
    packet_verification()
