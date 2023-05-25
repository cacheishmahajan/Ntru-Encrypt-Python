import inverse as inverse
import numpy as np

def polAdd(pol1,pol2,mod):
    pol1, pol2 = inverse.resize(pol1, pol2)
    return [(a + b) % mod for a, b in zip(pol1, pol2)]

def polSub(pol1,pol2,mod):
    pol1, pol2 = inverse.resize(pol1, pol2)
    return [(a - b) % mod for a, b in zip(pol1, pol2)]


def star_multiply(p1, p2, q):
    """Multiply two polynomials in Z_q[X]/(X**n - 1)"""
    p1, p2 = inverse.resize(p1, p2)
    n = len(p1)
    out = [sum(p1[i] * p2[(k - i) % n] for i in range(n)) % q for k in range(n)]
    return out
   
   