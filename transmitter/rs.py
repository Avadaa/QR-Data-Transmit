# rs.py — Reed-Solomon over GF(256), numpy-vectorized across equal-size blocks.
# Systematic RS(n, n-nsym): corrects up to nsym//2 byte errors per block.
import numpy as np

_EXP = np.zeros(512, np.int32)
_LOG = np.zeros(256, np.int32)
x = 1
for i in range(255):
    _EXP[i] = x; _LOG[x] = i
    x <<= 1
    if x & 0x100: x ^= 0x11d
_EXP[255:510] = _EXP[:255]

def _mul(a, b):            # elementwise GF mul (arrays)
    out = _EXP[_LOG[a] + _LOG[b]]
    return np.where((a == 0) | (b == 0), 0, out)

def _gen_poly(nsym):
    g = np.array([1], np.int32)
    for i in range(nsym):
        g2 = np.zeros(len(g) + 1, np.int32)
        g2[:-1] ^= g                        # g * x
        g2[1:] ^= _mul(g, _EXP[i])          # + g * alpha^i
        g = g2
    return g

def encode(data, nsym):    # data (nblk, k) -> (nblk, k+nsym)
    g = _gen_poly(nsym)[1:]
    nblk, k = data.shape
    par = np.zeros((nblk, nsym), np.int32)
    for j in range(k):
        f = data[:, j] ^ par[:, 0]
        par = np.roll(par, -1, 1); par[:, -1] = 0
        par ^= _mul(f[:, None], g[None, :])
    return np.concatenate([data, par], 1).astype(np.uint8)

_SYN_T = {}                # (n, nsym) -> log-power matrix, built once

def _syndromes(code, nsym):
    # S_j = C(alpha^j) for all blocks/syndromes at once. This used to be a Horner loop
    # over the 255 columns (255 python iterations of _mul per decode) — profiled at
    # ~40ms/frame on the Surface, paid even by CLEAN frames, since syndromes are always
    # computed. Direct evaluation instead: S_j = XOR_i c_i * alpha^(j*(n-1-i)). The
    # log of each power IS (j*(n-1-i)) mod 255 — no table lookup to build it — so the
    # whole thing is one fancy-index EXP[LOG[c] + T], a zero-mask (log of 0 is
    # undefined; GF mul by 0 is 0), and an XOR-reduce. ~19x64x255 int32 = 1.2MB, ~2ms.
    nblk, n = code.shape
    if (n, nsym) not in _SYN_T:
        _SYN_T[(n, nsym)] = (np.arange(nsym)[:, None] * (n - 1 - np.arange(n))[None, :]) % 255
    prod = _EXP[_LOG[code][:, None, :] + _SYN_T[(n, nsym)][None, :, :]]
    return np.bitwise_xor.reduce(np.where(code[:, None, :] == 0, 0, prod), 2)

def _solve(V, s):          # GF Gaussian elimination: V e = s, V (L,L)
    L = V.shape[0]
    A = np.concatenate([V, s[:, None]], 1).astype(np.int32)
    for col in range(L):
        piv = next((r for r in range(col, L) if A[r, col]), None)
        if piv is None: return None
        A[[col, piv]] = A[[piv, col]]
        A[col] = _mul(A[col], _EXP[255 - _LOG[A[col, col]]])
        for r in range(L):
            if r != col and A[r, col]:
                f = A[r, col].copy()
                A[r] ^= _mul(f, A[col])
    return A[:, -1]

def decode(code, nsym):    # (nblk, n) -> (data, ok mask)
    code = code.astype(np.int32)
    nblk, n = code.shape
    syn = _syndromes(code, nsym)
    ok = ~syn.any(1)
    fixed = []
    EXPL, LOGL = _EXP.tolist(), _LOG.tolist()   # python ints: ~10x faster in the BM loop
    for bi in np.where(~ok)[0]:
        s = syn[bi].tolist()
        C, B, L, m, b = [1], [1], 0, 1, 1       # Berlekamp-Massey
        for i in range(nsym):
            d = s[i]
            for j in range(1, min(L, len(C) - 1) + 1):
                cj, sij = C[j], s[i - j]
                if cj and sij: d ^= EXPL[LOGL[cj] + LOGL[sij]]
            if d == 0:
                m += 1; continue
            coef = (LOGL[d] - LOGL[b]) % 255
            Bp = [0] * m + [EXPL[LOGL[x] + coef] if x else 0 for x in B]
            Cn = C + [0] * max(0, len(Bp) - len(C))
            for j, x in enumerate(Bp): Cn[j] ^= x
            if 2 * L <= i:
                L, B, b, m = i + 1 - L, C[:], d, 1
            else:
                m += 1
            C = Cn
        if L > nsym // 2: continue
        C = np.array(C, np.int32)
        e_arr = np.arange(n)             # Chien: sigma(alpha^-e) == 0 -> error at power e
        val = np.zeros(n, np.int32)
        for j, cf in enumerate(C):
            if cf: val ^= _mul(int(cf), _EXP[(-e_arr * j) % 255])
        powers = e_arr[val == 0]
        if len(powers) != L: continue
        # magnitudes: solve S_j = sum_k e_k * (alpha^{p_k})^j  for j = 0..L-1
        V = np.array([[_EXP[(p * j) % 255] for p in powers] for j in range(L)], np.int32)
        ev = _solve(V, np.array(s[:L], np.int32))
        if ev is None: continue
        code[bi, n - 1 - powers] ^= ev
        fixed.append(bi)
    if fixed:                            # verify corrected blocks in one vectorized pass
        chk = _syndromes(code[fixed], nsym)
        ok[np.array(fixed)[~chk.any(1)]] = True
    return code[:, :n - nsym].astype(np.uint8), ok

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    for nsym in (8, 16, 32):
        data = rng.integers(0, 256, (40, 200 - nsym)).astype(np.uint8)
        code = encode(data.astype(np.int32), nsym)
        nerr = nsym // 2
        corr = code.copy()
        for bi in range(40):
            idx = rng.choice(200, nerr, replace=False)
            corr[bi, idx] ^= rng.integers(1, 256, nerr).astype(np.uint8)
        dec, ok = decode(corr, nsym)
        print(f"nsym={nsym}: ok={ok.all()} exact={np.array_equal(dec, data)}")
