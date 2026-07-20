// RS.swift — Reed-Solomon over GF(256), straight port of rs.py (encode + full decode:
// syndromes, Berlekamp-Massey, Chien search, magnitudes by GF Gaussian elimination).
// Systematic RS(n, n-nsym), corrects nsym/2 byte errors per block.
import Foundation

enum RS {
    static let tables: (exp: [Int], log: [Int]) = {
        var exp = [Int](repeating: 0, count: 512), log = [Int](repeating: 0, count: 256)
        var x = 1
        for i in 0 ..< 255 {
            exp[i] = x; log[x] = i
            x <<= 1
            if x & 0x100 != 0 { x ^= 0x11d }
        }
        for i in 255 ..< 510 { exp[i] = exp[i - 255] }
        return (exp, log)
    }()

    static func mul(_ a: Int, _ b: Int) -> Int {
        a == 0 || b == 0 ? 0 : tables.exp[tables.log[a] + tables.log[b]]
    }

    static func genPoly(_ nsym: Int) -> [Int] {
        var g = [1]
        for i in 0 ..< nsym {
            var g2 = [Int](repeating: 0, count: g.count + 1)
            for j in 0 ..< g.count { g2[j] ^= g[j] }                       // g * x
            for j in 0 ..< g.count { g2[j + 1] ^= mul(g[j], tables.exp[i]) }
            g = g2
        }
        return g
    }

    static func encode(_ data: [UInt8], _ nsym: Int) -> [UInt8] {   // one block -> +parity
        let g = Array(genPoly(nsym).dropFirst())
        var par = [Int](repeating: 0, count: nsym)
        for d in data {
            let f = Int(d) ^ par[0]
            par.removeFirst(); par.append(0)
            if f != 0 { for j in 0 ..< nsym { par[j] ^= mul(f, g[j]) } }
        }
        return data + par.map { UInt8($0) }
    }

    static func decode(_ codeIn: [UInt8], _ nsym: Int) -> (data: [UInt8], ok: Bool) {
        let n = codeIn.count
        var code = codeIn.map { Int($0) }
        let (EXP, LOG) = tables
        func syndromes() -> [Int] {
            (0 ..< nsym).map { j in
                var s = 0
                for i in 0 ..< n where code[i] != 0 {
                    s ^= EXP[(LOG[code[i]] + j * (n - 1 - i)) % 255]
                }
                return s
            }
        }
        let syn = syndromes()
        if !syn.contains(where: { $0 != 0 }) {
            return (Array(codeIn[0 ..< n - nsym]), true)
        }
        var C = [1], B = [1], L = 0, m = 1, b = 1                    // Berlekamp-Massey
        for i in 0 ..< nsym {
            var d = syn[i]
            let jmax = min(L, C.count - 1)
            if jmax >= 1 {
                for j in 1 ... jmax where j <= i && C[j] != 0 && syn[i - j] != 0 {
                    d ^= EXP[LOG[C[j]] + LOG[syn[i - j]]]
                }
            }
            if d == 0 { m += 1; continue }
            let coef = ((LOG[d] - LOG[b]) % 255 + 255) % 255
            let Bp = [Int](repeating: 0, count: m) + B.map { $0 != 0 ? EXP[LOG[$0] + coef] : 0 }
            var Cn = C + [Int](repeating: 0, count: max(0, Bp.count - C.count))
            for j in 0 ..< Bp.count { Cn[j] ^= Bp[j] }
            if 2 * L <= i { L = i + 1 - L; B = C; b = d; m = 1 } else { m += 1 }
            C = Cn
        }
        if L > nsym / 2 { return (Array(codeIn[0 ..< n - nsym]), false) }
        var powers = [Int]()                                          // Chien search
        for e in 0 ..< n {
            var v = 0
            for (j, cf) in C.enumerated() where cf != 0 {
                v ^= EXP[(LOG[cf] + ((255 - e * j % 255) % 255)) % 255]
            }
            if v == 0 { powers.append(e) }
        }
        if powers.count != L { return (Array(codeIn[0 ..< n - nsym]), false) }
        // magnitudes: solve V e = s, V[j][k] = alpha^(p_k * j), GF Gaussian elimination
        var A = (0 ..< L).map { j in powers.map { EXP[($0 * j) % 255] } + [syn[j]] }
        for col in 0 ..< L {
            guard let piv = (col ..< L).first(where: { A[$0][col] != 0 }) else {
                return (Array(codeIn[0 ..< n - nsym]), false)
            }
            A.swapAt(col, piv)
            let inv = EXP[255 - LOG[A[col][col]]]
            for j in 0 ... L { A[col][j] = mul(A[col][j], inv) }
            for r in 0 ..< L where r != col && A[r][col] != 0 {
                let f = A[r][col]
                for j in 0 ... L { A[r][j] ^= mul(f, A[col][j]) }
            }
        }
        for (k, p) in powers.enumerated() { code[n - 1 - p] ^= A[k][L] }
        for j in 0 ..< nsym {                                         // verify the fix
            var s = 0
            for i in 0 ..< n where code[i] != 0 {
                s ^= EXP[(LOG[code[i]] + j * (n - 1 - i)) % 255]
            }
            if s != 0 { return (Array(codeIn[0 ..< n - nsym]), false) }
        }
        return (code[0 ..< n - nsym].map { UInt8($0) }, true)
    }

    static func selfTest() -> Bool {              // encode -> corrupt nsym/2 -> decode
        var rng = SystemRandomNumberGenerator()
        for nsym in [32, 64] {
            let data = (0 ..< 255 - nsym).map { _ in UInt8.random(in: 0 ... 255, using: &rng) }
            var code = encode(data, nsym)
            var idx = Set<Int>()
            while idx.count < nsym / 2 { idx.insert(Int.random(in: 0 ..< 255, using: &rng)) }
            for i in idx { code[i] ^= UInt8.random(in: 1 ... 255, using: &rng) }
            let (dec, ok) = decode(code, nsym)
            if !ok || dec != data { return false }
        }
        return true
    }
}
