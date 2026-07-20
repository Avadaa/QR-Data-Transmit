// PCG64.swift — bit-exact port of numpy's default_rng(int) chain: SeedSequence ->
// PCG64 (setseq_128_xsl_rr) -> Generator.integers for power-of-2 ranges (32-bit Lemire
// on a buffered next32 stream, low half first). Validated against numpy on the PC
// (pcg_check.py) and self-tested on device against pcg_ref.json. The wire depends on
// this being EXACT: training truth = default_rng(TRAIN_SEED+idx), repair masks =
// default_rng(REPAIR_SEED+j).
import Foundation

struct SeedSeq {
    var pool = [UInt32](repeating: 0, count: 4)

    init(_ entropy: Int) {
        var hc: UInt32 = 0x43b0d7e5
        func hashmix(_ v0: UInt32) -> UInt32 {
            var v = v0 ^ hc
            hc = hc &* 0x931e8875
            v = v &* hc
            v ^= v >> 16
            return v
        }
        func mix(_ x: UInt32, _ y: UInt32) -> UInt32 {
            var r = (x &* 0xca01f9dd) &- (y &* 0x4973f715)   // SUBTRACT, not xor
            r ^= r >> 16
            return r
        }
        var words = [UInt32(truncatingIfNeeded: entropy)]
        if entropy >> 32 != 0 { words.append(UInt32(truncatingIfNeeded: entropy >> 32)) }
        for i in 0 ..< 4 { pool[i] = hashmix(i < words.count ? words[i] : 0) }
        for s in 0 ..< 4 { for d in 0 ..< 4 where s != d { pool[d] = mix(pool[d], hashmix(pool[s])) } }
    }

    func generateState(_ n: Int) -> [UInt32] {
        var hc: UInt32 = 0x8b51f9dd
        var out = [UInt32]()
        for i in 0 ..< n {
            var v = pool[i % 4] ^ hc
            hc = hc &* 0x58f38ded
            v = v &* hc
            v ^= v >> 16
            out.append(v)
        }
        return out
    }
}

struct PCG64 {
    var hi: UInt64 = 0, lo: UInt64 = 0          // 128-bit state
    var incHi: UInt64, incLo: UInt64
    var buf: UInt32?
    static let mHi: UInt64 = 0x2360ed051fc65da4, mLo: UInt64 = 0x4385df649fccf645

    init(_ seed: Int) {
        let w = SeedSeq(seed).generateState(8)
        let u = (0 ..< 4).map { UInt64(w[2 * $0]) | (UInt64(w[2 * $0 + 1]) << 32) }
        incHi = (u[2] << 1) | (u[3] >> 63)      // inc = (initseq << 1) | 1
        incLo = (u[3] << 1) | 1
        step()
        let (l, ov) = lo.addingReportingOverflow(u[1])   // state += initstate
        lo = l; hi = hi &+ u[0] &+ (ov ? 1 : 0)
        step()
    }

    mutating func step() {                       // state = state * MULT + inc mod 2^128
        let m = lo.multipliedFullWidth(by: PCG64.mLo)
        let h = m.high &+ lo &* PCG64.mHi &+ hi &* PCG64.mLo
        let (l, ov) = m.low.addingReportingOverflow(incLo)
        lo = l; hi = h &+ incHi &+ (ov ? 1 : 0)
    }

    mutating func next64() -> UInt64 {           // step FIRST, then xsl-rr output
        step()
        let rot = hi >> 58
        let x = hi ^ lo
        return rot == 0 ? x : (x >> rot) | (x << (64 - rot))
    }

    mutating func next32() -> UInt32 {           // buffered halves: LOW first, then high
        if let b = buf { buf = nil; return b }
        let d = next64()
        buf = UInt32(truncatingIfNeeded: d >> 32)
        return UInt32(truncatingIfNeeded: d)
    }

    // Generator.integers(0, hi) for power-of-2 hi <= 2^32: Lemire, rejection never fires
    mutating func integers(_ hiEx: Int, _ n: Int) -> [UInt8] {
        (0 ..< n).map { _ in UInt8((UInt64(next32()) &* UInt64(hiEx)) >> 32) }
    }
}
