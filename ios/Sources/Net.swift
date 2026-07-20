// Net.swift — the receiver's tile classifier (loop_L16_best), hand-rolled: conv3x3(3->16)
// relu pool2, conv3x3(16->32) relu pool2, fc(512->64) relu -> heads shape(16) + color(4).
// Forward AND backward (im2col + cblas_sgemm), weights read raw from weights.bin in the
// exact torch state_dict order/layout. No ML framework — the whole model is this file.
import Accelerate
import Foundation

final class Net {
    let k: Int                              // input side: 16 (r16) or 12 (r12 native)
    let counts: [Int]                       // tensor sizes; only the fc layer varies with k
    // torch layouts: conv (out, in, 3, 3) flattened to (out, in*9); fc (out, in)
    var w1 = [Float](), b1 = [Float]()      // (16,27)  (16)
    var w2 = [Float](), b2 = [Float]()      // (32,144) (32)
    var w3 = [Float](), b3 = [Float]()      // (64, 32*(k/4)^2)  (64)
    var ws = [Float](), bs = [Float]()      // (16,64)  (16)
    var wc = [Float](), bc = [Float]()      // (4,64)   (4)
    static let keys: [ReferenceWritableKeyPath<Net, [Float]>] =
        [\.w1, \.b1, \.w2, \.b2, \.w3, \.b3, \.ws, \.bs, \.wc, \.bc]
    static func counts(k: Int) -> [Int] {
        [16 * 27, 16, 32 * 144, 32, 64 * 32 * (k / 4) * (k / 4), 64, 16 * 64, 16, 4 * 64, 4]
    }

    init(_ d: Data, k: Int = 16) {
        self.k = k
        counts = Net.counts(k: k)
        var off = 0
        for (i, kp) in Net.keys.enumerated() {
            let n = counts[i]
            self[keyPath: kp] = d.subdata(in: off ..< off + n * 4).withUnsafeBytes {
                Array($0.bindMemory(to: Float.self))
            }
            off += n * 4
        }
    }

    func perturb(_ sigma: Float) {          // damage every tensor: N(0, sigma * its std)
        for kp in Net.keys {
            var p = self[keyPath: kp]
            let mean = p.reduce(0, +) / Float(p.count)
            let sd = sqrt(p.map { ($0 - mean) * ($0 - mean) }.reduce(0, +) / Float(p.count))
            for j in 0 ..< p.count { p[j] += gauss() * sigma * max(sd, 1e-3) }
            self[keyPath: kp] = p
        }
    }

    // im2col: x (C,H,W) -> col (C*9, H*W) row-major, 3x3 kernel, pad 1
    static func im2col(_ x: [Float], _ C: Int, _ H: Int, _ W: Int, into col: inout [Float]) {
        let N = H * W
        for c in 0 ..< C { for ky in -1 ... 1 { for kx in -1 ... 1 {
            let r = (c * 9 + (ky + 1) * 3 + (kx + 1)) * N
            for y in 0 ..< H {
                let sy = y + ky
                if sy < 0 || sy >= H { for xx in 0 ..< W { col[r + y * W + xx] = 0 }; continue }
                for xx in 0 ..< W {
                    let sx = xx + kx
                    col[r + y * W + xx] = (sx >= 0 && sx < W) ? x[(c * H + sy) * W + sx] : 0
                }
            }
        } } }
    }

    static func gemm(_ tA: Bool, _ tB: Bool, _ m: Int, _ n: Int, _ k: Int,
                     _ a: [Float], _ lda: Int, _ b: [Float], _ ldb: Int,
                     _ beta: Float, _ out: inout [Float], _ ldc: Int) {
        cblas_sgemm(CblasRowMajor, tA ? CblasTrans : CblasNoTrans, tB ? CblasTrans : CblasNoTrans,
                    Int32(m), Int32(n), Int32(k), 1, a, Int32(lda), b, Int32(ldb),
                    beta, &out, Int32(ldc))
    }

    static func pool2(_ x: [Float], _ C: Int, _ H: Int, into out: inout [Float], _ idx: inout [Int32]) {
        let Ho = H / 2
        for c in 0 ..< C { for y in 0 ..< Ho { for xx in 0 ..< Ho {
            var best: Float = -.infinity; var bi = 0
            for dy in 0 ..< 2 { for dx in 0 ..< 2 {
                let i = (c * H + y * 2 + dy) * H + xx * 2 + dx
                if x[i] > best { best = x[i]; bi = i }
            } }
            out[(c * Ho + y) * Ho + xx] = best; idx[(c * Ho + y) * Ho + xx] = Int32(bi)
        } } }
    }

    struct Acts {                           // everything backward needs, per sample
        var col1, a1, p1, col2, a2, p2, h, s, c: [Float]
        var i1, i2: [Int32]
        init(k: Int) {                      // n1 = conv1 plane, n2 = post-pool1 plane
            let n1 = k * k, n2 = (k / 2) * (k / 2), fcIn = 32 * (k / 4) * (k / 4)
            col1 = .init(repeating: 0, count: 27 * n1)
            a1 = .init(repeating: 0, count: 16 * n1)      // conv1 post-relu
            p1 = .init(repeating: 0, count: 16 * n2)
            col2 = .init(repeating: 0, count: 144 * n2)
            i1 = .init(repeating: 0, count: 16 * n2)
            a2 = .init(repeating: 0, count: 32 * n2)      // conv2 post-relu
            p2 = .init(repeating: 0, count: fcIn)         // torch flatten order
            i2 = .init(repeating: 0, count: fcIn)
            h = .init(repeating: 0, count: 64)
            s = .init(repeating: 0, count: 16)
            c = .init(repeating: 0, count: 4)
        }
    }

    func forward(_ x: [Float], _ A: inout Acts) {
        let n1 = k * k, h2 = k / 2, n2 = h2 * h2, fcIn = 32 * (k / 4) * (k / 4)
        Net.im2col(x, 3, k, k, into: &A.col1)
        Net.gemm(false, false, 16, n1, 27, w1, 27, A.col1, n1, 0, &A.a1, n1)
        for o in 0 ..< 16 { for i in 0 ..< n1 { A.a1[o * n1 + i] = max(0, A.a1[o * n1 + i] + b1[o]) } }
        Net.pool2(A.a1, 16, k, into: &A.p1, &A.i1)
        Net.im2col(A.p1, 16, h2, h2, into: &A.col2)
        Net.gemm(false, false, 32, n2, 144, w2, 144, A.col2, n2, 0, &A.a2, n2)
        for o in 0 ..< 32 { for i in 0 ..< n2 { A.a2[o * n2 + i] = max(0, A.a2[o * n2 + i] + b2[o]) } }
        Net.pool2(A.a2, 32, h2, into: &A.p2, &A.i2)
        Net.gemm(false, false, 64, 1, fcIn, w3, fcIn, A.p2, 1, 0, &A.h, 1)
        for o in 0 ..< 64 { A.h[o] = max(0, A.h[o] + b3[o]) }
        Net.gemm(false, false, 16, 1, 64, ws, 64, A.h, 1, 0, &A.s, 1)
        for o in 0 ..< 16 { A.s[o] += bs[o] }
        Net.gemm(false, false, 4, 1, 64, wc, 64, A.h, 1, 0, &A.c, 1)
        for o in 0 ..< 4 { A.c[o] += bc[o] }
    }

    // inference-only batched path: one big sgemm per layer for B tiles (BLAS efficiency
    // beats 5244 tiny per-tile gemms). X = prebuilt (T,3,16,16) tensors, writes syms.
    func forwardBatch(_ X: [Float], _ t0: Int, _ B: Int, into sym: UnsafeMutableBufferPointer<UInt8>) {
        let n1 = 256, n2 = 64, ld1 = B * n1, ld2 = B * n2
        var col1 = [Float](repeating: 0, count: 27 * ld1)
        X.withUnsafeBufferPointer { xp in
            for bb in 0 ..< B {
                let base = (t0 + bb) * 768
                for c in 0 ..< 3 { for ky in -1 ... 1 { for kx in -1 ... 1 {
                    let r = (c * 9 + (ky + 1) * 3 + (kx + 1)) * ld1 + bb * n1
                    for y in 0 ..< 16 {
                        let sy = y + ky
                        if sy < 0 || sy >= 16 { for xx in 0 ..< 16 { col1[r + y * 16 + xx] = 0 }; continue }
                        for xx in 0 ..< 16 {
                            let sx = xx + kx
                            col1[r + y * 16 + xx] = (sx >= 0 && sx < 16) ? xp[base + (c * 16 + sy) * 16 + sx] : 0
                        }
                    }
                } } }
            }
        }
        var a1 = [Float](repeating: 0, count: 16 * ld1)
        Net.gemm(false, false, 16, ld1, 27, w1, 27, col1, ld1, 0, &a1, ld1)
        for o in 0 ..< 16 { let v = b1[o]; for i in 0 ..< ld1 { a1[o * ld1 + i] = max(0, a1[o * ld1 + i] + v) } }
        var p1 = [Float](repeating: 0, count: 16 * ld2)
        for o in 0 ..< 16 { for bb in 0 ..< B {
            let src = o * ld1 + bb * n1, dst = o * ld2 + bb * n2
            for y in 0 ..< 8 { for xx in 0 ..< 8 {
                let i = src + y * 32 + xx * 2
                p1[dst + y * 8 + xx] = max(max(a1[i], a1[i + 1]), max(a1[i + 16], a1[i + 17]))
            } }
        } }
        var col2 = [Float](repeating: 0, count: 144 * ld2)
        for bb in 0 ..< B {
            for c in 0 ..< 16 { for ky in -1 ... 1 { for kx in -1 ... 1 {
                let r = (c * 9 + (ky + 1) * 3 + (kx + 1)) * ld2 + bb * n2
                let src = c * ld2 + bb * n2
                for y in 0 ..< 8 {
                    let sy = y + ky
                    if sy < 0 || sy >= 8 { for xx in 0 ..< 8 { col2[r + y * 8 + xx] = 0 }; continue }
                    for xx in 0 ..< 8 {
                        let sx = xx + kx
                        col2[r + y * 8 + xx] = (sx >= 0 && sx < 8) ? p1[src + sy * 8 + sx] : 0
                    }
                }
            } } }
        }
        var a2 = [Float](repeating: 0, count: 32 * ld2)
        Net.gemm(false, false, 32, ld2, 144, w2, 144, col2, ld2, 0, &a2, ld2)
        for o in 0 ..< 32 { let v = b2[o]; for i in 0 ..< ld2 { a2[o * ld2 + i] = max(0, a2[o * ld2 + i] + v) } }
        var p2m = [Float](repeating: 0, count: B * 512)   // (B,512) in torch flatten order
        for o in 0 ..< 32 { for bb in 0 ..< B {
            let src = o * ld2 + bb * n2
            for y in 0 ..< 4 { for xx in 0 ..< 4 {
                let i = src + y * 16 + xx * 2
                p2m[bb * 512 + o * 16 + y * 4 + xx] = max(max(a2[i], a2[i + 1]), max(a2[i + 8], a2[i + 9]))
            } }
        } }
        var Hm = [Float](repeating: 0, count: B * 64)
        Net.gemm(false, true, B, 64, 512, p2m, 512, w3, 512, 0, &Hm, 64)
        for bb in 0 ..< B { for o in 0 ..< 64 { Hm[bb * 64 + o] = max(0, Hm[bb * 64 + o] + b3[o]) } }
        var S = [Float](repeating: 0, count: B * 16)
        Net.gemm(false, true, B, 16, 64, Hm, 64, ws, 64, 0, &S, 16)
        var C = [Float](repeating: 0, count: B * 4)
        Net.gemm(false, true, B, 4, 64, Hm, 64, wc, 64, 0, &C, 4)
        for bb in 0 ..< B {
            var si = 0; var sv = S[bb * 16] + bs[0]
            for j in 1 ..< 16 { let v = S[bb * 16 + j] + bs[j]; if v > sv { sv = v; si = j } }
            var ci = 0; var cv = C[bb * 4] + bc[0]
            for j in 1 ..< 4 { let v = C[bb * 4 + j] + bc[j]; if v > cv { cv = v; ci = j } }
            sym[t0 + bb] = UInt8(si << 2 | ci)
        }
    }

    final class Grads {
        var t: [[Float]]                    // same order as keys
        init(_ counts: [Int]) { t = counts.map { [Float](repeating: 0, count: $0) } }
        func zero() { for i in 0 ..< t.count { for j in 0 ..< t[i].count { t[i][j] = 0 } } }
    }

    // accumulate d(CE_shape + CE_color)/dW into g (gradients summed over the batch)
    func backward(_ A: inout Acts, _ ys: Int, _ yc: Int, _ g: Grads) {
        let n1 = k * k, h2 = k / 2, n2 = h2 * h2, fcIn = 32 * (k / 4) * (k / 4)
        var ds = softmax(A.s); ds[ys] -= 1
        var dc = softmax(A.c); dc[yc] -= 1
        var dh = [Float](repeating: 0, count: 64)
        for o in 0 ..< 16 { let d = ds[o]; g.t[7][o] += d
            for i in 0 ..< 64 { g.t[6][o * 64 + i] += d * A.h[i]; dh[i] += d * ws[o * 64 + i] } }
        for o in 0 ..< 4 { let d = dc[o]; g.t[9][o] += d
            for i in 0 ..< 64 { g.t[8][o * 64 + i] += d * A.h[i]; dh[i] += d * wc[o * 64 + i] } }
        for i in 0 ..< 64 where A.h[i] <= 0 { dh[i] = 0 }
        var dp2 = [Float](repeating: 0, count: fcIn)
        for o in 0 ..< 64 { let d = dh[o]; if d == 0 { continue }
            g.t[5][o] += d
            for i in 0 ..< fcIn { g.t[4][o * fcIn + i] += d * A.p2[i]; dp2[i] += d * w3[o * fcIn + i] } }
        var da2 = [Float](repeating: 0, count: 32 * n2)     // unpool + relu mask
        for j in 0 ..< fcIn { let i = Int(A.i2[j]); if A.a2[i] > 0 { da2[i] += dp2[j] } }
        Net.gemm(false, true, 32, 144, n2, da2, n2, A.col2, n2, 1, &g.t[2], 144)   // dW2
        for o in 0 ..< 32 { var s: Float = 0; for i in 0 ..< n2 { s += da2[o * n2 + i] }; g.t[3][o] += s }
        var dcol2 = [Float](repeating: 0, count: 144 * n2)
        Net.gemm(true, false, 144, n2, 32, w2, 144, da2, n2, 0, &dcol2, n2)
        var dp1 = [Float](repeating: 0, count: 16 * n2)     // col2im (h2 x h2, pad 1)
        for c in 0 ..< 16 { for ky in -1 ... 1 { for kx in -1 ... 1 {
            let r = (c * 9 + (ky + 1) * 3 + (kx + 1)) * n2
            for y in 0 ..< h2 { let sy = y + ky
                if sy < 0 || sy >= h2 { continue }
                for xx in 0 ..< h2 { let sx = xx + kx
                    if sx >= 0 && sx < h2 { dp1[(c * h2 + sy) * h2 + sx] += dcol2[r + y * h2 + xx] } }
            }
        } } }
        var da1 = [Float](repeating: 0, count: 16 * n1)
        for j in 0 ..< 16 * n2 { let i = Int(A.i1[j]); if A.a1[i] > 0 { da1[i] += dp1[j] } }
        Net.gemm(false, true, 16, 27, n1, da1, n1, A.col1, n1, 1, &g.t[0], 27)  // dW1
        for o in 0 ..< 16 { var s: Float = 0; for i in 0 ..< n1 { s += da1[o * n1 + i] }; g.t[1][o] += s }
    }
}

final class Adam {
    var m: [[Float]], v: [[Float]]
    var t = 0
    var lr: Float = 1e-4

    init(_ counts: [Int]) {
        m = counts.map { [Float](repeating: 0, count: $0) }
        v = counts.map { [Float](repeating: 0, count: $0) }
    }

    func step(_ net: Net, _ g: Net.Grads, batch: Int) {
        t += 1
        let bc1 = 1 - pow(0.9, Float(t)), bc2 = 1 - pow(0.999, Float(t))
        for i in 0 ..< Net.keys.count {
            var p = net[keyPath: Net.keys[i]]
            for j in 0 ..< p.count {
                let gj = g.t[i][j] / Float(batch)
                m[i][j] = 0.9 * m[i][j] + 0.1 * gj
                v[i][j] = 0.999 * v[i][j] + 0.001 * gj * gj
                p[j] -= lr * (m[i][j] / bc1) / (sqrt(v[i][j] / bc2) + 1e-8)
            }
            net[keyPath: Net.keys[i]] = p
        }
    }
}

func softmax(_ x: [Float]) -> [Float] {
    let mx = x.max()!
    let e = x.map { exp($0 - mx) }
    let s = e.reduce(0, +)
    return e.map { $0 / s }
}

func gauss() -> Float {
    sqrt(-2 * log(Float.random(in: 1e-6 ..< 1))) * cos(2 * .pi * Float.random(in: 0 ..< 1))
}
