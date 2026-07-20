// Demap.swift — the v3.1 payload demapper (PAM-2 DCT cells), forward-only:
// conv3x3(3->32) relu pool2, conv3x3(32->64) relu pool2, fc(1600->512) relu,
// fc(512 -> 3*NC*2). Input = a cell's 20x20 canonical crop (cell + 2px margin —
// the C=20 window, so canonical-scaled px12/px30 grids feed the SAME model).
// Weights raw fp32 in torch state_dict order (test/pictrain/realv3/export_demap.py).
// CPU only for now (batched over cells across cores); ANE port is the next step.
import Accelerate
import Foundation

final class Demap {
    let nc: Int                              // coefficients per channel (8 / 12)
    var w1 = [Float](), b1 = [Float]()       // (32,27)   (32)
    var w2 = [Float](), b2 = [Float]()       // (64,288)  (64)
    var w3 = [Float](), b3 = [Float]()       // (512,1600)(512)
    var w4 = [Float](), b4 = [Float]()       // (3*nc*2,512) (3*nc*2)

    init?(_ d: Data, nc: Int) {
        self.nc = nc
        let counts = [32 * 27, 32, 64 * 288, 64, 512 * 1600, 512,
                      3 * nc * 2 * 512, 3 * nc * 2]
        guard d.count == counts.reduce(0, +) * 4 else { return nil }
        var off = 0
        for (i, kp) in [\Demap.w1, \Demap.b1, \Demap.w2, \Demap.b2,
                        \Demap.w3, \Demap.b3, \Demap.w4, \Demap.b4].enumerated() {
            self[keyPath: kp] = d.subdata(in: off ..< off + counts[i] * 4)
                .withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
            off += counts[i] * 4
        }
    }

    func data() -> Data {
        var d = Data()
        for p in [w1, b1, w2, b2, w3, b3, w4, b4] {
            p.withUnsafeBufferPointer { d.append(Data(buffer: $0)) }
        }
        return d
    }

    // X = T cells x (3,20,20) normalized floats; bits out = T x (3*nc), 0/1 (PAM-2
    // level index: 1 = +3). ~3 MFLOP/cell — spread over all cores.
    func classify(_ X: [Float], _ T: Int, into bits: inout [UInt8]) {
        let no = 3 * nc * 2
        bits.withUnsafeMutableBufferPointer { bp in
            X.withUnsafeBufferPointer { xp in
                DispatchQueue.concurrentPerform(iterations: 8) { k in
                    var col1 = [Float](repeating: 0, count: 27 * 400)
                    var a1 = [Float](repeating: 0, count: 32 * 400)
                    var p1 = [Float](repeating: 0, count: 32 * 100)
                    var i1 = [Int32](repeating: 0, count: 32 * 100)
                    var col2 = [Float](repeating: 0, count: 288 * 100)
                    var a2 = [Float](repeating: 0, count: 64 * 100)
                    var p2 = [Float](repeating: 0, count: 64 * 25)
                    var i2 = [Int32](repeating: 0, count: 64 * 25)
                    var h = [Float](repeating: 0, count: 512)
                    var o = [Float](repeating: 0, count: no)
                    var x = [Float](repeating: 0, count: 1200)
                    let per = (T + 7) / 8
                    for t in k * per ..< min((k + 1) * per, T) {
                        for j in 0 ..< 1200 { x[j] = xp[t * 1200 + j] }
                        Net.im2col(x, 3, 20, 20, into: &col1)
                        Net.gemm(false, false, 32, 400, 27, w1, 27, col1, 400, 0, &a1, 400)
                        for c in 0 ..< 32 { for i in 0 ..< 400 {
                            a1[c * 400 + i] = max(0, a1[c * 400 + i] + b1[c]) } }
                        Net.pool2(a1, 32, 20, into: &p1, &i1)
                        Net.im2col(p1, 32, 10, 10, into: &col2)
                        Net.gemm(false, false, 64, 100, 288, w2, 288, col2, 100, 0, &a2, 100)
                        for c in 0 ..< 64 { for i in 0 ..< 100 {
                            a2[c * 100 + i] = max(0, a2[c * 100 + i] + b2[c]) } }
                        Net.pool2(a2, 64, 10, into: &p2, &i2)
                        Net.gemm(false, false, 512, 1, 1600, w3, 1600, p2, 1, 0, &h, 1)
                        for i in 0 ..< 512 { h[i] = max(0, h[i] + b3[i]) }
                        Net.gemm(false, false, no, 1, 512, w4, 512, h, 1, 0, &o, 1)
                        for j in 0 ..< 3 * nc {
                            bp[t * 3 * nc + j] =
                                o[2 * j + 1] + b4[2 * j + 1] > o[2 * j] + b4[2 * j] ? 1 : 0
                        }
                    }
                }
            }
        }
    }
}
