// Runner.swift — the receiver's decode path on one bundled Surface frame: perspective
// warp (Hinv from the PC pipeline) -> 20x20 tile crops -> Net over all tiles -> header
// majority+CRC -> compare against the PC's reference symbols and the RS-proven truth.
// Train demo: perturb the weights, then recover them by training on this frame's tiles
// (truth symbols = labels), the receiver's exact augmentation and Adam schedule.
import CoreML
import Foundation
import UIKit

struct Meta: Decodable {
    let cols: Int, rows: Int, px: Int, K: Int, C: Int, idx: Int, nsym: Int
    let pc_err: Double, Hinv: [Double]
}

final class Runner: ObservableObject {
    @Published var log = ""
    @Published var busy = false
    var meta: Meta!
    var net: Net!
    var crops = [UInt8]()          // T x 20x20x3, field row-major
    var refSym = [UInt8](), truthSym = [UInt8]()
    var T = 0
    var srcPix = [UInt8](), sw = 0, sh = 0   // camera frame RGBA (warp bench reuses it)

    static let logURL = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        .appendingPathComponent("log.txt")     // pulled from the PC via devicectl copy

    func out(_ s: String) { DispatchQueue.main.async {
        self.log += s + "\n"
        try? self.log.write(to: Self.logURL, atomically: true, encoding: .utf8)
    } }

    func bg(_ work: @escaping () -> Void) {
        busy = true
        DispatchQueue.global(qos: .userInitiated).async {
            work()
            DispatchQueue.main.async { self.busy = false }
        }
    }

    func asset(_ name: String, _ ext: String) -> Data {
        try! Data(contentsOf: Bundle.main.url(forResource: name, withExtension: ext)!)
    }

    func boot() {
        guard meta == nil else { return }
        bg {
            self.meta = try! JSONDecoder().decode(Meta.self, from: self.asset("meta", "json"))
            self.net = Net(self.asset("weights", "bin"))
            self.refSym = [UInt8](self.asset("ref_sym", "bin"))
            self.truthSym = [UInt8](self.asset("truth_sym", "bin"))
            self.T = self.meta.cols * self.meta.rows
            self.out("frame idx \(self.meta.idx), grid \(self.meta.cols)x\(self.meta.rows) r\(self.meta.px) = \(self.T) tiles")
            let t0 = Date()
            let img = UIImage(data: self.asset("frame", "jpg"))!
            let field = self.warp(img)
            self.cutCrops(field)
            self.out(String(format: "warp+crops  %.0f ms", -t0.timeIntervalSinceNow * 1000))
            self.runDecode()
            self.runWarpBench()
        }
    }

    // pure-NN benchmark, comparable to docs/iGPU_CPU_performance.md ("one frame" = the
    // rung's real tile batch, median of 12): inputs prebuilt, forward only, chunk sweep
    func runBench() {
        self.out("\n-- bench: r16 field = \(T) tiles, fp32 forward only, median of 12 --")
        var X = [Float](repeating: 0, count: T * 768)
        let t0 = Date()
        var xt = [Float](repeating: 0, count: 768)
        for t in 0 ..< T {
            input(t, 2, 2, &xt)
            for j in 0 ..< 768 { X[t * 768 + j] = xt[j] }
        }
        out(String(format: "prep %.0f ms (crops -> tensors, single core)", -t0.timeIntervalSinceNow * 1000))
        for chunks in [1, 2, 4, 6, 16] {
            var times = [Double]()
            for _ in 0 ..< 12 {
                let t1 = Date()
                let per = (T + chunks - 1) / chunks
                DispatchQueue.concurrentPerform(iterations: chunks) { k in
                    var A = Net.Acts(k: 16)
                    var x = [Float](repeating: 0, count: 768)
                    for t in k * per ..< min((k + 1) * per, self.T) {
                        for j in 0 ..< 768 { x[j] = X[t * 768 + j] }
                        self.net.forward(x, &A)
                    }
                }
                times.append(-t1.timeIntervalSinceNow * 1000)
            }
            let med = times.sorted()[6]
            out(String(format: "%2d chunks: %6.1f ms/frame = %5.1f fps  (%.0fk tiles/s)",
                       chunks, med, 1000 / med, Double(T) / med))
        }
        // batched-GEMM CPU path: one big sgemm per layer, 256-tile chunks over all cores
        var symB = [UInt8](repeating: 0, count: T)
        let nch = (T + 255) / 256
        var tb = [Double]()
        for _ in 0 ..< 12 {
            let t1 = Date()
            symB.withUnsafeMutableBufferPointer { p in
                DispatchQueue.concurrentPerform(iterations: nch) { k in
                    self.net.forwardBatch(X, k * 256, min(256, self.T - k * 256), into: p)
                }
            }
            tb.append(-t1.timeIntervalSinceNow * 1000)
        }
        var med = tb.sorted()[6]
        var mm = zip(symB, refSym).filter { $0 != $1 }.count
        out(String(format: "cpu batched: %6.1f ms/frame = %5.1f fps  (vs ref: %d differ)",
                   med, 1000 / med, mm))
        // Core ML single-tensor model (TileNet2, fp16): silicon via computeUnits
        for (name, units) in [("cpuOnly ", MLComputeUnits.cpuOnly),
                              ("cpu+gpu ", .cpuAndGPU), ("all(ANE)", .all)] {
            do {
                let tL = Date()
                let eng = try CoreMLFieldEngine(units: units, T: T)
                var sym = try eng.classify(X, T)         // warmup: compile/load the units
                sym = try eng.classify(X, T)
                let loadS = -tL.timeIntervalSinceNow
                var ts = [Double]()
                for _ in 0 ..< 12 {
                    let t1 = Date()
                    sym = try eng.classify(X, T)
                    ts.append(-t1.timeIntervalSinceNow * 1000)
                }
                med = ts.sorted()[6]
                mm = zip(sym, refSym).filter { $0 != $1 }.count
                out(String(format: "coreml %@: %6.1f ms/frame = %5.1f fps  (vs ref: %d differ, load+warm %.1f s)",
                           name, med, 1000 / med, mm, loadS))
            } catch { out("coreml \(name): FAILED \(error.localizedDescription)") }
        }
        out("surface i5-7300U ref (cool, same 5244t): torch fp32 76.9 ms/13.0 fps")
        out("  OV fp32 53.6/18.7   OV int8 36.4/27.5 (its live engine)")
    }

    func warp(_ img: UIImage) -> [UInt8] {   // = cv2.warpPerspective(src, Hm, ...) : for
        let cg = img.cgImage!                // each field pixel, Hinv -> bilinear sample
        sw = cg.width; sh = cg.height
        srcPix = [UInt8](repeating: 0, count: sw * sh * 4)
        let ctx = CGContext(data: &srcPix, width: sw, height: sh, bitsPerComponent: 8,
                            bytesPerRow: sw * 4, space: CGColorSpaceCreateDeviceRGB(),
                            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        ctx.draw(cg, in: CGRect(x: 0, y: 0, width: sw, height: sh))
        return warpFull()
    }

    func warpFull() -> [UInt8] {
        let sw = self.sw, sh = self.sh, src = srcPix
        let ow = meta.cols * meta.C, oh = meta.rows * meta.C, H = meta.Hinv
        var out = [UInt8](repeating: 0, count: ow * oh * 3)
        out.withUnsafeMutableBufferPointer { o in
            src.withUnsafeBufferPointer { s in
                DispatchQueue.concurrentPerform(iterations: oh) { y in
                    for x in 0 ..< ow {
                        let d = Double(x), e = Double(y)
                        let q = H[6] * d + H[7] * e + H[8]
                        let sx = (H[0] * d + H[1] * e + H[2]) / q
                        let sy = (H[3] * d + H[4] * e + H[5]) / q
                        let x0 = Int(sx.rounded(.down)), y0 = Int(sy.rounded(.down))
                        if x0 < 0 || y0 < 0 || x0 + 1 >= sw || y0 + 1 >= sh { continue }
                        let fx = Float(sx) - Float(x0), fy = Float(sy) - Float(y0)
                        for ch in 0 ..< 3 {
                            let p00 = Float(s[(y0 * sw + x0) * 4 + ch])
                            let p01 = Float(s[(y0 * sw + x0 + 1) * 4 + ch])
                            let p10 = Float(s[((y0 + 1) * sw + x0) * 4 + ch])
                            let p11 = Float(s[((y0 + 1) * sw + x0 + 1) * 4 + ch])
                            let v = p00 * (1 - fx) * (1 - fy) + p01 * fx * (1 - fy)
                                  + p10 * (1 - fx) * fy + p11 * fx * fy
                            o[(y * ow + x) * 3 + ch] = UInt8(max(0, min(255, v.rounded())))
                        }
                    }
                }
            }
        }
        return out
    }

    func cutCrops(_ field: [UInt8]) {
        let c = meta.C, ow = meta.cols * c
        crops = [UInt8](repeating: 0, count: T * c * c * 3)
        for t in 0 ..< T {
            let r = t / meta.cols, cc = t % meta.cols
            for y in 0 ..< c { for x in 0 ..< c { for ch in 0 ..< 3 {
                crops[(t * c + y) * c * 3 + x * 3 + ch] = field[(((r * c + y) * ow) + cc * c + x) * 3 + ch]
            } } }
        }
    }

    // tile -> NN input (3,16,16), crop offset (ox,oy) into the 20x20, optional augment
    func input(_ t: Int, _ ox: Int, _ oy: Int, _ x: inout [Float], augment: Bool = false) {
        let c = meta.C
        for ch in 0 ..< 3 {
            let br: Float = augment ? .random(in: 0.7 ... 1.3) : 1
            for y in 0 ..< 16 { for xx in 0 ..< 16 {
                var v = Float(crops[(t * c + y + oy) * c * 3 + (xx + ox) * 3 + ch]) * br
                if augment { v += gauss() * 4 }
                x[(ch * 16 + y) * 16 + xx] = v / 255 - 0.5
            } }
        }
    }

    func classify() -> [UInt8] {             // all tiles, parallel over chunks
        var sym = [UInt8](repeating: 0, count: T)
        let chunks = 8, per = (T + chunks - 1) / chunks
        sym.withUnsafeMutableBufferPointer { p in
            DispatchQueue.concurrentPerform(iterations: chunks) { k in
                var A = Net.Acts(k: 16)
                var x = [Float](repeating: 0, count: 768)
                for t in k * per ..< min((k + 1) * per, self.T) {
                    self.input(t, 2, 2, &x)
                    self.net.forward(x, &A)
                    let s = A.s.indices.max { A.s[$0] < A.s[$1] }!
                    let c = A.c.indices.max { A.c[$0] < A.c[$1] }!
                    p[t] = UInt8(s << 2 | c)
                }
            }
        }
        return sym
    }

    func headerIdx(_ sym: [UInt8]) -> Int? { // 42 bits x 7 copies, majority, CRC-12
        let corners: Set<Int> = [0, meta.cols - 1, (meta.rows - 1) * meta.cols, T - 1]
        var hdr = [Int](); var i = 0
        while hdr.count < 49 { if !corners.contains(i) { hdr.append(i) }; i += 1 }
        var votes = [Int](repeating: 0, count: 42)
        for b in 0 ..< 294 { votes[b % 42] += Int(sym[hdr[b / 6]]) >> (5 - b % 6) & 1 }
        var idx = 0, crc = 0
        for j in 0 ..< 30 { idx = idx << 1 | (votes[j] > 3 ? 1 : 0) }
        for j in 30 ..< 42 { crc = crc << 1 | (votes[j] > 3 ? 1 : 0) }
        let be = (0 ..< 4).map { UInt8(idx >> (24 - 8 * $0) & 0xFF) }
        return crc == Int(crc32(be) & 0xFFF) ? idx : nil
    }

    func runDecode() {
        let t0 = Date()
        let sym = classify()
        let ms = -t0.timeIntervalSinceNow * 1000
        let refMiss = zip(sym, refSym).filter { $0 != $1 }.count
        let err = Double(zip(sym, truthSym).filter { $0 != $1 }.count) / Double(T)
        let idx = headerIdx(sym)
        out(String(format: "nn %.0f ms (%.0f tiles/s)", ms, Double(T) / ms * 1000))
        out("vs PC reference: \(T - refMiss)/\(T) match (\(refMiss) differ)")
        out(String(format: "tile err vs truth: %.3f%% (PC: %.3f%%)", err * 100, meta.pc_err * 100))
        out(idx == nil ? "header CRC FAIL" : "header idx \(idx!) CRC OK "
            + (idx == meta.idx ? "== PC" : "!= PC \(meta.idx)"))
    }

    func buildX() -> [Float] {               // classic prep: crops -> normalized tensors
        var X = [Float](repeating: 0, count: T * 768)
        var xt = [Float](repeating: 0, count: 768)
        for t in 0 ..< T {
            input(t, 2, 2, &xt)
            for j in 0 ..< 768 { X[t * 768 + j] = xt[j] }
        }
        return X
    }

    func med(_ n: Int, _ f: () -> Void) -> Double {
        var ts = [Double]()
        for _ in 0 ..< n { let t1 = Date(); f(); ts.append(-t1.timeIntervalSinceNow * 1000) }
        return ts.sorted()[n / 2]
    }

    // warp bench: camera frame -> NN-ready input, four ways. The rig is static, so the
    // gather approaches precompute source coords ONCE and pay only reads per frame.
    func runWarpBench() {
        out("\n-- warp bench: camera frame -> NN input, \(T) tiles --")
        let cols = meta.cols, rows = meta.rows
        var field = [UInt8]()
        let a = med(5, { field = self.warpFull() })
        let a2 = med(5, { self.cutCrops(field) })
        var X = [Float]()
        let a3 = med(5, { X = self.buildX() })
        out(String(format: "A naive: warp %.1f + crops %.1f + prep %.1f = %.1f ms", a, a2, a3, a + a2 + a3))

        var t0 = Date()
        let lut20 = GatherLUT(meta.Hinv, cols: cols, rows: rows, C: 20, side: 20, off: 0)
        let lut16 = GatherLUT(meta.Hinv, cols: cols, rows: rows, C: 20, side: 16, off: 2)
        out(String(format: "LUT build (once per session): %.0f ms both", -t0.timeIntervalSinceNow * 1000))
        var crops2 = [UInt8](repeating: 0, count: T * 1200)
        let bm = med(12, { gatherCrops(lut20, self.srcPix, self.sw, self.sh, self.T, into: &crops2) })
        var diff = zip(crops2, crops).filter { $0 != $1 }.count
        out(String(format: "B gather-LUT -> 20x20 crops: %.1f ms  (vs A: %d/%d bytes differ)",
                   bm, diff, crops2.count))
        var X2 = [Float](repeating: 0, count: T * 768)
        let cm = med(12, { gatherX(lut16, self.srcPix, self.sw, self.sh, self.T, into: &X2) })
        diff = zip(X2, X).filter { abs($0 - $1) > 1e-6 }.count
        out(String(format: "C gather-LUT -> NN tensor (fused, decode path): %.1f ms  (vs A: %d/%d floats differ)",
                   cm, diff, X2.count))

        guard let mw = MetalWarp(src: srcPix, sw: sw, sh: sh, Hinv: meta.Hinv,
                                 cols: cols, rows: rows) else { out("D metal: no device"); return }
        mw.run()                             // warmup: first dispatch pays pipeline setup
        let dm = med(12, { mw.run() })
        let gpuX = mw.outBuf.contents().assumingMemoryBound(to: Float.self)
        diff = 0
        for i in 0 ..< T * 768 where abs(gpuX[i] - X[i]) > 1e-6 { diff += 1 }
        out(String(format: "D Metal -> NN tensor (GPU): %.1f ms  (vs A: %d/%d floats differ)",
                   dm, diff, T * 768))

        do {                                 // E: whole NN side of a frame, no CPU pixels
            let eng = try CoreMLFieldEngine(units: .all, T: T)
            let arr = try MLMultiArray(dataPointer: mw.outBuf.contents(),
                                       shape: [NSNumber(value: T), 3, 16, 16],
                                       dataType: .float32,
                                       strides: [768, 256, 16, 1], deallocator: nil)
            var sym = try eng.classifyArr(arr, T)        // warmup ANE
            var ts = [Double]()
            for _ in 0 ..< 12 {
                let t1 = Date()
                mw.run()
                sym = try eng.classifyArr(arr, T)
                ts.append(-t1.timeIntervalSinceNow * 1000)
            }
            let mm = zip(sym, refSym).filter { $0 != $1 }.count
            out(String(format: "E Metal warp + ANE classify, full frame: %.1f ms  (vs ref: %d differ)",
                       ts.sorted()[6], mm))
        } catch { out("E: FAILED \(error.localizedDescription)") }
        t0 = Date()
        let symC = classify()
        out(String(format: "   (cpu16 classify on same frame for scale: %.1f ms, vs ref %d differ)",
                   -t0.timeIntervalSinceNow * 1000, zip(symC, refSym).filter { $0 != $1 }.count))
    }

    func evalErr() -> Double {               // joint err on the val tiles (i % 10 == 0)
        var wrong = 0, n = 0
        var A = Net.Acts(k: 16); var x = [Float](repeating: 0, count: 768)
        for t in stride(from: 0, to: T, by: 10) {
            input(t, 2, 2, &x)
            net.forward(x, &A)
            let s = A.s.indices.max { A.s[$0] < A.s[$1] }!
            let c = A.c.indices.max { A.c[$0] < A.c[$1] }!
            n += 1
            if UInt8(s << 2 | c) != truthSym[t] { wrong += 1 }
        }
        return Double(wrong) / Double(n)
    }

    func trainDemo() {
        bg {
            self.out("\n-- train demo: perturb weights, recover by training on this frame --")
            self.net.perturb(0.15)
            self.out(String(format: "after perturb: val err %.2f%%", self.evalErr() * 100))
            let trainIdx = (0 ..< self.T).filter { $0 % 10 != 0 }
            let opt = Adam(Net.counts(k: 16)), g = Net.Grads(Net.counts(k: 16))
            let bs = 256, steps = 200
            var A = Net.Acts(k: 16); var x = [Float](repeating: 0, count: 768)
            let t0 = Date()
            for step in 1 ... steps {
                let e = Float(self.evalOnce(step))    // LR tier follows current err
                opt.lr = e >= 0.03 ? 1e-4 : e >= 0.01 ? 5e-5 : e >= 0.005 ? 2e-5 : 1e-5
                g.zero()
                for _ in 0 ..< bs {
                    let t = trainIdx.randomElement()!
                    self.input(t, Int.random(in: 0 ... 4), Int.random(in: 0 ... 4), &x, augment: true)
                    self.net.forward(x, &A)
                    self.net.backward(&A, Int(self.truthSym[t]) >> 2, Int(self.truthSym[t]) & 3, g)
                }
                opt.step(self.net, g, batch: bs)
                if step % 20 == 0 {
                    self.out(String(format: "step %3d  val err %.2f%%  (%.1f s)",
                                    step, self.evalErr() * 100, -t0.timeIntervalSinceNow))
                }
            }
            self.out(String(format: "done: val err %.2f%% in %.1f s (%d steps x %d)",
                            self.evalErr() * 100, -t0.timeIntervalSinceNow, steps, bs))
        }
    }

    var lastErr = 1.0
    func evalOnce(_ step: Int) -> Double {   // cheap: refresh the LR-tier err every 20 steps
        if step % 20 == 1 { lastErr = evalErr() }
        return lastErr
    }

    func reset() {
        bg {
            self.net = Net(self.asset("weights", "bin"))
            self.out("\nweights reset from bundle")
            self.runDecode()
        }
    }
}

let crcTable: [UInt32] = (0 ..< 256).map { i in
    var c = UInt32(i)
    for _ in 0 ..< 8 { c = c & 1 != 0 ? 0xEDB88320 ^ (c >> 1) : c >> 1 }
    return c
}

func crc32(_ bytes: [UInt8]) -> UInt32 {     // = zlib.crc32
    var c: UInt32 = 0xFFFFFFFF
    for b in bytes { c = crcTable[Int((c ^ UInt32(b)) & 0xFF)] ^ (c >> 8) }
    return c ^ 0xFFFFFFFF
}
