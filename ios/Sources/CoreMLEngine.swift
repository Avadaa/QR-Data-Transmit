// CoreMLEngine.swift — the same net through Core ML (TileNet.mlmodel built on the mac
// straight from weights.bin). computeUnits picks the silicon: .cpuOnly / .cpuAndGPU /
// .all (Neural Engine). Batch = one MLArrayBatchProvider for the whole field, built
// once and reused; outputs read via dataPointer (NSNumber subscripts are slow).
import CoreML
import Foundation

final class CoreMLEngine {
    let model: MLModel

    init(units: MLComputeUnits) throws {
        let cfg = MLModelConfiguration()
        cfg.computeUnits = units
        if let compiled = Bundle.main.url(forResource: "TileNet", withExtension: "mlmodelc") {
            model = try MLModel(contentsOf: compiled, configuration: cfg)
        } else {                     // raw .mlmodel in the bundle -> compile once on device
            let src = Bundle.main.url(forResource: "TileNet", withExtension: "mlmodel")!
            let dst = FileManager.default.temporaryDirectory.appendingPathComponent("TileNet.mlmodelc")
            if !FileManager.default.fileExists(atPath: dst.path) {
                let c = try MLModel.compileModel(at: src)
                try? FileManager.default.removeItem(at: dst)
                try FileManager.default.moveItem(at: c, to: dst)
            }
            model = try MLModel(contentsOf: dst, configuration: cfg)
        }
    }

    func makeBatch(_ X: [Float], _ T: Int) throws -> MLArrayBatchProvider {
        var provs = [MLFeatureProvider]()
        provs.reserveCapacity(T)
        for t in 0 ..< T {
            let a = try MLMultiArray(shape: [3, 16, 16], dataType: .float32)
            X.withUnsafeBufferPointer { p in
                memcpy(a.dataPointer, p.baseAddress! + t * 768, 768 * 4)
            }
            provs.append(try MLDictionaryFeatureProvider(dictionary: ["x": a]))
        }
        return MLArrayBatchProvider(array: provs)
    }

    private func amax(_ a: MLMultiArray, _ n: Int) -> Int {
        var bi = 0
        if a.dataType == .double {
            let p = a.dataPointer.assumingMemoryBound(to: Double.self)
            var bv = p[0]
            for j in 1 ..< n where p[j] > bv { bv = p[j]; bi = j }
        } else {
            let p = a.dataPointer.assumingMemoryBound(to: Float.self)
            var bv = p[0]
            for j in 1 ..< n where p[j] > bv { bv = p[j]; bi = j }
        }
        return bi
    }

    func classify(_ batch: MLArrayBatchProvider) throws -> [UInt8] {
        let out = try model.predictions(fromBatch: batch)
        var sym = [UInt8](repeating: 0, count: out.count)
        for i in 0 ..< out.count {
            let f = out.features(at: i)
            let s = amax(f.featureValue(for: "shape_logits")!.multiArrayValue!, 16)
            let c = amax(f.featureValue(for: "color_logits")!.multiArrayValue!, 4)
            sym[i] = UInt8(s << 2 | c)
        }
        return sym
    }
}

// TileNet2: whole field in ONE (T,3,16,16) tensor, one predict call — kills the
// per-item batch dispatch that made the v1 path 199-291 ms. fp16 mlprogram (ANE-native).
final class CoreMLFieldEngine {
    let model: MLModel
    let inArr: MLMultiArray

    init(units: MLComputeUnits, T: Int) throws {
        let cfg = MLModelConfiguration()
        cfg.computeUnits = units
        let url = Bundle.main.url(forResource: "TileNet2", withExtension: "mlmodelc")!
        model = try MLModel(contentsOf: url, configuration: cfg)
        inArr = try MLMultiArray(shape: [NSNumber(value: T), 3, 16, 16], dataType: .float32)
    }

    private func argmaxRows(_ a: MLMultiArray, _ T: Int, _ n: Int, into best: inout [Int]) {
        if a.dataType == .float16 {
            let p = a.dataPointer.assumingMemoryBound(to: Float16.self)
            for t in 0 ..< T {
                var bi = 0, bv = p[t * n]
                for j in 1 ..< n where p[t * n + j] > bv { bv = p[t * n + j]; bi = j }
                best[t] = bi
            }
        } else {
            let p = a.dataPointer.assumingMemoryBound(to: Float.self)
            for t in 0 ..< T {
                var bi = 0, bv = p[t * n]
                for j in 1 ..< n where p[t * n + j] > bv { bv = p[t * n + j]; bi = j }
                best[t] = bi
            }
        }
    }

    func classify(_ X: [Float], _ T: Int) throws -> [UInt8] {
        X.withUnsafeBufferPointer { p in
            memcpy(inArr.dataPointer, p.baseAddress!, T * 768 * 4)
        }
        return try classifyArr(inArr, T)
    }

    func classifyArr(_ arr: MLMultiArray, _ T: Int) throws -> [UInt8] {
        let out = try model.prediction(from: MLDictionaryFeatureProvider(dictionary: ["x": arr]))
        var si = [Int](repeating: 0, count: T), ci = [Int](repeating: 0, count: T)
        argmaxRows(out.featureValue(for: "shape_logits")!.multiArrayValue!, T, 16, into: &si)
        argmaxRows(out.featureValue(for: "color_logits")!.multiArrayValue!, T, 4, into: &ci)
        return (0 ..< T).map { UInt8(si[$0] << 2 | ci[$0]) }
    }
}

// ANEField: the LIVE receiver's ANE engine. The bundled TileNet2.mlmodelc has the
// FACTORY weights baked in, but on-device training tunes the CPU net away from them
// and Core ML has no weight-swap API. So: copy the compiled model to Documents, find
// each tensor's offset in weights/weight.bin ONCE by searching for its factory fp16
// byte pattern (blob stores tensors in op order -> ordered search disambiguates),
// then rebuild() overwrites them with the current net's weights and reloads (~0.3 s).
// The caller cross-checks the first frame against the CPU net before trusting it.
final class ANEField {
    let T: Int
    let dir: URL
    var offsets = [Int]()                 // per Net.keys tensor, offset into weight.bin
    var milPatches = [Int]()              // tensors coremltools INLINED into model.mil
    var milOrig = ""                      // (consts < 10 elements — measured: bc only)
    var model: MLModel?

    init?(T: Int, factory: Data, resource: String, templateT: Int) {
        self.T = T                          // template: TileNet2 (5244) / TileNet12 (9348)
        dir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent(T == templateT ? "ane_\(resource).mlmodelc"
                                                   : "ane_\(resource)_T\(T).mlmodelc")
        guard let src = Bundle.main.url(forResource: resource, withExtension: "mlmodelc")
        else { return nil }
        try? FileManager.default.removeItem(at: dir)
        guard (try? FileManager.default.copyItem(at: src, to: dir)) != nil else { return nil }
        if T != templateT {                 // ON-DEVICE MINT: T is only the BATCH dim —
            guard ANEField.reshape(dir, from: templateT, to: T) else { return nil }
        }                                   // weights are per-tile, untouched
        guard let blob = try? Data(contentsOf: dir.appendingPathComponent("weights/weight.bin"))
        else { return nil }
        var foff = 0, off = 0
        for (i, n) in Net.counts(k: 16).enumerated() {   // ANE model is the r16 net
            let pat = ANEField.fp16(factory.subdata(in: foff ..< foff + n * 4))
            foff += n * 4
            if let r = blob.range(of: pat, in: off ..< blob.count) {
                offsets.append(r.lowerBound)
                off = r.lowerBound + pat.count
            } else if n < 10 {            // tiny const -> lives in model.mil as a literal
                offsets.append(-1); milPatches.append(i)
            } else { return nil }         // a real weight tensor missing = layout mismatch
        }
        if !milPatches.isEmpty {
            guard let s = try? String(contentsOf: dir.appendingPathComponent("model.mil"),
                                      encoding: .utf8) else { return nil }
            milOrig = s
        }
    }

    // Reshape a compiled mlmodelc to a new tile count. Three places carry the shape:
    // model.mil (text — every tensor annotation), metadata.json (text), and
    // coremldata.bin (the model DESCRIPTION predict validates against — the T lives
    // there as protobuf varints; equal-length varint swap keeps the framing intact).
    // Proven on mac CoreML: minted model's per-tile logits are IDENTICAL to the
    // template's. The first-frame ANE-vs-CPU cross-check remains the on-device guard.
    static func reshape(_ dir: URL, from tmpl: Int, to T: Int) -> Bool {
        func vint(_ v0: Int) -> Data {
            var v = v0, d = Data()
            while v >= 0x80 { d.append(UInt8(v & 0x7F | 0x80)); v >>= 7 }
            d.append(UInt8(v))
            return d
        }
        let old = vint(tmpl), new = vint(T)
        guard old.count == new.count,       // framing must not shift (T 128..16383 ok)
              let re = try? NSRegularExpression(pattern: "(?<![0-9])\(tmpl)(?![0-9])")
        else { return false }
        for name in ["model.mil", "metadata.json"] {
            let url = dir.appendingPathComponent(name)
            guard let s = try? String(contentsOf: url, encoding: .utf8) else { return false }
            let out = re.stringByReplacingMatches(in: s, range: NSRange(s.startIndex..., in: s),
                                                  withTemplate: "\(T)")
            guard (try? out.write(to: url, atomically: true, encoding: .utf8)) != nil
            else { return false }
        }
        let cdURL = dir.appendingPathComponent("coremldata.bin")
        guard var cd = try? Data(contentsOf: cdURL) else { return false }
        var sites = 0, pos = cd.startIndex
        while let r = cd.range(of: old, in: pos ..< cd.endIndex) {
            cd.replaceSubrange(r, with: new)
            sites += 1
            pos = r.lowerBound + new.count
        }
        guard sites > 0, sites <= 8,        // expected 3 (input + 2 outputs); wildly
              (try? cd.write(to: cdURL)) != nil else { return false }   // off = bail
        return true
    }

    static func fp16(_ fp32: Data) -> Data {
        fp32.withUnsafeBytes { raw in
            let f = raw.bindMemory(to: Float.self)
            var d = Data(capacity: f.count * 2)
            for v in f { withUnsafeBytes(of: Float16(v)) { d.append(contentsOf: $0) } }
            return d
        }
    }

    func rebuild(_ net: Net) -> Bool {    // current weights -> blob + mil -> reload on ANE
        let url = dir.appendingPathComponent("weights/weight.bin")
        guard var blob = try? Data(contentsOf: url) else { return false }
        for (i, kp) in Net.keys.enumerated() where offsets[i] >= 0 {
            let p = net[keyPath: kp]
            let d = p.withUnsafeBufferPointer { ANEField.fp16(Data(buffer: $0)) }
            blob.replaceSubrange(offsets[i] ..< offsets[i] + d.count, with: d)
        }
        guard (try? blob.write(to: url)) != nil else { return false }
        if !milPatches.isEmpty {          // rewrite inline literals, always from pristine
            var s = milOrig
            for i in milPatches {
                let prefix = "val = tensor<fp16, [\(Net.counts(k: 16)[i])]>(["
                guard let a = s.range(of: prefix),
                      let b = s.range(of: "])", range: a.upperBound ..< s.endIndex)
                else { return false }
                let vals = net[keyPath: Net.keys[i]]
                    .map { String(format: "%a", Double(Float16($0))) }.joined(separator: ", ")
                s.replaceSubrange(a.upperBound ..< b.lowerBound, with: vals)
            }
            guard (try? s.write(to: dir.appendingPathComponent("model.mil"),
                                atomically: true, encoding: .utf8)) != nil else { return false }
        }
        let cfg = MLModelConfiguration()
        cfg.computeUnits = .cpuAndNeuralEngine   // never the GPU — Metal warp + UI own it;
        model = try? MLModel(contentsOf: dir, configuration: cfg)   // live contention there
        return model != nil                                         // is what .all risks
    }

    // chunked predict: arrs are views into ONE shared warp buffer at T-tile strides —
    // grids past the 2-byte-varint mint cap (16383) run ceil(total/T) predicts; the
    // padded tail of the last chunk is garbage and simply not read out
    func classify(_ arrs: [MLMultiArray], _ total: Int) -> [UInt8]? {
        guard let m = model else { return nil }
        var sym = [UInt8](repeating: 0, count: total)
        for (ci, arr) in arrs.enumerated() {
            guard let inp = try? MLDictionaryFeatureProvider(dictionary: ["x": arr]),
                  let out = try? m.prediction(from: inp) else { return nil }
            let base = ci * T, n = min(T, total - base)
            amax(out.featureValue(for: "shape_logits")!.multiArrayValue!, 16,
                 into: &sym, shift: 2, base: base, count: n)
            amax(out.featureValue(for: "color_logits")!.multiArrayValue!, 4,
                 into: &sym, shift: 0, base: base, count: n)
        }
        return sym
    }

    private func amax(_ a: MLMultiArray, _ n: Int, into sym: inout [UInt8], shift: Int,
                      base: Int, count: Int) {
        if a.dataType == .float16 {
            let p = a.dataPointer.assumingMemoryBound(to: Float16.self)
            for t in 0 ..< count {
                var bi = 0, bv = p[t * n]
                for j in 1 ..< n where p[t * n + j] > bv { bv = p[t * n + j]; bi = j }
                sym[base + t] |= UInt8(bi << shift)
            }
        } else {
            let p = a.dataPointer.assumingMemoryBound(to: Float.self)
            for t in 0 ..< count {
                var bi = 0, bv = p[t * n]
                for j in 1 ..< n where p[t * n + j] > bv { bv = p[t * n + j]; bi = j }
                sym[base + t] |= UInt8(bi << shift)
            }
        }
    }
}
