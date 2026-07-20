// Warp.swift — the warp bottleneck, attacked two ways. The rig is static (cached Hm),
// so the per-pixel projective math never changes: precompute WHERE each output pixel
// comes from, once, and every frame is pure gathering (same trick as the receiver's FA
// remap LUT / notes2 "warp->gather"). CPU version + Metal version (GPU texture reads,
// kernel writes the NN's float tensor directly — camera frame to ANE with no CPU pixels).
import Foundation
import Metal

struct GatherLUT {          // per output pixel: source coords in the camera frame
    var sx: [Float], sy: [Float]
    let side: Int           // crop side gathered (20 = training crops, 16 = NN center)

    init(_ Hinv: [Double], cols: Int, rows: Int, C: Int, side: Int, off: Int) {
        self.side = side
        let n = rows * cols * side * side
        sx = [Float](repeating: 0, count: n); sy = [Float](repeating: 0, count: n)
        var i = 0
        for r in 0 ..< rows { for c in 0 ..< cols {
            for y in 0 ..< side { for x in 0 ..< side {
                let dx = Double(c * C + x + off), dy = Double(r * C + y + off)
                let q = Hinv[6] * dx + Hinv[7] * dy + Hinv[8]
                sx[i] = Float((Hinv[0] * dx + Hinv[1] * dy + Hinv[2]) / q)
                sy[i] = Float((Hinv[3] * dx + Hinv[4] * dy + Hinv[5]) / q)
                i += 1
            } }
        } }
    }
}

// gather 20x20 uint8 crops (training path) — layout identical to Runner.cutCrops
func gatherCrops(_ lut: GatherLUT, _ src: [UInt8], _ sw: Int, _ sh: Int, _ T: Int,
                 into crops: inout [UInt8]) {
    crops.withUnsafeMutableBufferPointer { cp in
        src.withUnsafeBufferPointer { sp in
            DispatchQueue.concurrentPerform(iterations: 16) { k in
                let per = (T + 15) / 16
                for t in k * per ..< min((k + 1) * per, T) {
                    for pix in 0 ..< 400 {
                        let i = t * 400 + pix
                        let x0 = Int(lut.sx[i].rounded(.down)), y0 = Int(lut.sy[i].rounded(.down))
                        let o = i * 3
                        if x0 < 0 || y0 < 0 || x0 + 1 >= sw || y0 + 1 >= sh {
                            cp[o] = 0; cp[o + 1] = 0; cp[o + 2] = 0; continue
                        }
                        let fx = lut.sx[i] - Float(x0), fy = lut.sy[i] - Float(y0)
                        let b = (y0 * sw + x0) * 4, b2 = b + sw * 4
                        for ch in 0 ..< 3 {
                            let v = Float(sp[b + ch]) * (1 - fx) * (1 - fy)
                                  + Float(sp[b + 4 + ch]) * fx * (1 - fy)
                                  + Float(sp[b2 + ch]) * (1 - fx) * fy
                                  + Float(sp[b2 + 4 + ch]) * fx * fy
                            cp[o + ch] = UInt8(max(0, min(255, v.rounded())))
                        }
                    }
                }
            }
        }
    }
}

// gather straight into the NN input tensor (decode path): center 16x16 only, normalized
// floats in (T,3,16,16) layout — the separate crop + prep passes disappear entirely
func gatherX(_ lut: GatherLUT, _ src: [UInt8], _ sw: Int, _ sh: Int, _ T: Int,
             into X: inout [Float]) {
    X.withUnsafeMutableBufferPointer { xp in
        src.withUnsafeBufferPointer { sp in
            DispatchQueue.concurrentPerform(iterations: 16) { k in
                let per = (T + 15) / 16
                for t in k * per ..< min((k + 1) * per, T) {
                    for pix in 0 ..< 256 {
                        let i = t * 256 + pix
                        let o = t * 768 + pix
                        let x0 = Int(lut.sx[i].rounded(.down)), y0 = Int(lut.sy[i].rounded(.down))
                        if x0 < 0 || y0 < 0 || x0 + 1 >= sw || y0 + 1 >= sh {
                            xp[o] = -0.5; xp[o + 256] = -0.5; xp[o + 512] = -0.5; continue
                        }
                        let fx = lut.sx[i] - Float(x0), fy = lut.sy[i] - Float(y0)
                        let b = (y0 * sw + x0) * 4, b2 = b + sw * 4
                        for ch in 0 ..< 3 {
                            let v = Float(sp[b + ch]) * (1 - fx) * (1 - fy)
                                  + Float(sp[b + 4 + ch]) * fx * (1 - fy)
                                  + Float(sp[b2 + ch]) * (1 - fx) * fy
                                  + Float(sp[b2 + 4 + ch]) * fx * fy
                            xp[o + ch * 256] = max(0, min(255, v.rounded())) / 255 - 0.5
                        }
                    }
                }
            }
        }
    }
}

// Metal: one GPU thread per NN-input pixel — projective math + bilinear reads from the
// camera texture, writes the normalized float tensor into a SHARED buffer. That buffer
// is wrapped as the Core ML input, so warp+prep never touch the CPU.
final class MetalWarp {
    let pipe: MTLComputePipelineState
    let queue: MTLCommandQueue
    let tex: MTLTexture
    let outBuf, hBuf, dimBuf: MTLBuffer
    let count: Int

    static let msl = """
    #include <metal_stdlib>
    using namespace metal;
    kernel void warpX(texture2d<float, access::read> src [[texture(0)]],
                      device float *outX [[buffer(0)]],
                      constant float *H [[buffer(1)]],
                      constant int *dims [[buffer(2)]],
                      uint gid [[thread_position_in_grid]]) {
        int cols = dims[0], sw = dims[1], sh = dims[2], total = dims[3];
        if (int(gid) >= total) return;
        int t = gid / 256, pix = gid % 256;
        int y = pix / 16, x = pix % 16;
        int r = t / cols, c = t % cols;
        float dx = float(c * 20 + x + 2), dy = float(r * 20 + y + 2);
        float q = H[6] * dx + H[7] * dy + H[8];
        float sx = (H[0] * dx + H[1] * dy + H[2]) / q;
        float sy = (H[3] * dx + H[4] * dy + H[5]) / q;
        int x0 = int(floor(sx)), y0 = int(floor(sy));
        int o = t * 768 + y * 16 + x;
        if (x0 < 0 || y0 < 0 || x0 + 1 >= sw || y0 + 1 >= sh) {
            outX[o] = -0.5f; outX[o + 256] = -0.5f; outX[o + 512] = -0.5f; return;
        }
        float fx = sx - float(x0), fy = sy - float(y0);
        float4 p00 = src.read(uint2(x0, y0)),     p01 = src.read(uint2(x0 + 1, y0));
        float4 p10 = src.read(uint2(x0, y0 + 1)), p11 = src.read(uint2(x0 + 1, y0 + 1));
        float4 v = p00 * (1 - fx) * (1 - fy) + p01 * fx * (1 - fy)
                 + p10 * (1 - fx) * fy + p11 * fx * fy;
        outX[o]       = rint(clamp(v.r, 0.0f, 1.0f) * 255.0f) / 255.0f - 0.5f;
        outX[o + 256] = rint(clamp(v.g, 0.0f, 1.0f) * 255.0f) / 255.0f - 0.5f;
        outX[o + 512] = rint(clamp(v.b, 0.0f, 1.0f) * 255.0f) / 255.0f - 0.5f;
    }
    // fp16 twin: writes half straight into the Core ML input buffer (model input is
    // fp16) — the fp32 math is identical, only the final store narrows (RNE, same as
    // Core ML's own input conversion used to do on the CPU)
    kernel void warpXh(texture2d<float, access::read> src [[texture(0)]],
                       device half *outX [[buffer(0)]],
                       constant float *H [[buffer(1)]],
                       constant int *dims [[buffer(2)]],
                       uint gid [[thread_position_in_grid]]) {
        int cols = dims[0], sw = dims[1], sh = dims[2], total = dims[3];
        if (int(gid) >= total) return;
        int t = gid / 256, pix = gid % 256;
        int y = pix / 16, x = pix % 16;
        int r = t / cols, c = t % cols;
        float dx = float(c * 20 + x + 2), dy = float(r * 20 + y + 2);
        float q = H[6] * dx + H[7] * dy + H[8];
        float sx = (H[0] * dx + H[1] * dy + H[2]) / q;
        float sy = (H[3] * dx + H[4] * dy + H[5]) / q;
        int x0 = int(floor(sx)), y0 = int(floor(sy));
        int o = t * 768 + y * 16 + x;
        if (x0 < 0 || y0 < 0 || x0 + 1 >= sw || y0 + 1 >= sh) {
            outX[o] = -0.5h; outX[o + 256] = -0.5h; outX[o + 512] = -0.5h; return;
        }
        float fx = sx - float(x0), fy = sy - float(y0);
        float4 p00 = src.read(uint2(x0, y0)),     p01 = src.read(uint2(x0 + 1, y0));
        float4 p10 = src.read(uint2(x0, y0 + 1)), p11 = src.read(uint2(x0 + 1, y0 + 1));
        float4 v = p00 * (1 - fx) * (1 - fy) + p01 * fx * (1 - fy)
                 + p10 * (1 - fx) * fy + p11 * fx * fy;
        outX[o]       = half(rint(clamp(v.r, 0.0f, 1.0f) * 255.0f) / 255.0f - 0.5f);
        outX[o + 256] = half(rint(clamp(v.g, 0.0f, 1.0f) * 255.0f) / 255.0f - 0.5f);
        outX[o + 512] = half(rint(clamp(v.b, 0.0f, 1.0f) * 255.0f) / 255.0f - 0.5f);
    }
    """

    let sw, sh: Int

    // bgra: live camera frames are BGRA -> .bgra8Unorm so .r/.g/.b read as true RGB;
    // the bench feeds RGBA and keeps the old format. outTiles: allocate the output for
    // MORE tiles than the grid (chunked-ANE padding — the tail is never written)
    init?(sw: Int, sh: Int, cols: Int, rows: Int, bgra: Bool = false, half: Bool = false,
          outTiles: Int? = nil) {
        guard let dev = MTLCreateSystemDefaultDevice(),
              let lib = try? dev.makeLibrary(source: MetalWarp.msl, options: nil),
              let fn = lib.makeFunction(name: half ? "warpXh" : "warpX"),
              let p = try? dev.makeComputePipelineState(function: fn) else { return nil }
        let td = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: bgra ? .bgra8Unorm : .rgba8Unorm,
                                                          width: sw, height: sh, mipmapped: false)
        td.usage = .shaderRead
        let n = rows * cols * 256
        var dd: [Int32] = [Int32(cols), Int32(sw), Int32(sh), Int32(n)]
        self.sw = sw; self.sh = sh
        pipe = p
        queue = dev.makeCommandQueue()!
        tex = dev.makeTexture(descriptor: td)!
        count = n
        outBuf = dev.makeBuffer(length: (outTiles ?? rows * cols) * 768 * (half ? 2 : 4),
                                options: .storageModeShared)!
        hBuf = dev.makeBuffer(length: 9 * 4)!
        dimBuf = dev.makeBuffer(bytes: &dd, length: 16)!
    }

    convenience init?(src: [UInt8], sw: Int, sh: Int, Hinv: [Double], cols: Int, rows: Int) {
        self.init(sw: sw, sh: sh, cols: cols, rows: rows)
        upload(src)
        setHinv(Hinv)
    }

    func upload(_ src: [UInt8]) {
        src.withUnsafeBufferPointer {
            tex.replace(region: MTLRegionMake2D(0, 0, sw, sh), mipmapLevel: 0,
                        withBytes: $0.baseAddress!, bytesPerRow: sw * 4)
        }
    }

    func setHinv(_ Hinv: [Double]) {
        let p = hBuf.contents().assumingMemoryBound(to: Float.self)
        for i in 0 ..< 9 { p[i] = Float(Hinv[i]) }
    }

    func run() {
        let cb = queue.makeCommandBuffer()!
        let e = cb.makeComputeCommandEncoder()!
        e.setComputePipelineState(pipe)
        e.setTexture(tex, index: 0)
        e.setBuffer(outBuf, offset: 0, index: 0)
        e.setBuffer(hBuf, offset: 0, index: 1)
        e.setBuffer(dimBuf, offset: 0, index: 2)
        e.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                          threadsPerThreadgroup: MTLSize(width: pipe.threadExecutionWidth,
                                                         height: 1, depth: 1))
        e.endEncoding()
        cb.commit(); cb.waitUntilCompleted()
    }
}
