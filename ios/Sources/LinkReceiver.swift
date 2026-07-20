// LinkReceiver.swift — the receiver, ported. Camera frames -> dedup (per-band
// stability, newest-stable commit, 1.5s held) -> control QRs steer the phase machine
// (TRAIN -> train on PCG64-derived truth with live err; START -> armed; first moving
// commit -> data; END -> fountain repair + reassembly + md5). Geometry = Geometry.swift
// candidates, header CRC arbitrates and caches Hinv (static rig, RESEARCH_FAILS=0).
// Decode runs the CPU net so freshly tuned weights are live immediately.
// HANDHELD v2 replaces the dwell machinery entirely (see the HH pipeline section):
// per-frame header naming + best-shot decode + tracked geometry, on hh_* model files.
import CoreML
import CoreVideo
import CryptoKit
import Foundation
import UIKit
import Vision

final class LinkReceiver: ObservableObject {
    static let shared = LinkReceiver()   // menu buttons + receive view share the model
    @Published var log = ""
    @Published var stat = "idle"
    @Published var running = false
    @Published var mode = "WAITING"
    @Published var previewCG: CGImage?
    @Published var zoomPreview = false      // Z: 1:1 center crop — the NN's-eye view
    var lastPrev = Date(timeIntervalSince1970: 0)
    var calDone = false, calibrating = false   // sweep blurs the lens on purpose — the
                                               // phase machine must NOT interpret those
                                               // frames (blur ate the QR once and
                                               // "training" started on the QR screen)
    var lastFrameBuf = [UInt8]()            // newest camera frame (calibration grabs it)
    // debug mode: verbose geometry, per-frame ID probing, frame dumps, lapvar
    var debug = false
    var camFrames = 0, probeNil = 0
    var probeIds = [Int: Int]()
    var allIds = Set<Int>()
    var lastDbg = Date(), lastDump = Date(timeIntervalSince1970: 0)
    var dumpN = 0
    var sharp = 0.0
    var startT = Date()

    static let TRAIN_SEED = 7000, REPAIR_SEED = 9000, SCRAMBLE_SEED = 5000
    static let TH_CHANGE = 12.0, TH_STABLE = 6.0, TH_BLEND = 18.0
    static let TRAIN_EVERY = 20, TRAIN_STEPS = 120, TRAIN_BS = 256, QCAP = 20

    struct Grid {
        var px = 16, cols = 0, rows = 0, nsym = 32
        var T = 0, C = 20, K = 16
        var scaled = false              // canonical-16: 12px tiles sampled at 16x16
        var v3 = false, vnc = 8         // v3.1 DMT payload: PAM-2, vnc coeffs x 3 ch
        var corners = [Int](), hdr = [Int](), pay = [Int]()
        var nblk = 0, kpf = 0
        init(px: Int, cols: Int, rows: Int, nsym: Int, scaled: Bool = false,
             v3: Bool = false, vnc: Int = 8) {
            self.px = px; self.cols = cols; self.rows = rows; self.nsym = nsym
            self.scaled = scaled; self.v3 = v3; self.vnc = vnc
            if scaled { K = 16; C = 20 }   // r12 content through the r16-shaped net —
            else { K = px; C = px + 4 }    // 78% more samples per glyph at 4K
            T = cols * rows
            corners = [0, cols - 1, (rows - 1) * cols, T - 1]
            let free = (0 ..< T).filter { !corners.contains($0) }
            hdr = Array(free[0 ..< 49]); pay = Array(free[49...])
            nblk = v3 ? (pay.count * 3 * vnc / 8) / 255   // 1 bit/coeff/ch (PAM-2)
                      : (pay.count * 6 / 8) / 255
            kpf = nblk * (255 - nsym)
        }
    }

    let camera = Camera()
    var net: Net!
    var grid: Grid?
    var plan = [String: Double]()
    var Hinv: [Double]?
    var phase = "wait"
    var D = [Int: [UInt8]]()
    var failed = 0, commits = 0, dropped = 0
    var pool = [Int: [Int]]()
    var t0: Date?
    var locked = false, finished = false
    var handheld = false      // handheld v2: per-frame naming owns the session
    var relocks = 0           // shared-path geometry re-locks (train fallback etc)
    var fileName: String?     // NAME= from the arm QR: a complete transfer saves the
                              // payload as this file under Documents/QR Transmit/
    var v3Wire = false        // V=3 announced with the menu wire on v2: footage mode
                              // (glyph training/CAL must NOT run — label poison)
    var wireV3 = false        // menu "v3.1 DMT": decode v3 payloads with the demapper
    var demap: Demap?         // v3.1 payload net; the GLYPH net keeps header/corners
    var curDemapKey = 0, curDemapNC = 0
    var lastQRScan = Date(timeIntervalSince1970: 0)   // handheld: a hand never holds
                              // still for the 1.5 s held-commit rule, so control QRs
                              // are scanned on a raw frame every 0.7 s instead

    // handheld v2 (HH pipeline). Dwell stability never happens in a hand, so frame
    // IDENTITY comes from a per-frame 49-header-tile probe instead of dedup, and the
    // best-scoring shot per idx gets one full decode. ALL decision state below is
    // camera-thread-only; workQ answers through the mailboxes guarded by `lock`.
    var hhHinv: [Double]?                 // tracked geometry (camera thread's copy)
    var hhNewHinv: [Double]?              // mailbox: relock/shared-path lock -> camera
    var hhResults = [(idx: Int, ok: Bool)]()   // mailbox: decode outcomes
    var hhRelockOutcome: Bool?            // mailbox: full-search finished (ok?)
    var hhPend: (idx: Int, score: Double, f: [UInt8], hv: [Double], t: Date)?
    var hhSeen = Set<Int>()               // decoded idxs — never flush twice
    var hhInFlight = false                // one HH decode at a time on the workQ
    var hhMiss = 0                        // consecutive unnamed frames
    var hhShiftPos = 0                    // round-robin cursor over the shift ring
    var hhRelockBusy = false, hhRelockFails = 0
    var hhLastRelock = Date(timeIntervalSince1970: 0)
    var hhLastAF = Date(timeIntervalSince1970: 0)
    var hhLastHarvest = Date(timeIntervalSince1970: 0)
    var hhLastLog = Date(timeIntervalSince1970: 0)
    var hhFramesN = 0, hhNamed = 0, hhSwaps = 0
    var relockShift = 0, relockFull = 0, afKicks = 0, hhRetries = 0
    var hhBlockMsg = false
    static let hhShiftRing: [(Double, Double)] = [(8, 0), (-8, 0), (0, 8), (0, -8),
                                                  (16, 0), (-16, 0), (0, 16), (0, -16)]
    // native r12 has no ANE model — its 343 ms CPU decode can't hold the dwell rate,
    // so HH falls back to the v1.2 re-lock path there instead of going dark
    var hhActive: Bool { handheld && !debug && (grid?.K ?? 16) == 16
                         && grid?.v3 != true }   // v3.1 + handheld: not yet

    // dedup state (camera queue only). pend is a LAZY byte provider: the copy path
    // captures the already-copied frame; the zero-copy path captures the retained
    // CVPixelBuffer and copies only if this dwell actually commits.
    var prevS, pendS, keptS: [Int16]?
    var pendGet: (() -> [UInt8])?
    var pendQ = Double.infinity, pendT = Date(), poolN = 0
    var fw = 0, fh = 0, lastStat = Date()

    let workQ = DispatchQueue(label: "rx.worker", qos: .userInteractive)
    // decode is the critical path: at .userInitiated the predict call's CPU phases were
    // losing P-core time to the camera queue's 60fps frame copies (nn 44ms vs 8.3 bench)
    let lock = NSLock()
    var backlog = 0
    // spill tier (Surface doctrine: capture never blocks on decode, the backlog drains
    // after END): when the raw backlog is full, frames JPEG-compress (~2-3 MB vs 8-33)
    // instead of dropping, and decode late in assemble(). Bounded by bytes, not count.
    let spillQ = DispatchQueue(label: "rx.spill", qos: .utility)
    var spillJpgs = [(Data, Int)](), spillBytes = 0, spilled = 0   // lock-guarded

    // training reservoirs (worker queue only)
    var RTX = [UInt8](), RTy = [UInt8](), RVX = [UInt8](), RVy = [UInt8]()
    var rtn = 0, rvn = 0, trn = 0
    var terr: Double?
    // chunk trigger: every TRAIN_EVERY frames, OR >=4 frames waiting 20+ s — a cold
    // model (r12 factory!) harvests slowly and waiting for 20 full frames starves the
    // very training that would speed the harvest up
    var gatherT0: Date?, lastChunkT: Date?, sinceChunk = 0
    var chunkBusy = false     // one chunk in flight at a time; harvest never blocks
    var chunkNth = [Int: Double]()   // thread count -> best chunk seconds. The first
                                     // chunks A/B 4 -> 2 -> 1 (AMX is per-CLUSTER, not
                                     // per-core — more threads can anti-scale, measured
                                     // 12.6 s @ 4t); later chunks use the measured winner

    // augmentation noise pool: the old per-pixel gauss() was ~24M Box-Muller calls
    // per chunk. Random start offset per sample, sequential reads after that.
    static let gaussPool: [Float] = {
        var p = [Float](repeating: 0, count: 1 << 18)
        for i in 0 ..< p.count { p[i] = gauss() }
        return p
    }()
    // runtime tuning (data phase): every rtEvery decoded frames harvest the LAST
    // rtTake of them (freshest = tracks drift) and train ONE background thread while
    // decode keeps the rest — the fix for thermal channel drift on long transfers
    // (the 6.4 MB mp3 died at ~75 s: model trained cold, phone heated, tail + all
    // repairs crossed the nsym wall). Counting pauses while a tune runs and restarts
    // from 0 after the swap. Decoded payload = free labels (re-encode -> every tile).
    var rtEvery = 50, rtTake = 10          // Settings sliders; rtEvery 0 = off
    var rtCount = 0, rtBusy = false, rtTunes = 0

    var tuned = false, trainingOver = false, trainFails = 0   // START = training frames still queued on
                                              // the serial worker must EVAPORATE, or
                                              // their chunks block live data decode
                                              // (measured: 14 frames dropped = repair
                                              // budget exceeded = failed transfer)
    var TM = ["gather": 0.0, "nn": 0.0, "rs": 0.0], MN = 0

    // Metal warp + ANE decode (docs/iPhone.md bench E: 8.3 ms vs ~50 ms CPU). Built at
    // START (grid + frame size known), weights patched from the live net (ANEField).
    // Training/CAL/probing stay on the CPU net — live-weights doctrine.
    var metal: MetalWarp?
    var ane: ANEField?
    var aneArrs = [MLMultiArray]()  // one view per predict chunk into the warp buffer
    var aneOK = false, aneChecked = false, aneBuilt = false, aneDisabled = false
    var anePatched = false    // START re-runs aneReady to patch the TUNED weights in
                              // (the engine itself is minted early, at the TRAIN QR)

    var runLogURL: URL?           // per-run persistent copy: QR-receiver-log-{timestamp}
    var runLogStart = 0           // where this run begins inside the app-session log

    func out(_ s: String) { DispatchQueue.main.async {
        self.log += s + "\n"
        try? self.log.write(to: Runner.logURL, atomically: true, encoding: .utf8)
        if let u = self.runLogURL {
            try? String(self.log.dropFirst(self.runLogStart))
                .write(to: u, atomically: true, encoding: .utf8)
        }
    } }

    // ---- session -----------------------------------------------------------------
    func start(debug dbg: Bool = false) {
        guard !running else { return }
        debug = dbg
        camFrames = 0; probeNil = 0; probeIds = [:]; allIds = []; dumpN = 0
        startT = Date()
        running = true; finished = false
        phase = "wait"; plan = [:]; grid = nil; Hinv = nil; fileName = nil; v3Wire = false
        D = [:]; failed = 0; commits = 0; dropped = 0; pool = [:]; t0 = nil
        locked = false; trn = 0; terr = nil; tuned = false; trainingOver = false
        trainFails = 0; rtn = 0; rvn = 0; calDone = false; aneBuilt = false; aneOK = false
        anePatched = false
        aneDisabled = false   // a mint/patch failure is a PER-SESSION verdict — sticky
                              // disable silently cost an r8 run 693 ms/frame CPU decode
        gatherT0 = nil; lastChunkT = nil; sinceChunk = 0; chunkBusy = false
        prevS = nil; pendS = nil; keptS = nil; pendGet = nil; pendQ = .infinity; poolN = 0
        TM = ["gather": 0.0, "nn": 0.0, "rs": 0.0]; MN = 0
        TMA = ["up": 0.0, "mtl": 0.0, "pred": 0.0]; MNA = 0
        lock.lock(); decCnt = 0; spillJpgs = []; spillBytes = 0; lock.unlock()
        spilled = 0                              // a fresh Receive shows a fresh run
        statTick()
        let df = DateFormatter()
        df.dateFormat = "yyyyMMdd-HHmmss"
        runLogURL = Self.docs.appendingPathComponent("QR-receiver-log-\(df.string(from: Date())).txt")
        runLogStart = log.count
        handheld = (UserDefaults.standard.object(forKey: "handheld") as? Bool) ?? false
        wireV3 = (UserDefaults.standard.object(forKey: "wirev3") as? Bool) ?? false
        relocks = 0
        rtEvery = (UserDefaults.standard.object(forKey: "rtEvery") as? Int) ?? 50
        rtTake = max(1, min((UserDefaults.standard.object(forKey: "rtTake") as? Int) ?? 10,
                            max(rtEvery, 1)))
        rtCount = 0; rtBusy = false; rtTunes = 0
        hhHinv = nil; hhPend = nil; hhSeen = []; hhInFlight = false; hhMiss = 0
        hhShiftPos = 0; hhRelockBusy = false; hhRelockFails = 0; hhBlockMsg = false
        hhFramesN = 0; hhNamed = 0; hhSwaps = 0
        relockShift = 0; relockFull = 0; afKicks = 0; hhRetries = 0
        lock.lock(); hhNewHinv = nil; hhResults = []; hhRelockOutcome = nil; lock.unlock()
        loadNet(px: curPx, set: wantSet())   // provisional (CAL before any QR); the
        out("receiver armed — aim at the transmitter (idle field / TRAIN / START QR)")
        let zc = (UserDefaults.standard.object(forKey: "zeroCopyCapture") as? Bool) ?? true
        camera.zeroCopy = zc && !dbg && !handheld   // debug probing and the HH best-shot
                                                    // ledger both need owned frame bytes
        if handheld {
            calDone = true       // auto-CAL under motion scores -1 everywhere and its
                                 // lapvar fallback would lock a garbage lens; the CAL
                                 // button still works when braced
            out("HANDHELD v2: per-frame header naming + best-shot decode "
                + "(copy capture, auto-CAL off)")
        }
        camera.onFrame = { [weak self] buf, w, h in self?.dedup(buf, w, h) }
        camera.onPixelBuffer = { [weak self] pb, w, h in self?.dedupPB(pb, w, h) }
        if let err = camera.start() { out("camera FAILED: \(err)"); running = false }
        else if !camera.info.isEmpty { out(camera.info) }
        out("capture: " + (camera.zeroCopy ? "ZERO-COPY (sample in place, copy on commit)"
                                           : "copy every frame"))
    }

    func stop() {
        camera.stop()
        running = false
        out("stopped")
    }

    // ---- dedup (port of the receiver's capture loop) ------------------------------
    func small(_ f: [UInt8], _ w: Int, _ h: Int) -> [Int16] {
        var s = [Int16](repeating: 0, count: (h / 16) * (w / 16))
        f.withUnsafeBufferPointer { p in
            var i = 0
            for y in stride(from: 0, to: h - 15, by: 16) {
                for x in stride(from: 0, to: w - 15, by: 16) {
                    let o = (y * w + x) * 4
                    let r = Int(p[o + 2]), g = Int(p[o + 1]), b = Int(p[o])
                    let gray = (299 * r + 587 * g + 114 * b) / 1000
                    s[i] = Int16(gray)
                    i += 1
                }
            }
        }
        return s
    }

    func colorful(_ f: [UInt8], _ w: Int, _ h: Int) -> Bool {
        var hits = 0, n = 0
        f.withUnsafeBufferPointer { p in
            for y in stride(from: 0, to: h - 15, by: 16) {
                for x in stride(from: 0, to: w - 15, by: 16) {
                    let o = (y * w + x) * 4
                    let mx = max(p[o], max(p[o + 1], p[o + 2]))
                    let mn = min(p[o], min(p[o + 1], p[o + 2]))
                    if mx - mn > 60 { hits += 1 }
                    n += 1
                }
            }
        }
        return Double(hits) / Double(max(n, 1)) > 0.15
    }

    func meanAbsDiff(_ a: [Int16], _ b: [Int16]) -> Double {
        var s = 0
        for i in 0 ..< a.count { s += abs(Int(a[i]) - Int(b[i])) }
        return Double(s) / Double(a.count)
    }

    func bandMax(_ a: [Int16], _ b: [Int16]) -> Double {
        let band = a.count / 6
        var worst = 0.0
        for k in 0 ..< 6 {
            var s = 0
            for i in k * band ..< (k + 1) * band { s += abs(Int(a[i]) - Int(b[i])) }
            worst = max(worst, Double(s) / Double(band))
        }
        return worst
    }

    func dedup(_ f: [UInt8], _ w: Int, _ h: Int) {
        guard running, phase != "done" else { return }
        fw = w; fh = h
        lock.lock(); lastFrameBuf = f; lock.unlock()
        let s = small(f, w, h)
        if calibrating {                 // frozen: preview + stats only, no commits,
            prevS = s                    // no QR reads, no phase changes
            pendGet = nil; pendS = nil; pendQ = .infinity; poolN = 0
            if Date().timeIntervalSince(lastPrev) > 0.33 {
                lastPrev = Date(); buildPreview(f, w, h); sharp = lapvar(f, w, h)
            }
            if Date().timeIntervalSince(lastStat) > 0.3 { lastStat = Date(); statTick() }
            return
        }
        if let ps = prevS {
            if let pS = pendS, meanAbsDiff(s, pS) > Self.TH_CHANGE { commit(held: false) }
            let q = bandMax(s, ps)
            if q < min(Self.TH_BLEND, max(Self.TH_STABLE, pendQ)) {
                if pendS == nil { pendT = Date() }
                pendGet = { f }; pendS = s; pendQ = q
                if q < Self.TH_STABLE { poolN += 1 }
            }
            if pendS != nil, Date().timeIntervalSince(pendT) > 1.5 { commit(held: true) }
        }
        prevS = s
        if handheld, Date().timeIntervalSince(lastQRScan) > (phase == "data" ? 1.0 : 0.7) {
            lastQRScan = Date()          // motion never satisfies the held-commit rule:
            if let txt = scanQR(f, fw, fh) { _ = handleQR(txt) }   // scan raw frames
        }
        if handheld, !debug, grid?.v3 != true { hhFrame(f) }
        if debug {
            camFrames += 1
            if let hv = Hinv, let g = grid { probeID(f, hv, g) }
            if Date().timeIntervalSince(lastDbg) > 2 {
                lastDbg = Date()
                if Hinv == nil {
                    out("dbg: \(camFrames) cam frames, geom UNLOCKED, sharp \(Int(sharp))")
                } else {
                    let ids = probeIds.keys.sorted()
                    let idStr = ids.map { "\($0):\(probeIds[$0]!)" }.joined(separator: " ")
                    out("dbg: \(camFrames)f  ids[\(ids.count)] \(idStr)  "
                        + "noCRC \(probeNil)  sharp \(Int(sharp))")
                }
                camFrames = 0; probeNil = 0; probeIds = [:]
            }
            if Date().timeIntervalSince(lastDump) > 10 {
                lastDump = Date(); dumpFrame(f, w, h)
            }
        }
        if Date().timeIntervalSince(lastPrev) > 0.33 {
            lastPrev = Date(); buildPreview(f, w, h)
            sharp = lapvar(f, w, h)
        }
        if Date().timeIntervalSince(lastStat) > 0.3 { lastStat = Date(); statTick() }
        checkTimeout()
    }

    func checkTimeout() {     // never hang up on a LIVE transmission: quiet-gap based
        guard phase == "data", let t = t0, let fr = plan["FRAMES"],
              let fps = plan["FPS"] else { return }
        lock.lock(); let last = lastDecT; lock.unlock()
        if Date().timeIntervalSince(last) > 12 {
            finish("stalled (no decode for 12 s)")
        } else if Date().timeIntervalSince(t) > (fr + (plan["REPAIR"] ?? 0)) / fps * 2 + 30 {
            finish("timeout")             // runaway guard only (2x nominal + 30 s)
        }
    }

    func commit(held: Bool) {
        defer { pendGet = nil; pendS = nil; pendQ = .infinity; poolN = 0 }
        guard let get = pendGet, let pS = pendS else { return }
        let f = get()                    // zero-copy path pays its 8 MB copy HERE only
        let th = held || !colorful(f, fw, fh) ? 2.0 : Self.TH_CHANGE
        if keptS == nil || meanAbsDiff(pS, keptS!) > th {
            keep(f, held: held, poolSz: poolN)
            keptS = pS
        }
    }

    // ---- zero-copy capture (Settings toggle) -----------------------------------------
    // Dedup metrics are sampled straight from the locked camera pool buffer; the 8 MB
    // copy happens only when a dwell commits and for the 3 fps preview. We retain AT
    // MOST ONE pool buffer (the pend candidate) — holding more starves the ~6-deep
    // capture pool, which presents as the "flaky camera" all over again. CAL routes to
    // the copy path (it wants fresh full frames); debug never enters (start() forces
    // copy mode there).
    static func copyPB(_ pb: CVPixelBuffer) -> [UInt8] {
        CVPixelBufferLockBaseAddress(pb, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pb, .readOnly) }
        let w = CVPixelBufferGetWidth(pb), h = CVPixelBufferGetHeight(pb)
        let rowB = CVPixelBufferGetBytesPerRow(pb)
        let base = CVPixelBufferGetBaseAddress(pb)!.assumingMemoryBound(to: UInt8.self)
        var buf = [UInt8](repeating: 0, count: w * h * 4)
        buf.withUnsafeMutableBufferPointer { p in
            if rowB == w * 4 { memcpy(p.baseAddress!, base, w * h * 4) }
            else { for y in 0 ..< h { memcpy(p.baseAddress! + y * w * 4, base + y * rowB, w * 4) } }
        }
        return buf
    }

    func smallPB(_ pb: CVPixelBuffer, _ w: Int, _ h: Int) -> [Int16] {
        CVPixelBufferLockBaseAddress(pb, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pb, .readOnly) }
        let rowB = CVPixelBufferGetBytesPerRow(pb)
        let p = CVPixelBufferGetBaseAddress(pb)!.assumingMemoryBound(to: UInt8.self)
        var s = [Int16](repeating: 0, count: (h / 16) * (w / 16))
        var i = 0
        for y in stride(from: 0, to: h - 15, by: 16) {
            for x in stride(from: 0, to: w - 15, by: 16) {
                let o = y * rowB + x * 4
                let r = Int(p[o + 2]), g = Int(p[o + 1]), b = Int(p[o])
                s[i] = Int16((299 * r + 587 * g + 114 * b) / 1000)
                i += 1
            }
        }
        return s
    }

    func dedupPB(_ pb: CVPixelBuffer, _ w: Int, _ h: Int) {
        guard running, phase != "done" else { return }
        if calibrating { dedup(Self.copyPB(pb), w, h); return }
        fw = w; fh = h
        let s = smallPB(pb, w, h)
        if let ps = prevS {
            if let pS = pendS, meanAbsDiff(s, pS) > Self.TH_CHANGE { commit(held: false) }
            let q = bandMax(s, ps)
            if q < min(Self.TH_BLEND, max(Self.TH_STABLE, pendQ)) {
                if pendS == nil { pendT = Date() }
                pendGet = { Self.copyPB(pb) }    // retains pb until commit/replacement
                pendS = s; pendQ = q
                if q < Self.TH_STABLE { poolN += 1 }
            }
            if pendS != nil, Date().timeIntervalSince(pendT) > 1.5 { commit(held: true) }
        }
        prevS = s
        if handheld, Date().timeIntervalSince(lastQRScan) > (phase == "data" ? 1.0 : 0.7) {
            lastQRScan = Date()          // motion never satisfies the held-commit rule:
            let fb = Self.copyPB(pb)     // scan raw frames for the control QRs
            if let txt = scanQR(fb, w, h) { _ = handleQR(txt) }
        }
        if Date().timeIntervalSince(lastPrev) > 0.33 {
            lastPrev = Date()
            let f = Self.copyPB(pb)
            lock.lock(); lastFrameBuf = f; lock.unlock()
            buildPreview(f, w, h)
            sharp = lapvar(f, w, h)
        }
        if Date().timeIntervalSince(lastStat) > 0.3 { lastStat = Date(); statTick() }
        checkTimeout()
    }

    // ---- phase machine -------------------------------------------------------------
    // control-plane QR handling, shared by held commits and the handheld periodic scan
    func handleQR(_ txt: String) -> Bool {
        if txt.hasPrefix("---START---") || txt.hasPrefix("---TRAIN---") {
            for kv in txt.split(separator: ",").dropFirst() {
                let p = kv.split(separator: "=")
                if p.count == 2 {
                    if p[0] == "NAME" { fileName = String(p[1]) }   // string, not a plan
                    else { plan[String(p[0])] = Double(p[1]) }      // Double like the rest
                }
            }
            let vAnn = (plan["V"] ?? 2) >= 3
            let v3ok = vAnn && wireV3 && Int(plan["LV"] ?? 2) == 2   // PAM-2 only
            if vAnn, !v3ok, !v3Wire {
                v3Wire = true
                trainingOver = true      // glyph net must never train on DMT cells
                calDone = true           // CAL scores glyphs vs truth — meaningless here
                out("v3 DMT wire announced — menu wire is v2 (or PAM-4): "
                    + "footage/log mode (glyph training + auto-CAL disabled)")
            }
            if grid == nil, let c = plan["COLS"], let r = plan["ROWS"] {
                let px = Int(plan["PX"] ?? 16)
                let scaled = px != 16 && (v3ok    // the demapper's 20x20 IS the C=20
                    // canonical window — a scaled r12 v3 cell is statistically the
                    // r16 training data, so v3 px!=16 is ALWAYS canonical
                    || (UserDefaults.standard
                        .object(forKey: "scale16r12") as? Bool ?? false)
                    || (px != 12 && px != 16))     // canonical-16: ANY px reads through
                    // the r16 net shape (12c16 proved it; 30c16 = handheld big glyphs).
                    // Only 12/16 have native bundled weights, so other px are FORCED
                    // scaled — native would load r16 bytes into a mismatched net
                grid = Grid(px: px, cols: Int(c), rows: Int(r),
                            nsym: Int(plan["NSYM"] ?? 32), scaled: scaled,
                            v3: v3ok, vnc: Int(plan["NC"] ?? 8))
                out("grid \(Int(c))x\(Int(r)) r\(px)"
                    + (scaled ? " SCALED to 16px input" : "")
                    + (v3ok ? " v3.1 PAM-2 nc\(grid!.vnc)" : "")
                    + " nsym \(grid!.nsym) kpf \(grid!.kpf)")
                let key = gridKey(grid!)
                if net == nil || curPx != key || curSet != wantSet() {
                    loadNet(px: key, set: wantSet())    // QR px + scale16 + model SET —
                }   // the set check stops an hh-trained in-memory net leaking into the
                    // next stable session (and vice versa)
                if grid!.v3 {
                    trainingOver = true  // demapper on-device training: not yet — the
                    calDone = true       // pretrained model decodes; glyph CAL is moot
                    loadDemap(px: key, nc: grid!.vnc)
                }
                if grid!.K == 16, !grid!.v3, !aneBuilt {   // mint/warm the ANE engine
                    aneBuilt = true             // DURING training; START re-patches
                    workQ.async { self.aneReady() }        // (v3 decodes on the CPU
                }                                          // demapper — no mint)
            }
            if let g = grid, let ns = plan["NSYM"], Int(ns) != g.nsym {
                // grid was built at the TRAIN QR, which predates NSYM: an nsym-64
                // codeword PASSES an nsym-32 check (generator roots nest), so a stale
                // nsym decodes "cleanly" into garbage — rebuild before data flows
                grid = Grid(px: g.px, cols: g.cols, rows: g.rows, nsym: Int(ns),
                            scaled: g.scaled, v3: g.v3, vnc: g.vnc)
                out("nsym \(g.nsym) -> \(Int(ns)) (QR) kpf \(grid!.kpf)")
            }
            if !locked {
                locked = true
                camera.settleAndLock { msg in
                    self.out(msg)             // then refine focus by truth score
                    if !self.calDone { self.calDone = true; self.calibrate() }
                }
            }
            let newPhase = txt.hasPrefix("---TRAIN---") ? "train" : "armed"
            if newPhase == "armed" {
                trainingOver = true          // queued training frames now no-op
                if trn > 0, !tuned { workQ.async { self.saveTuned() } }
                if !anePatched, grid?.v3 != true {   // TUNED weights patched in +
                    anePatched = true        // fresh cross-check before the first data
                    aneBuilt = true          // frame (engine minted back at the TRAIN QR;
                    workQ.async { self.aneReady() }   // aneReady reuses it and only
                }                                     // re-patches — ~0.3 s)
            }
            if phase != newPhase { out("QR: \(txt.split(separator: ",")[0]) -> \(newPhase)") }
            phase = newPhase
            return true
        }
        if txt.hasPrefix("---END---") { finish("END seen"); return true }
        return false
    }

    func keep(_ f: [UInt8], held: Bool, poolSz: Int) {
        commits += 1
        if held || !colorful(f, fw, fh) {
            if let txt = scanQR(f, fw, fh), handleQR(txt) { return }
            if !colorful(f, fw, fh) { return }
        }
        if debug, grid == nil, Date().timeIntervalSince(startT) > 5 {
            grid = Grid(px: 16, cols: 92, rows: 57, nsym: 32)   // this rig's known grid
            out("dbg: no QR after 5 s -> fallback grid 92x57 adopted")
            if net == nil || curPx != 16 || curSet != wantSet() {
                loadNet(px: 16, set: wantSet())
            }
            if !locked { locked = true; camera.settleAndLock { self.out($0) } }
        }
        let big = fw > 2000        // 4K frames are 33 MB — cap the queues by RAM, not count
        switch phase {
        case "train":
            // harvest exactly the NEXT batch, then stop: unbounded harvest ate 2-3
            // cores (gather+classify per frame) and starved the chunk threads
            // (measured: 20 s chunks). Gate reopens when the chunk consumes the batch.
            if !held, sinceChunk < Self.TRAIN_EVERY, backlogCount() < (big ? 6 : Self.QCAP) {
                addBacklog(1)
                workQ.async { self.trainFrame(f); self.addBacklog(-1) }
            }
        case "armed", "data":
            if hhActive { return }        // the HH pipeline owns data decode AND the
                                          // armed->data flip (single owner, no race)
            if phase == "armed" {
                if held { return }
                phase = "data"; t0 = Date()
                lock.lock(); lastDecT = Date(); lock.unlock()
                out("data phase — receiving")
            }
            if backlogCount() < (big ? 8 : 30) {
                addBacklog(1)
                workQ.async { self.dataFrame(f, poolSz); self.addBacklog(-1) }
            } else { spill(f, poolSz) }
        default:
            if debug, grid != nil, Hinv == nil, !held, backlogCount() < 2 {
                addBacklog(1)               // no QR needed: lock geometry off any frame
                workQ.async { self.dbgLock(f); self.addBacklog(-1) }
            }
        }
    }

    func dbgLock(_ f: [UInt8]) {
        guard Hinv == nil, let g = grid, f.count == fw * fh * 4 else { return }
        let cands = Geometry.findQuad(f, fw, fh, cols: g.cols, rows: g.rows,
                                      px: g.px, C: g.C)
        for hv in cands {
            if let (idx, _) = tryDecode(f, hv, g) {
                Hinv = hv
                out("dbg: geometry LOCKED (header idx \(idx)) — probing every frame now")
                return
            }
        }
        out("dbg lock attempt failed: \(cands.count) cands, " + Geometry.lastInfo)
    }

    func backlogCount() -> Int { lock.lock(); defer { lock.unlock() }; return backlog }
    func addBacklog(_ d: Int) { lock.lock(); backlog += d; lock.unlock() }

    func scanQR(_ f: [UInt8], _ w: Int, _ h: Int) -> String? {
        var copy = f
        let info = CGBitmapInfo.byteOrder32Little.rawValue
                 | CGImageAlphaInfo.noneSkipFirst.rawValue
        guard let ctx = CGContext(data: &copy, width: w, height: h, bitsPerComponent: 8,
                                  bytesPerRow: w * 4, space: CGColorSpaceCreateDeviceRGB(),
                                  bitmapInfo: info),
              let img = ctx.makeImage() else { return nil }
        let req = VNDetectBarcodesRequest()
        req.symbologies = [.qr]
        try? VNImageRequestHandler(cgImage: img, options: [:]).perform([req])
        return req.results?.compactMap { $0.payloadStringValue }
            .first { $0.hasPrefix("---") }
    }

    // ---- decode --------------------------------------------------------------------
    func gatherX(_ f: [UInt8], _ hv: [Double], _ g: Grid) -> [Float] {
        let K = g.K, n1 = K * K
        var X = [Float](repeating: 0, count: g.T * 3 * n1)
        let w = fw, h = fh, C = Double(g.C)
        X.withUnsafeMutableBufferPointer { xp in
            f.withUnsafeBufferPointer { sp in
                DispatchQueue.concurrentPerform(iterations: 8) { k in
                    let per = (g.T + 7) / 8
                    for t in k * per ..< min((k + 1) * per, g.T) {
                        let r = t / g.cols, c = t % g.cols
                        for pix in 0 ..< n1 {
                            let y = pix / K, x = pix % K
                            let dx = Double(c) * C + Double(x) + 2
                            let dy = Double(r) * C + Double(y) + 2
                            let q = hv[6] * dx + hv[7] * dy + hv[8]
                            let sx = (hv[0] * dx + hv[1] * dy + hv[2]) / q
                            let sy = (hv[3] * dx + hv[4] * dy + hv[5]) / q
                            let o = t * 3 * n1 + pix
                            // double-domain bounds: NaN/inf from a degenerate CANDIDATE
                            // homography land here instead of trapping in Int()
                            if !(sx >= 0 && sy >= 0 && sx + 1 < Double(w) && sy + 1 < Double(h)) {
                                xp[o] = -0.5; xp[o + n1] = -0.5; xp[o + 2 * n1] = -0.5
                                continue
                            }
                            let x0 = Int(sx), y0 = Int(sy)
                            let fx = Float(sx) - Float(x0), fy = Float(sy) - Float(y0)
                            let b = (y0 * w + x0) * 4, b2 = b + w * 4
                            for ch in 0 ..< 3 {         // BGRA -> RGB
                                let sc = 2 - ch
                                let v = Float(sp[b + sc]) * (1 - fx) * (1 - fy)
                                      + Float(sp[b + 4 + sc]) * fx * (1 - fy)
                                      + Float(sp[b2 + sc]) * (1 - fx) * fy
                                      + Float(sp[b2 + 4 + sc]) * fx * fy
                                xp[o + ch * n1] = max(0, min(255, v.rounded())) / 255 - 0.5
                            }
                        }
                    }
                }
            }
        }
        return X
    }

    func gatherCrops20(_ f: [UInt8], _ hv: [Double], _ g: Grid) -> [UInt8] {
        let Ci = g.C, cn = Ci * Ci
        var crops = [UInt8](repeating: 0, count: g.T * cn * 3)
        let w = fw, h = fh, C = Double(g.C)
        crops.withUnsafeMutableBufferPointer { cp in
            f.withUnsafeBufferPointer { sp in
                for t in 0 ..< g.T {
                    let r = t / g.cols, c = t % g.cols
                    for pix in 0 ..< cn {
                        let y = pix / Ci, x = pix % Ci
                        let dx = Double(c) * C + Double(x), dy = Double(r) * C + Double(y)
                        let q = hv[6] * dx + hv[7] * dy + hv[8]
                        let sx = (hv[0] * dx + hv[1] * dy + hv[2]) / q
                        let sy = (hv[3] * dx + hv[4] * dy + hv[5]) / q
                        let o = (t * cn + pix) * 3
                        if !(sx >= 0 && sy >= 0 && sx + 1 < Double(w) && sy + 1 < Double(h)) {
                            continue           // incl. NaN/inf — Int() traps on those
                        }
                        let x0 = Int(sx), y0 = Int(sy)
                        let fx = Float(sx) - Float(x0), fy = Float(sy) - Float(y0)
                        let b = (y0 * w + x0) * 4, b2 = b + w * 4
                        for ch in 0 ..< 3 {
                            let sc = 2 - ch
                            let v = Float(sp[b + sc]) * (1 - fx) * (1 - fy)
                                  + Float(sp[b + 4 + sc]) * fx * (1 - fy)
                                  + Float(sp[b2 + sc]) * (1 - fx) * fy
                                  + Float(sp[b2 + 4 + sc]) * fx * fy
                            cp[o + ch] = UInt8(max(0, min(255, v.rounded())))
                        }
                    }
                }
            }
        }
        return crops
    }

    func classify(_ X: [Float], _ T: Int) -> [UInt8] {
        let n = 3 * net.k * net.k
        var sym = [UInt8](repeating: 0, count: T)
        sym.withUnsafeMutableBufferPointer { p in
            DispatchQueue.concurrentPerform(iterations: 8) { k in
                var A = Net.Acts(k: self.net.k)
                var x = [Float](repeating: 0, count: n)
                let per = (T + 7) / 8
                for t in k * per ..< min((k + 1) * per, T) {
                    for j in 0 ..< n { x[j] = X[t * n + j] }
                    self.net.forward(x, &A)
                    let s = A.s.indices.max { A.s[$0] < A.s[$1] }!
                    let c = A.c.indices.max { A.c[$0] < A.c[$1] }!
                    p[t] = UInt8(s << 2 | c)
                }
            }
        }
        return sym
    }

    func headerIdx(_ sym: [UInt8], _ g: Grid) -> Int? {
        var votes = [Int](repeating: 0, count: 42)
        for b in 0 ..< 294 { votes[b % 42] += Int(sym[g.hdr[b / 6]]) >> (5 - b % 6) & 1 }
        var idx = 0, crc = 0
        for j in 0 ..< 30 { idx = idx << 1 | (votes[j] > 3 ? 1 : 0) }
        for j in 30 ..< 42 { crc = crc << 1 | (votes[j] > 3 ? 1 : 0) }
        let be = (0 ..< 4).map { UInt8(idx >> (24 - 8 * $0) & 0xFF) }
        return crc == Int(crc32(be) & 0xFFF) ? idx : nil
    }

    // full decode attempt: (headerIdx, payload bytes or nil) — nil result = CRC fail
    func tryDecode(_ f: [UInt8], _ hv: [Double], _ g: Grid) -> (Int, [UInt8]?)? {
        guard f.count == fw * fh * 4 else { return nil }   // a queued frame from before
        var t = Date()                                     // a rotation flip: skip, never
                                                           // index past its end
        let X = gatherX(f, hv, g)
        TM["gather"]! += -t.timeIntervalSinceNow; t = Date()
        let sym = classify(X, g.T)
        TM["nn"]! += -t.timeIntervalSinceNow
        return decodeSym(sym, g)
    }

    func aneReady() {         // workQ, once per session at START: build + patch weights
        guard !aneDisabled else { out("ANE disabled this session — CPU decode"); return }
        guard let g = grid, fw > 0 else { return }
        guard g.K == 16 else {           // the ANE models are 16px-input builds
            out("ANE engine needs 16px inputs — CPU decode at r\(g.px)"); return
        }
        // the mlprogram bakes the tile COUNT into its shapes — but T is only the BATCH
        // dim, so any grid's model is MINTED on-device from the TileNet2 template
        // (ANEField.reshape). Grids past the 2-byte-varint cap (16383 — fullscreen r8
        // is 21090) mint a HALF/THIRD-size batch and predict in chunks off the same
        // warp buffer. The two shipped grids keep their bundled builds.
        let resName = g.T == 9348 ? "TileNet12" : "TileNet2"
        let tmplT = g.T == 9348 ? 9348 : 5244
        let nCh = (g.T + 16383) / 16384          // predicts per frame
        let Tc = (g.T + nCh - 1) / nCh           // minted batch size
        if let a = ane, a.T != Tc {      // grid changed since last session: rebuild all
            ane = nil; metal = nil; aneArrs = []
        }
        if let mw = metal, mw.sw != fw || mw.sh != fh {
            metal = nil; aneArrs = []    // camera RESOLUTION changed between sessions
        }                                // (4K toggle / handheld's forced 1080p) with
                                         // the grid unchanged: a stale 4K texture's
                                         // tex.replace reads past a 1080p buffer's end
                                         // — the START-transition SIGSEGV (crash
                                         // reports 2026-07-18 13:19 + 22:30)
        if ane == nil {
            guard let fac = Bundle.main.url(forResource: "weights", withExtension: "bin")
                    .flatMap({ try? Data(contentsOf: $0) }),
                  let a = ANEField(T: Tc, factory: fac, resource: resName, templateT: tmplT)
            else { aneDisabled = true; out("ANE engine unavailable — CPU decode"); return }
            ane = a
            if Tc != tmplT {
                out("ANE: minted on-device model for T=\(Tc)"
                    + (nCh > 1 ? " (\(nCh) chunks over \(g.T) tiles)" : ""))
            }
        }
        if metal == nil {
            // fp32 input + Core ML's own fp16 conversion BEATS a native-fp16
            // input (predict 16.3 vs 20.3 ms measured live) — don't "optimize"
            guard let mw = MetalWarp(sw: fw, sh: fh, cols: g.cols, rows: g.rows,
                                     bgra: true, outTiles: nCh * Tc)
            else { aneDisabled = true; out("ANE engine unavailable — CPU decode"); return }
            var arrs = [MLMultiArray]()
            for c in 0 ..< nCh {
                guard let arr = try? MLMultiArray(
                    dataPointer: mw.outBuf.contents() + c * Tc * 768 * 4,
                    shape: [NSNumber(value: Tc), 3, 16, 16], dataType: .float32,
                    strides: [768, 256, 16, 1], deallocator: nil)
                else { aneDisabled = true; out("ANE engine unavailable — CPU decode"); return }
                arrs.append(arr)
            }
            metal = mw; aneArrs = arrs
        }
        guard ane!.rebuild(net) else {
            aneDisabled = true
            out("ANE weight patch FAILED — CPU decode"); return
        }
        aneOK = true; aneChecked = false
        out("ANE decode LIVE (Metal warp + Core ML ANE, session weights patched in)")
    }

    var TMA = ["up": 0.0, "mtl": 0.0, "pred": 0.0], MNA = 0   // ANE-path stage split

    func aneDecode(_ f: [UInt8], _ hv: [Double], _ g: Grid) -> (Int, [UInt8]?)? {
        guard let mw = metal, let a = ane, !aneArrs.isEmpty,
              f.count == mw.sw * mw.sh * 4       // any size mismatch -> CPU, never a
        else { return tryDecode(f, hv, g) }      // driver read past the buffer end
        var t = Date()
        mw.setHinv(hv)
        mw.upload(f)
        TMA["up"]! += -t.timeIntervalSinceNow; TM["gather"]! += -t.timeIntervalSinceNow
        t = Date()
        mw.run()
        TMA["mtl"]! += -t.timeIntervalSinceNow; TM["gather"]! += -t.timeIntervalSinceNow
        t = Date()
        guard var sym = a.classify(aneArrs, g.T) else {
            aneOK = false; out("ANE predict FAILED — CPU decode")
            return tryDecode(f, hv, g)
        }
        TMA["pred"]! += -t.timeIntervalSinceNow; TM["nn"]! += -t.timeIntervalSinceNow
        MNA += 1
        if MNA % 100 == 0 {
            out(String(format: "ane stages: upload %.1f  metal %.1f  predict %.1f ms/frame",
                       TMA["up"]! / Double(MNA) * 1000, TMA["mtl"]! / Double(MNA) * 1000,
                       TMA["pred"]! / Double(MNA) * 1000))
        }
        if !aneChecked {                 // one-time cross-check vs the CPU net
            aneChecked = true
            let cpu = classify(gatherX(f, hv, g), g.T)
            let diff = zip(sym, cpu).filter { $0 != $1 }.count
            out("ANE vs CPU cross-check: \(diff)/\(g.T) tiles differ")
            if diff > g.T / 100 {
                aneOK = false; out("ANE MISMATCH — falling back to CPU decode")
                sym = cpu
            }
        }
        return decodeSym(sym, g)
    }

    func decodeSym(_ sym: [UInt8], _ g: Grid) -> (Int, [UInt8]?)? {
        guard let idx = headerIdx(sym, g) else { return nil }
        let t = Date()
        var bits = [UInt8](); bits.reserveCapacity(g.pay.count * 6)
        for p in g.pay {
            let s = Int(sym[p])
            for b in 0 ..< 6 { bits.append(UInt8(s >> (5 - b) & 1)) }
        }
        var bytes = [UInt8](repeating: 0, count: bits.count / 8)
        for i in 0 ..< bytes.count {
            var v = 0
            for b in 0 ..< 8 { v = v << 1 | Int(bits[i * 8 + b]) }
            bytes[i] = UInt8(v)
        }
        var data = [UInt8](); data.reserveCapacity(g.kpf)
        var allOk = true
        for b in 0 ..< g.nblk {                    // deinterleave: block b = flat[i*nblk+b]
            let block = (0 ..< 255).map { bytes[$0 * g.nblk + b] }
            let (dec, ok) = RS.decode(block, g.nsym)
            if !ok { allOk = false; break }
            data += dec
        }
        TM["rs"]! += -t.timeIntervalSinceNow; MN += 1
        return (idx, allOk ? data : nil)
    }

    func dataFrame(_ f: [UInt8], _ poolSz: Int) {
        guard let g = grid else { return }
        func attempt(_ hv: [Double]) -> (Int, [UInt8]?)? {
            g.v3 ? v3Decode(f, hv, g) : (aneOK ? aneDecode(f, hv, g) : tryDecode(f, hv, g))
        }
        var res: (Int, [UInt8]?)?
        if let hv = Hinv { res = attempt(hv) }
        if res == nil, Hinv == nil || handheld {   // bootstrap, or the hand moved:
            for hv in Geometry.findQuad(f, fw, fh, cols: g.cols, rows: g.rows,
                                        px: g.px, C: g.C) {   // re-find the ring NOW
                res = attempt(hv)
                if res != nil {
                    if Hinv == nil { out("geometry locked (data)") } else { relocks += 1 }
                    Hinv = hv
                    break
                }
            }
        }
        var p = pool[poolSz] ?? [0, 0]
        if let (idx, data) = res, let d = data {
            D[idx] = d; p[0] += 1
            lock.lock()
            decCnt = D.keys.filter { $0 < Int(plan["FRAMES"] ?? 1e9) }.count
            lastDecT = Date()
            lock.unlock()
            if let hv = Hinv, !g.v3 { rtTick(f, hv, idx, d, g) }   // runtime tuning is
        } else {                                                  // glyph-only for now
            failed += 1; p[1] += 1
        }
        pool[poolSz] = p
    }

    var decCnt = 0        // decoded-source counter mirrored for the stats thread (the
                          // D dictionary itself must never be read cross-thread)
    var lastDecT = Date() // last successful decode (lock) — the timeout is PROGRESS-
                          // based: a schedule-based one hung up on a LIVE transmission
                          // running 17% behind nominal (rAF tick stretch accumulates)

    func spill(_ f: [UInt8], _ poolSz: Int) {   // camera thread: hand off, never block
        let w = fw, h = fh
        lock.lock(); let room = spillBytes < 200 << 20; lock.unlock()
        guard room else { dropped += 1; return }     // ~200 MB cap, THEN drop for real
        spillQ.async {
            guard let d = self.jpegData(f, w, h) else { return }
            self.lock.lock()
            self.spillJpgs.append((d, poolSz)); self.spillBytes += d.count; self.spilled += 1
            self.lock.unlock()
        }
    }

    func drainSpill() {                          // workQ (assemble), after END: decode
        spillQ.sync {}                           // late instead of never. Barrier waits
        lock.lock()                              // for in-flight encodes.
        let sj = spillJpgs; spillJpgs = []; spillBytes = 0
        lock.unlock()
        guard !sj.isEmpty else { return }
        for (d, ps) in sj { if let f = jpegFrame(d) { dataFrame(f, ps) } }
        out("spill: \(sj.count) late frames decoded (backlog was full — not dropped)")
    }

    // ---- handheld v2 (HH pipeline) ----------------------------------------------------
    // Per camera frame (60 fps): probe the 49 header tiles with the CPU net (~3 ms,
    // proven by debug mode at 30 fps) -> (idx, margin score). Geometry is TRACKED: the
    // last-good Hinv plus a translation shift ring absorbs hand drift; the full ring
    // search is a throttled workQ fallback and doubles as the bootstrap (dedup never
    // commits under motion, so nothing else would seed geometry). The best-scoring shot
    // per idx gets ONE full ANE decode; an RS fail un-blocks the idx so later shots of
    // a live dwell retry. Constant-velocity sweeps decode fine (their rolling-shutter
    // shear is affine); jerks miss and fountain repair absorbs them.

    func hhProbe(_ f: [UInt8], _ hv: [Double], _ g: Grid) -> (idx: Int, score: Double)? {
        guard let nn = net, nn.k == g.K else { return nil }
        let n = 3 * g.K * g.K
        var X = [Float](repeating: 0, count: 49 * n)
        gatherTiles(f, hv, g, g.hdr, into: &X)
        var A = Net.Acts(k: nn.k)
        var x = [Float](repeating: 0, count: n)
        var syms = [UInt8](repeating: 0, count: 49)
        var score = 0.0
        for t in 0 ..< 49 {
            for j in 0 ..< n { x[j] = X[t * n + j] }
            nn.forward(x, &A)
            var s = 0, c = 0
            for i in 1 ..< 16 where A.s[i] > A.s[s] { s = i }
            for i in 1 ..< 4 where A.c[i] > A.c[c] { c = i }
            var s2 = -Float.infinity, c2 = -Float.infinity
            for i in 0 ..< 16 where i != s { s2 = max(s2, A.s[i]) }
            for i in 0 ..< 4 where i != c { c2 = max(c2, A.c[i]) }
            score += Double(A.s[s] - s2) + Double(A.c[c] - c2)   // margin = shot quality
            syms[t] = UInt8(s << 2 | c)
        }
        var votes = [Int](repeating: 0, count: 42)
        for b in 0 ..< 294 { votes[b % 42] += Int(syms[b / 6]) >> (5 - b % 6) & 1 }
        var idx = 0, crc = 0
        for j in 0 ..< 30 { idx = idx << 1 | (votes[j] > 3 ? 1 : 0) }
        for j in 30 ..< 42 { crc = crc << 1 | (votes[j] > 3 ? 1 : 0) }
        let be = (0 ..< 4).map { UInt8(idx >> (24 - 8 * $0) & 0xFF) }
        return crc == Int(crc32(be) & 0xFFF) ? (idx, score) : nil
    }

    func hhShifted(_ hv: [Double], _ tx: Double, _ ty: Double) -> [Double] {
        var h = hv                        // image-space translation composed into Hinv:
        for j in 0 ..< 3 {                // sx' = sx + tx, sy' = sy + ty
            h[j] += tx * hv[6 + j]
            h[3 + j] += ty * hv[6 + j]
        }
        return h
    }

    func hhFrame(_ f: [UInt8]) {          // camera thread, every frame
        guard let g = grid, net != nil else { return }
        guard g.K == 16 else {
            if !hhBlockMsg {
                hhBlockMsg = true
                out("HANDHELD: native r12 has no ANE model — using the v1.2 re-lock "
                    + "path (enable 'Scale to 16 px' for the real handheld pipeline)")
            }
            return
        }
        lock.lock()                        // drain the workQ mailboxes
        let newHv = hhNewHinv; hhNewHinv = nil
        let results = hhResults; hhResults = []
        let relockO = hhRelockOutcome; hhRelockOutcome = nil
        lock.unlock()
        if let o = relockO {
            hhRelockBusy = false
            if o { relockFull += 1 } else { hhRelockFails += 1 }
        }
        if let hv = newHv { hhHinv = hv; Hinv = hv; hhMiss = 0 }
        for r in results {
            hhInFlight = false
            if r.ok {
                hhSeen.insert(r.idx)
                if phase == "armed" {      // first RS success = data really started
                    phase = "data"; t0 = Date()
                lock.lock(); lastDecT = Date(); lock.unlock()
                    out("data phase — receiving (HH)")
                }
            } else if phase == "data" {
                hhRetries += 1             // idx stays un-seen: later shots retry
            }
        }
        guard phase == "train" || phase == "armed" || phase == "data" else { return }
        hhFramesN += 1
        var named: (idx: Int, score: Double)?
        if let hv = hhHinv {
            named = hhProbe(f, hv, g)
            if named == nil {              // 2 shifted probes/frame: the ±8/±16 ring is
                for _ in 0 ..< 2 {         // covered in 4 frames, inside one dwell
                    let (tx, ty) = Self.hhShiftRing[hhShiftPos]
                    hhShiftPos = (hhShiftPos + 1) % Self.hhShiftRing.count
                    let sh = hhShifted(hv, tx, ty)
                    if let nm = hhProbe(f, sh, g) {
                        hhHinv = sh; Hinv = sh; relockShift += 1
                        named = nm
                        break
                    }
                }
            }
        }
        guard let (idx, score) = named else {
            hhMiss += 1
            hhMaybeRelock(f)
            hhFlushStale()
            return
        }
        hhMiss = 0; hhRelockFails = 0
        hhNamed += 1
        let hv = hhHinv!
        if phase == "train" {              // feeder: probe-named frames harvest at ~8/s
            if Date().timeIntervalSince(hhLastHarvest) > 0.12,
               sinceChunk < Self.TRAIN_EVERY, backlogCount() < Self.QCAP {
                hhLastHarvest = Date()
                addBacklog(1)
                workQ.async { self.trainFrame(f); self.addBacklog(-1) }
            }
        } else {                           // armed / data: best-shot ledger
            if hhSeen.contains(idx) {
                hhPend = nil
            } else if let p = hhPend, p.idx == idx {
                if score > p.score { hhPend = (idx, score, f, hv, p.t); hhSwaps += 1 }
            } else {
                if let p = hhPend { hhFlush(p) }
                hhPend = (idx, score, f, hv, Date())
            }
            hhFlushStale()
        }
        if phase == "data", Date().timeIntervalSince(hhLastLog) > 2 {
            hhLastLog = Date()
            out("HH: named \(hhNamed)/\(hhFramesN)  relocks \(relockShift)s/\(relockFull)f"
                + (hhRetries > 0 ? "  retries \(hhRetries)" : ""))
        }
    }

    // flush the pend once it has collected shots for ~50 ms (the FIRST shot of a dwell
    // is usually the ISP transition blend — dedup doctrine — so give better shots a
    // beat to arrive), which also leaves the dwell's tail free to retry an RS fail.
    // In armed the held START screen fails RS BY DESIGN (train_img(0) payload), and
    // data frame 0 shares its header idx 0 — so pace armed attempts at 150 ms instead.
    func hhFlushStale() {
        guard let p = hhPend else { return }
        if Date().timeIntervalSince(p.t) > (phase == "armed" ? 0.15 : 0.05),
           hhFlush(p) {                    // keep the pend if a decode is in flight —
            hhPend = nil                   // the next frame retries the flush
        }
    }

    @discardableResult
    func hhFlush(_ p: (idx: Int, score: Double, f: [UInt8], hv: [Double], t: Date),
                 force: Bool = false) -> Bool {
        if hhSeen.contains(p.idx) { return true }        // done — drop the pend
        guard force || (!hhInFlight && backlogCount() < 4) else { return false }
        hhInFlight = true
        let counted = phase == "data"      // armed-screen fails are expected, not errors
        addBacklog(1)
        workQ.async { self.hhDecode(p.f, p.hv, counted); self.addBacklog(-1) }
        return true
    }

    func hhDecode(_ f: [UInt8], _ hv: [Double], _ counted: Bool) {   // workQ
        guard let g = grid, f.count == fw * fh * 4 else {
            lock.lock(); hhResults.append((idx: -1, ok: false)); lock.unlock()
            return          // ALWAYS post a result — a bare return leaks hhInFlight
        }
        let res = aneOK ? aneDecode(f, hv, g) : tryDecode(f, hv, g)
        var ok = false, idx = -1
        if let (i, data) = res {
            idx = i
            if let d = data {
                D[i] = d; ok = true
                lock.lock(); decCnt = D.keys.filter { $0 < Int(plan["FRAMES"] ?? 1e9) }.count
                lock.unlock()
                rtTick(f, hv, i, d, g)
            }
        }
        if counted, !ok { failed += 1 }
        lock.lock(); hhResults.append((idx: idx, ok: ok)); lock.unlock()
    }

    func hhMaybeRelock(_ f: [UInt8]) {     // naming lost: full ring search, throttled —
        if hhMiss > 20, !hhRelockBusy,     // this is also the BOOTSTRAP (dedup never
           Date().timeIntervalSince(hhLastRelock) > 0.7 {   // commits under motion)
            hhRelockBusy = true; hhLastRelock = Date()
            workQ.async { self.hhRelock(f) }
        }
        // last resort: Z-drift through the f/1.78 depth of field — re-settle AF.
        // Never during train (a 2-3 s AF blur stall would starve the harvest).
        if phase != "train", hhMiss > 120, hhRelockFails >= 2,
           Date().timeIntervalSince(hhLastAF) > 5 {
            hhLastAF = Date(); afKicks += 1
            out("HH: naming collapsed — AF re-settle #\(afKicks)")
            camera.refocus { self.out($0) }
        }
    }

    func hhRelock(_ f: [UInt8]) {          // workQ: search verified by probe CRC only
        var found: [Double]?
        if let g = grid, f.count == fw * fh * 4 {
            for hv in Geometry.findQuad(f, fw, fh, cols: g.cols, rows: g.rows,
                                        px: g.px, C: g.C) {
                if hhProbe(f, hv, g) != nil { found = hv; break }
            }
        }
        lock.lock()
        if let hv = found { hhNewHinv = hv }
        hhRelockOutcome = found != nil
        lock.unlock()
    }

    // ---- v3.1 DMT decode (payload = demapper, header/corners = glyph net) -------------
    func idxFrom49(_ syms: [UInt8]) -> Int? {   // majority + CRC over 7 header copies
        var votes = [Int](repeating: 0, count: 42)
        for b in 0 ..< 294 { votes[b % 42] += Int(syms[b / 6]) >> (5 - b % 6) & 1 }
        var idx = 0, crc = 0
        for j in 0 ..< 30 { idx = idx << 1 | (votes[j] > 3 ? 1 : 0) }
        for j in 30 ..< 42 { crc = crc << 1 | (votes[j] > 3 ? 1 : 0) }
        let be = (0 ..< 4).map { UInt8(idx >> (24 - 8 * $0) & 0xFF) }
        return crc == Int(crc32(be) & 0xFFF) ? idx : nil
    }

    // payload cells as 20x20 canonical crops = the FULL C window (cell + margin),
    // normalized floats in the demapper's (3,20,20) layout
    func gatherCells20(_ f: [UInt8], _ hv: [Double], _ g: Grid) -> [Float] {
        var X = [Float](repeating: 0, count: g.pay.count * 1200)
        let w = fw, h = fh, C = Double(g.C)
        X.withUnsafeMutableBufferPointer { xp in
            f.withUnsafeBufferPointer { sp in
                DispatchQueue.concurrentPerform(iterations: 8) { k in
                    let per = (g.pay.count + 7) / 8
                    for i in k * per ..< min((k + 1) * per, g.pay.count) {
                        let t = g.pay[i]
                        let r = t / g.cols, c = t % g.cols
                        for pix in 0 ..< 400 {
                            let y = pix / 20, x = pix % 20
                            let dx = Double(c) * C + Double(x)
                            let dy = Double(r) * C + Double(y)
                            let q = hv[6] * dx + hv[7] * dy + hv[8]
                            let sx = (hv[0] * dx + hv[1] * dy + hv[2]) / q
                            let sy = (hv[3] * dx + hv[4] * dy + hv[5]) / q
                            let o = i * 1200 + pix
                            if !(sx >= 0 && sy >= 0 && sx + 1 < Double(w) && sy + 1 < Double(h)) {
                                xp[o] = -0.5; xp[o + 400] = -0.5; xp[o + 800] = -0.5
                                continue
                            }
                            let x0 = Int(sx), y0 = Int(sy)
                            let fx = Float(sx) - Float(x0), fy = Float(sy) - Float(y0)
                            let b = (y0 * w + x0) * 4, b2 = b + w * 4
                            for ch in 0 ..< 3 {
                                let sc = 2 - ch
                                let v = Float(sp[b + sc]) * (1 - fx) * (1 - fy)
                                      + Float(sp[b + 4 + sc]) * fx * (1 - fy)
                                      + Float(sp[b2 + sc]) * (1 - fx) * fy
                                      + Float(sp[b2 + 4 + sc]) * fx * fy
                                xp[o + ch * 400] = max(0, min(255, v.rounded())) / 255 - 0.5
                            }
                        }
                    }
                }
            }
        }
        return X
    }

    func v3Decode(_ f: [UInt8], _ hv: [Double], _ g: Grid) -> (Int, [UInt8]?)? {
        guard f.count == fw * fh * 4, let dm = demap, dm.nc == g.vnc else { return nil }
        var t = Date()
        var Xh = [Float](repeating: 0, count: 49 * 3 * g.K * g.K)
        gatherTiles(f, hv, g, g.hdr, into: &Xh)         // header stays on the glyph net
        var A = Net.Acts(k: net.k)
        var x = [Float](repeating: 0, count: 3 * g.K * g.K)
        var syms = [UInt8](repeating: 0, count: 49)
        for i in 0 ..< 49 {
            for j in 0 ..< x.count { x[j] = Xh[i * x.count + j] }
            net.forward(x, &A)
            let s = A.s.indices.max { A.s[$0] < A.s[$1] }!
            let c = A.c.indices.max { A.c[$0] < A.c[$1] }!
            syms[i] = UInt8(s << 2 | c)
        }
        guard let idx = idxFrom49(syms) else { return nil }
        let X = gatherCells20(f, hv, g)
        TM["gather"]! += -t.timeIntervalSinceNow; t = Date()
        var bits = [UInt8](repeating: 0, count: g.pay.count * 3 * g.vnc)
        dm.classify(X, g.pay.count, into: &bits)
        TM["nn"]! += -t.timeIntervalSinceNow; t = Date()
        var bytes = [UInt8](repeating: 0, count: bits.count / 8)   // MSB-first bits
        for i in 0 ..< bytes.count {
            var v = 0
            for b in 0 ..< 8 { v = v << 1 | Int(bits[i * 8 + b]) }
            bytes[i] = UInt8(v)
        }
        var data = [UInt8](); data.reserveCapacity(g.kpf)
        var allOk = true
        for b in 0 ..< g.nblk {                 // same w-major/block-minor interleave
            let block = (0 ..< 255).map { bytes[$0 * g.nblk + b] }
            let (dec, ok) = RS.decode(block, g.nsym)
            if !ok { allOk = false; break }
            data += dec
        }
        TM["rs"]! += -t.timeIntervalSinceNow; MN += 1
        return (idx, allOk ? data : nil)
    }

    // v3.1 model files: one set per px KEY x NC, own namespace ("v3_"); factories fall
    // back to the bundled demap16_nc{N} — which the canonical window makes valid for
    // scaled keys (1216/3016) too
    func v3ActiveURL(_ px: Int, _ nc: Int) -> URL {
        Self.docs.appendingPathComponent("v3_active_model_\(px)_nc\(nc).bin")
    }
    func v3ForkURL(_ px: Int, _ nc: Int) -> URL {
        Self.docs.appendingPathComponent("v3_model_fork_\(px)_nc\(nc).bin")
    }
    func v3FactoryURL(_ px: Int, _ nc: Int) -> URL {
        Self.docs.appendingPathComponent("v3_factory_model_\(px)_nc\(nc).bin")
    }
    func demapFactoryData(_ px: Int, _ nc: Int) -> Data? {
        if let d = try? Data(contentsOf: v3FactoryURL(px, nc)) { return d }
        return Bundle.main.url(forResource: "demap16_nc\(nc)", withExtension: "bin")
            .flatMap { try? Data(contentsOf: $0) }
    }
    func loadDemap(px: Int, nc: Int) {
        curDemapKey = px; curDemapNC = nc
        if let d = try? Data(contentsOf: v3ActiveURL(px, nc)), let dm = Demap(d, nc: nc) {
            demap = dm
            out("demapper r\(pxLabel(px)) nc\(nc): saved tune (active)")
        } else if let d = demapFactoryData(px, nc), let dm = Demap(d, nc: nc) {
            demap = dm
            let user = FileManager.default.fileExists(atPath: v3FactoryURL(px, nc).path)
            out("demapper r\(pxLabel(px)) nc\(nc): factory "
                + (user ? "(user-overwritten)" : "(bundle, PC-trained)"))
        } else {
            demap = nil
            out("demapper nc\(nc): NO MODEL — v3 decode disabled")
        }
    }
    func v3SaveFork(px: Int, nc: Int, alsoFactory: Bool) {
        guard let dm = demap, curDemapKey == px, curDemapNC == nc else {
            out("save v3 fork r\(pxLabel(px)) nc\(nc): loaded demapper is "
                + "r\(pxLabel(curDemapKey)) nc\(curDemapNC) — run that combo first")
            return
        }
        let d = dm.data()
        try? d.write(to: v3ForkURL(px, nc))
        if alsoFactory { try? d.write(to: v3FactoryURL(px, nc)) }
        out("v3 fork r\(pxLabel(px)) nc\(nc) SAVED (\(d.count) B)"
            + (alsoFactory ? " + FACTORY OVERWRITTEN" : ""))
    }
    func v3LoadFork(px: Int, nc: Int) {
        guard let d = try? Data(contentsOf: v3ForkURL(px, nc)) else {
            out("no v3 fork r\(pxLabel(px)) nc\(nc) saved yet"); return
        }
        try? d.write(to: v3ActiveURL(px, nc))
        if let dm = Demap(d, nc: nc) { demap = dm; curDemapKey = px; curDemapNC = nc }
        out("v3 fork r\(pxLabel(px)) nc\(nc) LOADED -> active")
    }
    func v3Factory(px: Int, nc: Int) {
        try? FileManager.default.removeItem(at: v3ActiveURL(px, nc))
        if let d = demapFactoryData(px, nc), let dm = Demap(d, nc: nc) {
            demap = dm; curDemapKey = px; curDemapNC = nc
        }
        out("demapper r\(pxLabel(px)) nc\(nc) reset to FACTORY")
    }

    // ---- training -------------------------------------------------------------------
    func truthSym(_ idx: Int, _ g: Grid) -> [UInt8] {
        var sym = [UInt8](repeating: 0, count: g.T)
        let be = (0 ..< 4).map { UInt8(idx >> (24 - 8 * $0) & 0xFF) }
        let crc = Int(crc32(be) & 0xFFF)
        var hb = [Int]()
        for j in 0 ..< 30 { hb.append(idx >> (29 - j) & 1) }
        for j in 0 ..< 12 { hb.append(crc >> (11 - j) & 1) }
        for k in 0 ..< 49 {
            var v = 0
            for b in 0 ..< 6 { v = v << 1 | hb[(k * 6 + b) % 42] }
            sym[g.hdr[k]] = UInt8(v)
        }
        var rng = PCG64(Self.TRAIN_SEED + idx)
        let pay = rng.integers(64, g.pay.count)
        for (i, p) in g.pay.enumerated() { sym[p] = pay[i] }
        let sync: [UInt8] = [0 << 2 | 0, 1 << 2 | 1, 8 << 2 | 2, 9 << 2 | 3]
        for (ci, s) in zip(g.corners, sync) { sym[ci] = s }
        return sym
    }

    func trainFrame(_ f: [UInt8]) {
        guard let g = grid, !trainingOver, f.count == fw * fh * 4 else { return }
        let cel = g.C * g.C * 3              // crop bytes; px-dependent (r16 1200, r12 768)
        if RTX.count != 20000 * cel {
            RTX = [UInt8](repeating: 0, count: 20000 * cel)
            RTy = [UInt8](repeating: 0, count: 20000 * 2)
            RVX = [UInt8](repeating: 0, count: 3000 * cel)
            RVy = [UInt8](repeating: 0, count: 3000 * 2)
        }
        var crops: [UInt8]?
        var hidx: Int?
        if let hv = Hinv {                     // fast path: cached geometry, CRC verifies
            let cand = gatherCrops20(f, hv, g)
            if let i = headerIdx(classifyCrops(cand, g), g) { crops = cand; hidx = i }
        }
        if hidx == nil, Hinv == nil || handheld {
            for hv in Geometry.findQuad(f, fw, fh, cols: g.cols, rows: g.rows,
                                        px: g.px, C: g.C) {
                let cand = gatherCrops20(f, hv, g)
                if let i = headerIdx(classifyCrops(cand, g), g) {
                    if Hinv == nil { out("geometry locked (training)") } else { relocks += 1 }
                    Hinv = hv; crops = cand; hidx = i
                    if handheld { lock.lock(); hhNewHinv = hv; lock.unlock() }  // seed HH
                    break
                }
            }
        }
        guard let cr = crops, let idx = hidx else {
            trainFails += 1                     // ALWAYS visible: a silently-stuck TRAIN
            if trainFails <= 3 || trainFails % 25 == 0 {
                out("train geom/hdr fail #\(trainFails): " + Geometry.lastInfo)
            }
            return
        }
        if idx == 0 { return }        // frame 0 = the idle/QR screens: the QR overlay
                                      // covers center tiles, truth would be POISON
        if debug, trn < 3 { out("dbg train frame idx \(idx) harvested") }
        let truth = truthSym(idx, g)
        for t in (0 ..< g.T).shuffled().prefix(1200) {
            if t % 10 == 0 {
                let j = rvn < 3000 ? rvn : Int.random(in: 0 ..< 3000)
                for k in 0 ..< cel { RVX[j * cel + k] = cr[t * cel + k] }
                RVy[j * 2] = truth[t] >> 2; RVy[j * 2 + 1] = truth[t] & 3
                rvn = min(rvn + 1, 3000)
            } else {
                let j = rtn < 20000 ? rtn : Int.random(in: 0 ..< 20000)
                for k in 0 ..< cel { RTX[j * cel + k] = cr[t * cel + k] }
                RTy[j * 2] = truth[t] >> 2; RTy[j * 2 + 1] = truth[t] & 3
                rtn = min(rtn + 1, 20000)
            }
        }
        trn += 1
        sinceChunk += 1
        if gatherT0 == nil { gatherT0 = Date() }
        maybeChunk()
    }

    // fire a chunk when a batch is ready AND no chunk is running. Chunks train a CLONE
    // on their own thread, so harvest keeps running here at full rate — the next batch
    // accumulates DURING training and starts the moment the swap lands (no gather gap).
    func maybeChunk() {
        guard !trainingOver, !chunkBusy, rtn > 2000, let g0 = gatherT0 else { return }
        let ref = lastChunkT ?? g0
        guard sinceChunk >= Self.TRAIN_EVERY
              || (sinceChunk >= 4 && Date().timeIntervalSince(ref) > 20) else { return }
        let batch = sinceChunk
        sinceChunk = 0; lastChunkT = Date()
        chunkBusy = true
        let ncpu = ProcessInfo.processInfo.activeProcessorCount
        let cand = ncpu >= 3 ? [ncpu - 2, 2, 1] : [1]
        let nth: Int
        if let untested = cand.first(where: { chunkNth[$0] == nil }) {
            nth = untested
        } else {          // fastest wins, but fewer threads take it when within 10%
            let best = chunkNth.values.min()!    // (cooler for free; adapts if thermal
            nth = chunkNth.filter { $0.value <= best * 1.1 }.keys.min()!   // state shifts)
        }
        let clone = Net(netData(), k: net.k)   // harvest keeps the live net for headers
        // .userInteractive: during TRAIN nothing else needs the cores (decode idle,
        // harvest gated) — don't let camera-side work preempt the chunk
        DispatchQueue.global(qos: .userInteractive).async { self.runChunk(clone, batch, nth) }
    }

    func classifyCrops(_ crops: [UInt8], _ g: Grid) -> [UInt8] {
        let K = g.K, C = g.C, cn = C * C, n1 = K * K
        var X = [Float](repeating: 0, count: g.T * 3 * n1)
        for t in 0 ..< g.T {
            for ch in 0 ..< 3 { for y in 0 ..< K { for x in 0 ..< K {
                let v = Float(crops[(t * cn + (y + 2) * C + x + 2) * 3 + ch])
                X[t * 3 * n1 + (ch * K + y) * K + x] = v / 255 - 0.5
            } } }
        }
        return classify(X, g.T)
    }

    // the chunk itself, OFF the worker: batch split over all-but-two cores, per-thread
    // gradients summed for one Adam step. Reservoir rows may be overwritten by harvest
    // mid-read (same benign race the Surface's runtime_tune accepts — one torn sample
    // in ~30k). Aborts on START without swapping — the live net never sees half a chunk.
    func runChunk(_ clone: Net, _ batch: Int, _ nth: Int) {
        let t0 = Date()
        let K = clone.k, C = K + 4, cn = C * C
        let e = terr
        let opt = Adam(clone.counts)
        opt.lr = e == nil || e! >= 0.03 ? 1e-4 : e! >= 0.01 ? 5e-5 : e! >= 0.005 ? 2e-5 : 1e-5
        let per = (Self.TRAIN_BS + nth - 1) / nth
        let g = Net.Grads(clone.counts)
        let tg = (0 ..< nth).map { _ in Net.Grads(clone.counts) }
        for _ in 0 ..< Self.TRAIN_STEPS {
            if trainingOver { workQ.async { self.chunkBusy = false }; return }
            DispatchQueue.concurrentPerform(iterations: nth) { ti in
                let gl = tg[ti]
                gl.zero()
                var A = Net.Acts(k: K)
                var x = [Float](repeating: 0, count: 3 * K * K)
                for _ in 0 ..< max(0, min(per, Self.TRAIN_BS - ti * per)) {
                    let i = Int.random(in: 0 ..< self.rtn)
                    let ox = Int.random(in: 0 ... 4), oy = Int.random(in: 0 ... 4)
                    var gi = Int.random(in: 0 ..< Self.gaussPool.count)
                    for ch in 0 ..< 3 {
                        let br = Float.random(in: 0.7 ... 1.3)
                        for y in 0 ..< K { for xx in 0 ..< K {
                            let v = Float(self.RTX[(i * cn + (y + oy) * C + xx + ox) * 3 + ch])
                            x[(ch * K + y) * K + xx] = (v * br + Self.gaussPool[gi] * 4) / 255 - 0.5
                            gi += 1; if gi == Self.gaussPool.count { gi = 0 }
                        } }
                    }
                    clone.forward(x, &A)
                    clone.backward(&A, Int(self.RTy[i * 2]), Int(self.RTy[i * 2 + 1]), gl)
                }
            }
            g.zero()
            for gl in tg {
                for i in 0 ..< g.t.count {
                    for j in 0 ..< g.t[i].count { g.t[i][j] += gl.t[i][j] }
                }
            }
            opt.step(clone, g, batch: Self.TRAIN_BS)
        }
        var wrong = 0
        var A = Net.Acts(k: K)
        var x = [Float](repeating: 0, count: 3 * K * K)
        let n = min(2048, rvn)
        for k in 0 ..< n {
            let i = k * rvn / n
            for ch in 0 ..< 3 { for y in 0 ..< K { for xx in 0 ..< K {
                let v = Float(RVX[(i * cn + (y + 2) * C + xx + 2) * 3 + ch])
                x[(ch * K + y) * K + xx] = v / 255 - 0.5
            } } }
            clone.forward(x, &A)
            let s = A.s.indices.max { A.s[$0] < A.s[$1] }!
            let c = A.c.indices.max { A.c[$0] < A.c[$1] }!
            if UInt8(s) != RVy[i * 2] || UInt8(c) != RVy[i * 2 + 1] { wrong += 1 }
        }
        let newErr = Double(wrong) / Double(max(n, 1))
        let dt = Date().timeIntervalSince(t0)
        workQ.async {
            self.chunkBusy = false
            self.chunkNth[nth] = min(self.chunkNth[nth] ?? .infinity, dt)   // A/B record
            guard !self.trainingOver else { return }   // late abort: discard the clone
            self.net = clone
            self.terr = newErr
            self.out(String(format: "TRAIN frames=%d err=%.2f%%  (chunk %.1f s, %d threads, batch %d frames)",
                            self.trn, newErr * 100, dt, nth, batch))
            self.maybeChunk()   // the batch gathered DURING training starts instantly
        }
    }

    // ---- runtime tuning (data phase) --------------------------------------------------
    // truth for a DECODED data frame: the payload is known post-RS, so every tile's
    // symbol is known — re-encode + re-interleave exactly like the transmitter
    func dataTruthSym(_ idx: Int, _ payload: [UInt8], _ g: Grid) -> [UInt8] {
        var sym = truthSym(idx, g)          // header + corners right; pay overwritten
        var wire = [UInt8](repeating: 0, count: g.nblk * 255)
        let K = 255 - g.nsym
        for b in 0 ..< g.nblk {
            let block = RS.encode(Array(payload[b * K ..< (b + 1) * K]), g.nsym)
            for w in 0 ..< 255 { wire[w * g.nblk + b] = block[w] }
        }
        var buf = 0, bits = 0, wi = 0
        for k in 0 ..< g.pay.count {        // MSB-first bits, 6-bit groups, zero tail
            if bits < 6 {
                buf = buf << 8 | Int(wi < wire.count ? wire[wi] : 0)
                wi += 1; bits += 8
            }
            sym[g.pay[k]] = UInt8(buf >> (bits - 6) & 63)
            bits -= 6
        }
        return sym
    }

    func rtTick(_ f: [UInt8], _ hv: [Double], _ idx: Int, _ d: [UInt8], _ g: Grid) {
        guard rtEvery > 0, !rtBusy, phase == "data" else { return }
        rtCount += 1
        if rtCount > rtEvery - rtTake { runtimeHarvest(f, hv, idx, d, g) }
        if rtCount >= rtEvery, rtn > 500 {
            rtCount = 0
            rtBusy = true
            runtimeTune()
        }
    }

    // sampled harvest on the workQ: 1200 random tiles' C×C crops (~10-15 ms — a full
    // gatherCrops20 of a 4K web grid would eat 2-3 frame slots)
    func runtimeHarvest(_ f: [UInt8], _ hv: [Double], _ idx: Int, _ d: [UInt8], _ g: Grid) {
        guard f.count == fw * fh * 4 else { return }
        let cel = g.C * g.C * 3
        if RTX.count != 20000 * cel {       // no training phase ran: start fresh
            RTX = [UInt8](repeating: 0, count: 20000 * cel)
            RTy = [UInt8](repeating: 0, count: 20000 * 2)
            RVX = [UInt8](repeating: 0, count: 3000 * cel)
            RVy = [UInt8](repeating: 0, count: 3000 * 2)
            rtn = 0; rvn = 0
        }
        let truth = dataTruthSym(idx, d, g)
        let C = g.C, Cd = Double(g.C), w = fw, h = fh
        var crop = [UInt8](repeating: 0, count: cel)
        f.withUnsafeBufferPointer { sp in
            for t in (0 ..< g.T).shuffled().prefix(1200) {
                let r = t / g.cols, c = t % g.cols
                for pix in 0 ..< C * C {
                    let y = pix / C, x = pix % C
                    let dx = Double(c) * Cd + Double(x), dy = Double(r) * Cd + Double(y)
                    let q = hv[6] * dx + hv[7] * dy + hv[8]
                    let sx = (hv[0] * dx + hv[1] * dy + hv[2]) / q
                    let sy = (hv[3] * dx + hv[4] * dy + hv[5]) / q
                    let o = pix * 3
                    if !(sx >= 0 && sy >= 0 && sx + 1 < Double(w) && sy + 1 < Double(h)) {
                        crop[o] = 0; crop[o + 1] = 0; crop[o + 2] = 0
                        continue
                    }
                    let x0 = Int(sx), y0 = Int(sy)
                    let fx = Float(sx) - Float(x0), fy = Float(sy) - Float(y0)
                    let b = (y0 * w + x0) * 4, b2 = b + w * 4
                    for ch in 0 ..< 3 {
                        let sc = 2 - ch
                        let v = Float(sp[b + sc]) * (1 - fx) * (1 - fy)
                              + Float(sp[b + 4 + sc]) * fx * (1 - fy)
                              + Float(sp[b2 + sc]) * (1 - fx) * fy
                              + Float(sp[b2 + 4 + sc]) * fx * fy
                        crop[o + ch] = UInt8(max(0, min(255, v.rounded())))
                    }
                }
                if t % 10 == 0 {
                    let j = rvn < 3000 ? rvn : Int.random(in: 0 ..< 3000)
                    for k in 0 ..< cel { RVX[j * cel + k] = crop[k] }
                    RVy[j * 2] = truth[t] >> 2; RVy[j * 2 + 1] = truth[t] & 3
                    rvn = min(rvn + 1, 3000)
                } else {
                    let j = rtn < 20000 ? rtn : Int.random(in: 0 ..< 20000)
                    for k in 0 ..< cel { RTX[j * cel + k] = crop[k] }
                    RTy[j * 2] = truth[t] >> 2; RTy[j * 2 + 1] = truth[t] & 3
                    rtn = min(rtn + 1, 20000)
                }
            }
        }
    }

    // ONE thread at .utility: decode keeps the P-cores, the tune rides what's left —
    // drift moves over tens of seconds, a slow chunk is fine. Clone-train-swap like
    // runChunk (which can't be reused here: its abort flag trainingOver is TRUE in
    // data phase by design). The swap also re-patches the ANE (~0.3 s on the workQ —
    // a few frames lean on the backlog/spill, then decode runs on FRESH weights).
    func runtimeTune() {
        let clone = Net(netData(), k: net.k)
        rtTunes += 1
        let tuneN = rtTunes
        DispatchQueue.global(qos: .utility).async {
            let t0 = Date()
            let K = clone.k, C = K + 4, cn = C * C
            let e = self.terr
            let opt = Adam(clone.counts)
            opt.lr = e == nil || e! >= 0.03 ? 1e-4 : e! >= 0.01 ? 5e-5
                   : e! >= 0.005 ? 2e-5 : 1e-5
            let g = Net.Grads(clone.counts)
            var A = Net.Acts(k: K)
            var x = [Float](repeating: 0, count: 3 * K * K)
            for _ in 0 ..< Self.TRAIN_STEPS {
                if self.finished { self.workQ.async { self.rtBusy = false }; return }
                g.zero()
                for _ in 0 ..< Self.TRAIN_BS {
                    let i = Int.random(in: 0 ..< self.rtn)
                    let ox = Int.random(in: 0 ... 4), oy = Int.random(in: 0 ... 4)
                    var gi = Int.random(in: 0 ..< Self.gaussPool.count)
                    for ch in 0 ..< 3 {
                        let br = Float.random(in: 0.7 ... 1.3)
                        for y in 0 ..< K { for xx in 0 ..< K {
                            let v = Float(self.RTX[(i * cn + (y + oy) * C + xx + ox) * 3 + ch])
                            x[(ch * K + y) * K + xx] = (v * br + Self.gaussPool[gi] * 4) / 255 - 0.5
                            gi += 1; if gi == Self.gaussPool.count { gi = 0 }
                        } }
                    }
                    clone.forward(x, &A)
                    clone.backward(&A, Int(self.RTy[i * 2]), Int(self.RTy[i * 2 + 1]), g)
                }
                opt.step(clone, g, batch: Self.TRAIN_BS)
            }
            var wrong = 0
            let n = min(2048, self.rvn)
            for k in 0 ..< n {
                let i = k * self.rvn / n
                for ch in 0 ..< 3 { for y in 0 ..< K { for xx in 0 ..< K {
                    let v = Float(self.RVX[(i * cn + (y + 2) * C + xx + 2) * 3 + ch])
                    x[(ch * K + y) * K + xx] = v / 255 - 0.5
                } } }
                clone.forward(x, &A)
                let s = A.s.indices.max { A.s[$0] < A.s[$1] }!
                let c = A.c.indices.max { A.c[$0] < A.c[$1] }!
                if UInt8(s) != self.RVy[i * 2] || UInt8(c) != self.RVy[i * 2 + 1] { wrong += 1 }
            }
            let newErr = Double(wrong) / Double(max(n, 1))
            let dt = Date().timeIntervalSince(t0)
            self.workQ.async {
                defer { self.rtBusy = false }   // counting restarts AFTER completion
                guard !self.finished, self.phase == "data" else { return }
                self.net = clone
                self.terr = newErr
                if self.aneOK, let a = self.ane, a.rebuild(clone) { self.aneChecked = false }
                self.out(String(format: "RUNTIME tune #%d: err %.2f%%  (%.1f s, 1 thread)",
                                tuneN, newErr * 100, dt))
            }
        }
    }

    // ---- model files, ONE SET PER PX (12/16) x NAMESPACE (stable/hh): active (what
    // sessions start from), fork (the user's snapshot), factory (bundle default,
    // user-overwritable). The "hh" namespace keeps handheld training completely away
    // from the stable tunes: hh files carry an hh_ prefix and hh sessions warm-start
    // from the best stable weights when no hh file exists yet (handheld is
    // warm-start-only — a cold model can't read headers under handheld glare/noise).
    static let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
    var curPx = 16                         // model KEY of the loaded net: a plain px, or
                                           // px*100+16 for canonical-16 ("12c16"=1216,
                                           // "30c16"=3016 — the r16 net shape either way)
    var curSet = "stable"                  // model NAMESPACE of the loaded net
    func wantSet() -> String { handheld ? "hh" : "stable" }
    func setTag(_ set: String) -> String { set == "hh" ? " [HH]" : "" }
    func netK(_ key: Int) -> Int { key > 100 ? 16 : key }
    func pxLabel(_ key: Int) -> String { key > 100 ? "\(key / 100)c16" : "\(key)" }
    func gridKey(_ g: Grid) -> Int { g.scaled ? g.px * 100 + 16 : g.px }
    func activeURL(_ px: Int, _ set: String = "stable") -> URL {
        Self.docs.appendingPathComponent((set == "hh" ? "hh_" : "") + "active_model_\(px).bin")
    }
    func forkURL(_ px: Int, _ set: String = "stable") -> URL {
        Self.docs.appendingPathComponent((set == "hh" ? "hh_" : "") + "model_fork_\(px).bin")
    }
    func factoryURL(_ px: Int, _ set: String = "stable") -> URL {
        Self.docs.appendingPathComponent((set == "hh" ? "hh_" : "") + "factory_model_\(px).bin")
    }

    func bundleWeights(_ px: Int) -> Data {
        try! Data(contentsOf: Bundle.main.url(forResource: px == 12 ? "weights12" : "weights",
                                              withExtension: "bin")!)
    }

    func factoryData(_ px: Int, _ set: String = "stable") -> Data {
        if let d = try? Data(contentsOf: factoryURL(px, set)) { return d }
        if set == "hh" {                   // hh warm-start chain: best stable tune first
            if let d = try? Data(contentsOf: activeURL(px)) { return d }
            return factoryData(px)
        }
        if px > 100 {                      // canonical-16: warm-start from the best r16
            if let d = try? Data(contentsOf: activeURL(16)) { return d }   // weights
            if let d = try? Data(contentsOf: factoryURL(16)) { return d }  // around
            return bundleWeights(16)
        }
        return bundleWeights(px)
    }

    func loadNet(px: Int, set: String = "stable") {   // sessions start from ACTIVE (a
        curPx = px; curSet = set           // loaded fork) or factory; QR px + toggle pick
        if let d = try? Data(contentsOf: activeURL(px, set)), d.count > 0 {
            net = Net(d, k: netK(px))
            out("model r\(pxLabel(px))\(setTag(set)): "
                + activeURL(px, set).lastPathComponent + " (saved tune)")
        } else if set == "hh" {
            net = Net(factoryData(px, set), k: netK(px))
            let fm = FileManager.default
            let src = fm.fileExists(atPath: factoryURL(px, "hh").path) ? "hh factory"
                : fm.fileExists(atPath: activeURL(px).path) ? "stable active (warm start)"
                : "stable factory (warm start)"
            out("model r\(pxLabel(px)) [HH]: \(src)")
        } else {
            let userFactory = FileManager.default.fileExists(atPath: factoryURL(px).path)
            net = Net(factoryData(px), k: netK(px))
            out("model r\(pxLabel(px)): factory " + (userFactory ? "(user-overwritten)"
                : px == 12 ? "loop_L12n (bundle, picture-trained!)"
                : px > 100 ? "(warm-started from r16 weights)" : "loop_L16_best (bundle)"))
        }
    }

    func netData() -> Data {
        var d = Data()
        for kp in Net.keys {
            net[keyPath: kp].withUnsafeBufferPointer { d.append(Data(buffer: $0)) }
        }
        return d
    }

    func saveFork(px: Int, set: String, alsoFactory: Bool) {   // snapshot the CURRENT net
        guard net != nil, curPx == px, curSet == set else {
            out("save fork r\(pxLabel(px))\(setTag(set)): loaded model is "
                + "r\(pxLabel(curPx))\(setTag(curSet)) — run/load that px first")
            return
        }
        let d = netData()
        try? d.write(to: forkURL(px, set))
        if alsoFactory { try? d.write(to: factoryURL(px, set)) }
        out("model fork r\(pxLabel(px))\(setTag(set)) SAVED (\(d.count) B)"
            + (alsoFactory ? " + FACTORY\(setTag(set)) OVERWRITTEN" : ""))
    }

    func loadFork(px: Int, set: String) {  // fork -> active (+ live net if same px+set)
        guard let d = try? Data(contentsOf: forkURL(px, set)) else {
            out("no r\(pxLabel(px))\(setTag(set)) fork saved yet"); return
        }
        try? d.write(to: activeURL(px, set))
        if (curPx == px && curSet == set) || net == nil {
            net = Net(d, k: netK(px)); curPx = px; curSet = set
        }
        out("model fork r\(pxLabel(px))\(setTag(set)) LOADED -> active")
    }

    func factoryModel(px: Int, set: String) {   // back to factory (user or bundle/chain)
        try? FileManager.default.removeItem(at: activeURL(px, set))
        if (curPx == px && curSet == set) || net == nil {
            net = Net(factoryData(px, set), k: netK(px)); curPx = px; curSet = set
        }
        out("model r\(pxLabel(px))\(setTag(set)) reset to FACTORY"
            + (FileManager.default.fileExists(atPath: factoryURL(px, set).path)
               ? " (user-overwritten)" : set == "hh" ? " (stable warm start)" : " (bundle)"))
    }

    func saveTuned() {
        let url = Self.docs.appendingPathComponent(
            "rx_tuned_L\(pxLabel(curPx))\(curSet == "hh" ? "hh" : "").bin")
        let d = netData()
        try? d.write(to: url)
        tuned = true
        out(String(format: "tuned model saved (err %.2f%% on %d frames)",
                   (terr ?? 1) * 100, trn))
        if let e = terr, e < 0.05 {        // a good tune becomes this px+set's ACTIVE:
            try? d.write(to: activeURL(curPx, curSet))   // the next session warm-starts
            out("auto-saved -> " + activeURL(curPx, curSet).lastPathComponent
                + " (err < 5%)")                         // from it automatically
        }
    }

    // ---- end of session --------------------------------------------------------------
    func solveRepair(_ N: Int, _ g: Grid) -> Int {
        let miss = (0 ..< N).filter { D[$0] == nil }.sorted()
        if miss.isEmpty { return 0 }
        let pos = Dictionary(uniqueKeysWithValues: miss.enumerated().map { ($1, $0) })
        let words = (miss.count + 63) / 64
        var piv = [Int: (bm: [UInt64], rhs: [UInt8])]()
        for ri in D.keys.sorted() where ri >= N {
            var rng = PCG64(Self.REPAIR_SEED + ri - N)
            var mask = rng.integers(2, N)
            if !mask.contains(1) { mask[(ri - N) % N] = 1 }
            var bm = [UInt64](repeating: 0, count: words)
            var rhs = D[ri]!
            for k in 0 ..< N where mask[k] == 1 {
                if let p = pos[k] { bm[p / 64] |= 1 << (p % 64) }
                else { let dk = D[k]!; for j in 0 ..< rhs.count { rhs[j] ^= dk[j] } }
            }
            while let low = bm.enumerated().first(where: { $0.1 != 0 })
                .map({ $0.0 * 64 + $0.1.trailingZeroBitCount }) {
                if piv[low] == nil { piv[low] = (bm, rhs); break }
                let p = piv[low]!
                for j in 0 ..< words { bm[j] ^= p.bm[j] }
                for j in 0 ..< rhs.count { rhs[j] ^= p.rhs[j] }
            }
        }
        if piv.count < miss.count { return 0 }
        for b in piv.keys.sorted(by: >) {
            var (bm, rhs) = piv[b]!
            for b2 in b + 1 ..< miss.count where bm[b2 / 64] >> (b2 % 64) & 1 == 1 {
                if let p2 = piv[b2] { for j in 0 ..< rhs.count { rhs[j] ^= p2.rhs[j] } }
            }
            bm = [UInt64](repeating: 0, count: words)
            bm[b / 64] = 1 << (b % 64)
            piv[b] = (bm, rhs)
            D[miss[b]] = rhs
        }
        return miss.count
    }

    func finish(_ why: String) {
        guard !finished else { return }
        if hhActive, let p = hhPend {      // the last dwell's best shot is still pending
            hhPend = nil
            hhFlush(p, force: true)        // serial workQ: decodes before assemble runs
        }
        finished = true
        phase = "done"
        out("\n\(why) — draining backlog...")
        workQ.async { self.assemble() }
    }

    func assemble() {
        camera.stop()
        guard let g = grid, let fr = plan["FRAMES"] else {
            out("no plan/grid — nothing received"); DispatchQueue.main.async { self.running = false }
            return
        }
        if trn > 0, !tuned { saveTuned() }
        drainSpill()
        let N = Int(fr)
        let direct = D.keys.filter { $0 < N }.count
        let rebuilt = solveRepair(N, g)
        let missing = (0 ..< N).filter { D[$0] == nil }
        var outBytes = [UInt8](repeating: 0, count: N * g.kpf)
        let scrambled = (plan["SCR"] ?? 0) == 1   // SCR=1: source rows are whitened on
        for (i, d) in D where i < N {             // the wire (content-shaped white
            if scrambled {                        // floods bloomed the camera) —
                var rng = PCG64(Self.SCRAMBLE_SEED + i)   // unscramble at assembly only
                let ks = rng.integers(256, g.kpf)
                for j in 0 ..< g.kpf { outBytes[i * g.kpf + j] = d[j] ^ ks[j] }
            } else {
                for j in 0 ..< g.kpf { outBytes[i * g.kpf + j] = d[j] }
            }
        }
        let size = Int(plan["SIZE"] ?? Double(outBytes.count))
        outBytes = Array(outBytes[0 ..< min(size, outBytes.count)])
        let url = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("received_payload.bin")
        try? Data(outBytes).write(to: url)
        let md5 = Insecure.MD5.hash(data: Data(outBytes))
            .map { String(format: "%02x", $0) }.joined()
        let repDec = D.keys.filter { $0 >= N }.count
        out("\ndecoded \(direct)/\(N) + \(rebuilt) rebuilt (failed \(failed), dropped \(dropped), "
            + "repair \(repDec)/\(Int(plan["REPAIR"] ?? 0)) decoded)")
        if relocks > 0 { out("geometry re-locks: \(relocks) (shared path)") }
        if handheld, hhFramesN > 0 {
            out("HH: named \(hhNamed)/\(hhFramesN) frames  best-shot swaps \(hhSwaps)  "
                + "relocks \(relockShift) shift / \(relockFull) full  af \(afKicks)  "
                + "retries \(hhRetries)")
        }
        if !pool.isEmpty {
            out("pool size -> ok/failed: " + pool.sorted { $0.key < $1.key }
                .map { "\($0.key):\($0.value[0])/\($0.value[1])" }.joined(separator: "  "))
        }
        out("MISSING: " + (missing.isEmpty ? "none" : ranges(missing)))
        out("received_payload.bin  \(outBytes.count) bytes  md5 \(md5)")
        if let name = fileName {
            if missing.isEmpty {         // complete = RS-proven correct (all-or-nothing
                let dir = Self.docs.appendingPathComponent("QR Transmit", isDirectory: true)
                try? FileManager.default.createDirectory(at: dir,
                                                         withIntermediateDirectories: true)
                let safe = name.components(separatedBy: "/").last ?? name
                let fu = dir.appendingPathComponent(safe)
                try? FileManager.default.removeItem(at: fu)          // save-or-replace
                if (try? Data(outBytes).write(to: fu)) != nil {
                    out("file saved: QR Transmit/\(safe)  (Files app > On My iPhone > "
                        + "QR File Transmit > QR Transmit)")
                } else { out("file save FAILED: \(safe)") }
            } else {
                out("file NOT saved (\(missing.count) frames missing): \(name)")
            }
        }
        if let t = t0 {
            out(String(format: "goodput %.2f KB/s (%d B in %.1f s)",
                       Double(outBytes.count) / 1024 / Date().timeIntervalSince(t),
                       outBytes.count, Date().timeIntervalSince(t)))
        }
        if MN > 0 {
            out(String(format: "per-frame: gather %.0f  nn %.0f  rs %.0f ms over %d attempts",
                       TM["gather"]! / Double(MN) * 1000, TM["nn"]! / Double(MN) * 1000,
                       TM["rs"]! / Double(MN) * 1000, MN))
        }
        if MNA > 0 {
            out(String(format: "ane stages: upload %.1f  metal %.1f  predict %.1f ms over %d frames",
                       TMA["up"]! / Double(MNA) * 1000, TMA["mtl"]! / Double(MNA) * 1000,
                       TMA["pred"]! / Double(MNA) * 1000, MNA))
        }
        DispatchQueue.main.async { self.running = false }
        statTick()
    }

    func ranges(_ idxs: [Int]) -> String {
        var out = [[Int]]()
        for m in idxs.sorted() {
            if var last = out.last, m == last[1] + 1 { last[1] = m; out[out.count - 1] = last }
            else { out.append([m, m]) }
        }
        return out.map { $0[0] == $0[1] ? "\($0[0])" : "\($0[0])-\($0[1])" }
            .joined(separator: ", ")
    }

    // sharpness gauge (variance of Laplacian on the center window, green channel) —
    // the preview is 1/4-res and always looks soft; THIS is the focus truth
    func lapvar(_ f: [UInt8], _ w: Int, _ h: Int) -> Double {
        var n = 0, s = 0.0, s2 = 0.0
        f.withUnsafeBufferPointer { p in
            for y in stride(from: h / 3, to: 2 * h / 3, by: 2) {
                for x in stride(from: w / 3, to: 2 * w / 3, by: 2) {
                    let o = (y * w + x) * 4 + 1
                    let lap = 4 * Int(p[o]) - Int(p[o - 4]) - Int(p[o + 4])
                            - Int(p[o - w * 4]) - Int(p[o + w * 4])
                    let d = Double(lap)
                    s += d; s2 += d * d; n += 1
                }
            }
        }
        let m = s / Double(n)
        return s2 / Double(n) - m * m
    }

    func probeID(_ f: [UInt8], _ hv: [Double], _ g: Grid) {   // 49 header tiles only
        let n = 3 * g.K * g.K
        var X = [Float](repeating: 0, count: 49 * n)
        gatherTiles(f, hv, g, g.hdr, into: &X)
        var A = Net.Acts(k: net.k)
        var x = [Float](repeating: 0, count: n)
        var syms = [UInt8](repeating: 0, count: 49)
        for t in 0 ..< 49 {
            for j in 0 ..< n { x[j] = X[t * n + j] }
            net.forward(x, &A)
            let s = A.s.indices.max { A.s[$0] < A.s[$1] }!
            let c = A.c.indices.max { A.c[$0] < A.c[$1] }!
            syms[t] = UInt8(s << 2 | c)
        }
        var votes = [Int](repeating: 0, count: 42)
        for b in 0 ..< 294 { votes[b % 42] += Int(syms[b / 6]) >> (5 - b % 6) & 1 }
        var idx = 0, crc = 0
        for j in 0 ..< 30 { idx = idx << 1 | (votes[j] > 3 ? 1 : 0) }
        for j in 30 ..< 42 { crc = crc << 1 | (votes[j] > 3 ? 1 : 0) }
        let be = (0 ..< 4).map { UInt8(idx >> (24 - 8 * $0) & 0xFF) }
        if crc == Int(crc32(be) & 0xFFF) { probeIds[idx, default: 0] += 1; allIds.insert(idx) }
        else { probeNil += 1 }
    }

    func gatherTiles(_ f: [UInt8], _ hv: [Double], _ g: Grid, _ tiles: [Int],
                     into X: inout [Float]) {
        let w = fw, h = fh, C = Double(g.C), K = g.K, n1 = K * K
        for (i, t) in tiles.enumerated() {
            let r = t / g.cols, c = t % g.cols
            for pix in 0 ..< n1 {
                let y = pix / K, x = pix % K
                let dx = Double(c) * C + Double(x) + 2, dy = Double(r) * C + Double(y) + 2
                let q = hv[6] * dx + hv[7] * dy + hv[8]
                let sx = (hv[0] * dx + hv[1] * dy + hv[2]) / q
                let sy = (hv[3] * dx + hv[4] * dy + hv[5]) / q
                let o = i * 3 * n1 + pix
                if !(sx >= 0 && sy >= 0 && sx + 1 < Double(w) && sy + 1 < Double(h)) {
                    X[o] = -0.5; X[o + n1] = -0.5; X[o + 2 * n1] = -0.5; continue
                }
                let x0 = Int(sx), y0 = Int(sy)
                let fx = Float(sx) - Float(x0), fy = Float(sy) - Float(y0)
                let b = (y0 * w + x0) * 4, b2 = b + w * 4
                for ch in 0 ..< 3 {
                    let sc = 2 - ch
                    let v = Float(f[b + sc]) * (1 - fx) * (1 - fy)
                          + Float(f[b + 4 + sc]) * fx * (1 - fy)
                          + Float(f[b2 + sc]) * (1 - fx) * fy
                          + Float(f[b2 + 4 + sc]) * fx * fy
                    X[o + ch * n1] = max(0, min(255, v.rounded())) / 255 - 0.5
                }
            }
        }
    }

    func jpegData(_ f: [UInt8], _ w: Int, _ h: Int) -> Data? {
        var copy = f
        let info = CGBitmapInfo.byteOrder32Little.rawValue
                 | CGImageAlphaInfo.noneSkipFirst.rawValue
        guard let ctx = CGContext(data: &copy, width: w, height: h, bitsPerComponent: 8,
                                  bytesPerRow: w * 4, space: CGColorSpaceCreateDeviceRGB(),
                                  bitmapInfo: info),
              let img = ctx.makeImage() else { return nil }
        return UIImage(cgImage: img).jpegData(compressionQuality: 0.85)
    }

    func jpegFrame(_ d: Data) -> [UInt8]? {      // spill decode: back to BGRA bytes
        guard let ui = UIImage(data: d), let cg = ui.cgImage else { return nil }
        let w = cg.width, h = cg.height
        var buf = [UInt8](repeating: 0, count: w * h * 4)
        let info = CGBitmapInfo.byteOrder32Little.rawValue
                 | CGImageAlphaInfo.noneSkipFirst.rawValue
        guard let ctx = CGContext(data: &buf, width: w, height: h, bitsPerComponent: 8,
                                  bytesPerRow: w * 4, space: CGColorSpaceCreateDeviceRGB(),
                                  bitmapInfo: info) else { return nil }
        ctx.draw(cg, in: CGRect(x: 0, y: 0, width: w, height: h))
        return buf
    }

    func writeJpg(_ f: [UInt8], _ w: Int, _ h: Int, _ name: String) {
        let url = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent(name)
        try? jpegData(f, w, h)?.write(to: url)
    }

    func dumpFrame(_ f: [UInt8], _ w: Int, _ h: Int) {   // rotating full-res jpgs for
        writeJpg(f, w, h, "dbg\(dumpN % 3).jpg")          // remote eyeballs
        dumpN += 1
    }

    func buildPreview(_ f: [UInt8], _ w: Int, _ h: Int) {   // ~3 fps; Z = 1:1 center crop
        let zoom = zoomPreview
        let pw = zoom ? 320 : w / 4, ph = zoom ? 320 : h / 4
        var buf = [UInt8](repeating: 0, count: pw * ph * 4)
        f.withUnsafeBufferPointer { p in
            if zoom {
                let x0 = (w - pw) / 2, y0 = (h - ph) / 2
                for y in 0 ..< ph { for x in 0 ..< pw {
                    let s = ((y0 + y) * w + x0 + x) * 4, d = (y * pw + x) * 4
                    buf[d] = p[s]; buf[d + 1] = p[s + 1]; buf[d + 2] = p[s + 2]; buf[d + 3] = 255
                } }
            } else {
                for y in 0 ..< ph { for x in 0 ..< pw {
                    let s = (y * 4 * w + x * 4) * 4, d = (y * pw + x) * 4
                    buf[d] = p[s]; buf[d + 1] = p[s + 1]; buf[d + 2] = p[s + 2]; buf[d + 3] = 255
                } }
            }
        }
        let info = CGBitmapInfo.byteOrder32Little.rawValue
                 | CGImageAlphaInfo.noneSkipFirst.rawValue
        guard let ctx = CGContext(data: &buf, width: pw, height: ph, bitsPerComponent: 8,
                                  bytesPerRow: pw * 4, space: CGColorSpaceCreateDeviceRGB(),
                                  bitmapInfo: info),
              let img = ctx.makeImage() else { return }
        DispatchQueue.main.async { self.previewCG = img }
    }

    func statTick() {
        let N = plan["FRAMES"].map { Int($0) }
        lock.lock(); let src = decCnt; lock.unlock()
        let m: String
        if calibrating { m = "CAL" } else {
            switch phase {
            case "wait": m = running ? "WAITING" : "IDLE"
            case "train": m = trn == 0 ? "TRAIN" : "TRAINING"
            case "armed": m = "ARMED"
            case "data": m = "LIVE"
            default: m = "DONE"
            }
        }
        DispatchQueue.main.async { self.mode = m }
        var lines = ["[\(phase)]  commits \(commits)   backlog \(backlogCount())"
                     + String(format: "   sharp %.0f", sharp)]
        if let n = N { lines.append("decoded \(src)/\(n)" + (failed > 0 ? "   failed \(failed)" : "")) }
        if handheld, hhFramesN > 0 {
            let pct = hhNamed * 100 / max(hhFramesN, 1)
            lines.append("HH named \(pct)%   relocks \(relockShift)/\(relockFull)")
        }
        if trn > 0 {
            let e = terr == nil ? "--" : String(format: "%.2f%%", terr! * 100)
            lines.append("TRAIN frames=\(trn) err=\(e)")
        }
        let s = lines.joined(separator: "\n")
        DispatchQueue.main.async { self.stat = s }
    }

    // ---- camera calibration -----------------------------------------------------------
    // The transmitter's idle screen AND both control-QR screens show train_img(0) — a
    // field whose truth derives from seed 0. So while the transmitter is holding any of
    // those, we can sweep camera settings and score each one by DECODED GLYPHS AGAINST
    // TRUTH (excluding the center box the QR may cover). Decoded glyphs beat lapvar as a
    // metric: lapvar rewards clipping (it once picked the yellow-killing exposure).
    func calibrate() {
        guard grid?.v3 != true, !v3Wire else {
            out("CAL scores glyphs — not available on a v3 session"); return
        }
        guard !calibrating else { out("CAL already running — ignored"); return }
        calibrating = true    // claim BEFORE spawning: CAL button + auto-CAL raced here
                              // once — two sweeps fought the lens, cross-scored each
                              // other's blur, and locked 0.250 (soft). Measured disaster.
        DispatchQueue.global(qos: .userInitiated).async { self.calRun() }
    }

    func calRun() {
        if grid == nil {
            grid = Grid(px: 16, cols: 92, rows: 57, nsym: 32)
            out("CAL: no QR seen yet — assuming the 92x57 rig grid")
        }
        let g = grid!
        if net == nil || curPx != gridKey(g) || curSet != wantSet() {
            loadNet(px: gridKey(g), set: wantSet())
        }
        calibrating = true               // freeze the phase machine for the whole sweep
        defer { calibrating = false }
        let base = camera.lensPosition
        let truth = truthSym(0, g)
        let scoreable = (0 ..< g.T).filter { t in     // skip the QR-covered center box
            let r = t / g.cols, c = t % g.cols
            return !(r > g.rows * 3 / 10 && r < g.rows * 7 / 10
                     && c > g.cols * 3 / 10 && c < g.cols * 7 / 10)
        }
        let target = scoreable.count * 9 / 10   // <90% decoded glyphs = a FAILED sweep
        let deadline = Date().addingTimeInterval(25)   // hard budget: the phase machine
        var aborted = false                            // is frozen — never linger

        var calFrame = [UInt8](), calLap = -1.0   // sharpest frame CAL saw -> cal.jpg

        func score(lens: Float?, tag: String) -> (Int, Double) {   // -1 = no geometry
            if let l = lens { camera.setLens(l) }
            usleep(600_000)
            lock.lock(); let f = lastFrameBuf; lock.unlock()
            if f.isEmpty || f.count != fw * fh * 4 { return (-1, 0) }
            let lap = lapvar(f, fw, fh)
            if lap > calLap { calLap = lap; calFrame = f }
            guard let hv = Hinv ?? Geometry.findQuad(f, fw, fh, cols: g.cols,
                                                     rows: g.rows, px: g.px, C: g.C).first
            else { out("CAL \(tag): no geometry, lapvar \(Int(lap))"); return (-1, lap) }
            let sym = classify(gatherX(f, hv, g), g.T)
            if let hidx = headerIdx(sym, g), hidx != 0 { aborted = true; return (-1, lap) }
            var m = 0
            for t in scoreable where sym[t] == truth[t] { m += 1 }
            out("CAL \(tag): \(m)/\(scoreable.count) glyphs, lapvar \(Int(lap))")
            if Hinv == nil, m > scoreable.count / 2 { Hinv = hv }
            return (m, lap)
        }

        // tier 1: the normal sweep around plausible lens positions
        out("CAL: focus sweep (transmitter must be HOLDING the idle/QR screen)")
        var cands: [Float] = [0.25, 0.32, 0.38, 0.44, 0.5, 0.58, 0.7]
        if base >= 0, !cands.contains(where: { abs($0 - base) < 0.02 }) { cands.insert(base, at: 0) }
        var results = [(Float, Int, Double)]()
        for l in cands {
            if aborted || Date() > deadline { break }
            let (m, lap) = score(lens: l, tag: String(format: "lens %.3f", l))
            results.append((l, m, lap))
        }
        defer {                                    // whatever happens, leave the evidence
            if !calFrame.isEmpty { writeJpg(calFrame, fw, fh, "cal.jpg") }
        }
        var best = results.max { $0.1 < $1.1 }
        // tier 2: sweep failed -> widen to the full lens range
        if !aborted, Date() < deadline, (best?.1 ?? -1) < target {
            out("CAL: best \(max(best?.1 ?? 0, 0))/\(scoreable.count) < 90% — WIDE sweep")
            let tried = Set(results.map { Int($0.0 * 1000) })
            for l in stride(from: 0.05, through: 0.95, by: 0.12).map({ Float($0) })
            where !tried.contains(Int(l * 1000)) {
                if aborted || Date() > deadline { out("CAL: time budget hit"); break }
                let (m, lap) = score(lens: l, tag: String(format: "lens %.3f", l))
                results.append((l, m, lap))
            }
            best = results.max { $0.1 < $1.1 }
        }
        if aborted {
            camera.setLens(base >= 0 ? base : 0.44)
            out("CAL ABORT: header != 0 — the screen is not holding the idle field")
            return
        }
        guard var bl = best else { out("CAL: no frames at all"); return }
        if bl.1 <= 0 {                 // zero glyphs anywhere: still pick SOMETHING —
            bl = results.max { $0.2 < $1.2 }!   // the sharpest frame by lapvar
            camera.setLens(bl.0)
            out(String(format: "CAL: nothing decodable at ANY focus — locking sharpest "
                       + "lens %.3f (lapvar %d); check distance / framing / smudged lens",
                       bl.0, Int(bl.2)))
            return
        }
        camera.setLens(bl.0)
        var bestScore = bl.1
        // tier 3: still under target at the winning lens -> try brightness (ISO x0.6/x1.5)
        if bestScore < target, Date() < deadline {
            let iso0 = camera.currentISO
            var bestISO = iso0
            for sc in [Float(0.6), 1.5] {
                if aborted { break }
                let applied = camera.setISO(iso0 * sc)
                let (m, _) = score(lens: nil, tag: String(format: "ISO %.0f", applied))
                if m > bestScore { bestScore = m; bestISO = applied }
            }
            _ = camera.setISO(bestISO)
            if aborted { camera.setLens(base >= 0 ? base : 0.44); out("CAL ABORT: screen moved"); return }
        }
        out(String(format: "CAL: locked lens %.3f ISO %.0f (%d/%d glyphs = %.1f%% err)%@",
                   bl.0, camera.currentISO, bestScore, scoreable.count,
                   100 - 100 * Double(bestScore) / Double(scoreable.count),
                   bestScore < target ? "  — STILL BELOW TARGET, check distance/framing/lens" : ""))
        out("CAL done — phase machine resumed (start training on the mac NOW)")
    }

    // ---- self-tests (run once from the UI before going live) --------------------------
    func selfTest() {
        var okAll = true
        if let url = Bundle.main.url(forResource: "pcg_ref", withExtension: "json"),
           let d = try? Data(contentsOf: url),
           let j = try? JSONSerialization.jsonObject(with: d) as? [String: [String: [Int]]] {
            var ok = true
            for (seed, ref) in j {
                var rng = PCG64(Int(seed)!)
                let v = rng.integers(64, 6000)
                ok = ok && ref["head"]!.enumerated().allSatisfy { Int(v[$0.0]) == $0.1 }
                ok = ok && ref["mid"]!.enumerated().allSatisfy { Int(v[5000 + $0.0]) == $0.1 }
            }
            out("selftest PCG64 vs numpy: \(ok ? "EXACT" : "FAIL")"); okAll = okAll && ok
        } else { out("selftest PCG64: no reference bundled"); okAll = false }
        let rsOk = RS.selfTest()
        out("selftest RS roundtrip: \(rsOk ? "OK" : "FAIL")"); okAll = okAll && rsOk
        // end-to-end on the bundled Surface frame: geometry -> NN -> header -> RS
        if let mURL = Bundle.main.url(forResource: "meta", withExtension: "json"),
           let meta = try? JSONDecoder().decode(Meta.self, from: Data(contentsOf: mURL)),
           let img = UIImage(data: try! Data(contentsOf: Bundle.main.url(
               forResource: "frame", withExtension: "jpg")!)) {
            net = Net(try! Data(contentsOf: Bundle.main.url(forResource: "weights",
                                                            withExtension: "bin")!))
            curPx = 16; curSet = "selftest"   // sentinel: the next session ALWAYS
                                              // reloads (this bundled net stomped
                                              // whatever tune was in memory)
            let cg = img.cgImage!
            fw = cg.width; fh = cg.height
            var rgba = [UInt8](repeating: 0, count: fw * fh * 4)
            let ctx = CGContext(data: &rgba, width: fw, height: fh, bitsPerComponent: 8,
                                bytesPerRow: fw * 4, space: CGColorSpaceCreateDeviceRGB(),
                                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
            ctx.draw(cg, in: CGRect(x: 0, y: 0, width: fw, height: fh))
            for i in stride(from: 0, to: rgba.count, by: 4) { rgba.swapAt(i, i + 2) }   // -> BGRA
            let g = Grid(px: meta.px, cols: meta.cols, rows: meta.rows, nsym: meta.nsym)
            let t = Date()
            let cands = Geometry.findQuad(rgba, fw, fh, cols: g.cols, rows: g.rows,
                                          px: g.px, C: g.C)
            var hit = false
            for hv in cands {
                if let (idx, data) = tryDecode(rgba, hv, g) {
                    out(String(format: "selftest geometry+decode: idx %d rs %@ in %.0f ms",
                               idx, data != nil ? "OK" : "FAIL",
                               -t.timeIntervalSinceNow * 1000))
                    hit = idx == meta.idx && data != nil
                    break
                }
            }
            if !hit { out("selftest geometry+decode: FAIL (\(cands.count) candidates)") }
            okAll = okAll && hit
        }
        out(okAll ? "ALL SELF-TESTS PASS — ready for a live run\n" : "SELF-TEST FAILURES — fix before trusting a live run\n")
    }
}
