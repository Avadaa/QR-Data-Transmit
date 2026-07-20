// Camera.swift — AVFoundation capture: 1920x1080 BGRA at 30 fps, frames delivered as
// plain byte arrays. Unlike the Surface's MSMF driver, iOS locks 3A honestly: lock3A()
// freezes focus/exposure/WB at their current (settled) values — call it once the
// transmitter's field is on screen, never while aiming at something else.
import AVFoundation
import Foundation
import UIKit

final class Camera: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    let session = AVCaptureSession()

    // capture follows the phone's orientation: held upright -> portrait buffers
    // (1080x1920), horizontal -> landscape. The receiver's geometry search handles any
    // rotation, but portrait aiming at the 1:1 web field reads naturally this way.
    static var lastAngle: CGFloat = 90       // faceUp/faceDown/unknown keep the last
    static func rotAngle() -> CGFloat {
        switch UIDevice.current.orientation {
        case .portrait: lastAngle = 90
        case .portraitUpsideDown: lastAngle = 270
        case .landscapeLeft: lastAngle = 0   // device top LEFT = sensor-native landscape
        case .landscapeRight: lastAngle = 180
        default: break
        }
        return lastAngle
    }
    private var rotObserved = false
    func applyRotation() {
        let a = Camera.rotAngle()
        for out in session.outputs {
            if let c = out.connection(with: .video), c.isVideoRotationAngleSupported(a),
               c.videoRotationAngle != a {
                c.videoRotationAngle = a
            }
        }
    }
    var onFrame: (([UInt8], Int, Int) -> Void)?
    var onPixelBuffer: ((CVPixelBuffer, Int, Int) -> Void)?   // zero-copy delivery
    var zeroCopy = false          // true -> hand out the pool's CVPixelBuffer instead
                                  // of copying 8 MB per frame; the receiver may retain
                                  // AT MOST ONE (pool is ~6 deep — more starves capture)
    private let q = DispatchQueue(label: "camera", qos: .userInteractive)
    private var device: AVCaptureDevice?
    private var configured = false

    func resumeAuto() {                  // fresh session: all 3A back to continuous
        guard let d = device else { return }
        try? d.lockForConfiguration()
        if d.isFocusModeSupported(.continuousAutoFocus) { d.focusMode = .continuousAutoFocus }
        if d.isExposureModeSupported(.continuousAutoExposure) { d.exposureMode = .continuousAutoExposure }
        if d.isWhiteBalanceModeSupported(.continuousAutoWhiteBalance) {
            d.whiteBalanceMode = .continuousAutoWhiteBalance
        }
        d.unlockForConfiguration()
    }

    // the 9 bisected settings (docs/iphone_camera_specs.txt) as Settings toggles;
    // default: ALL 1-9 on (was the measured production set #1+#2+#5+#6)
    static func camOn(_ i: Int) -> Bool {
        (UserDefaults.standard.object(forKey: "cam\(i)") as? Bool) ?? true
    }
    // handheld overrides (docs/iphone_camera_handheld.txt): blur is linear in exposure,
    // so the short shutter is the whole game; 4K doubles readout = jello, so 1080p wins
    static func hhOn() -> Bool { (UserDefaults.standard.object(forKey: "handheld") as? Bool) ?? false }
    static func hhBool(_ k: String, _ def: Bool) -> Bool {
        (UserDefaults.standard.object(forKey: k) as? Bool) ?? def
    }
    static func hhShutterDen() -> Int {
        (UserDefaults.standard.object(forKey: "hhShutter") as? Int) ?? 500
    }
    static func settingsSig() -> String {
        (1 ... 9).map { camOn($0) ? "1" : "0" }.joined()
            + (hhOn() ? "-HH\(hhBool("hh1080", false) ? 1 : 0)\(hhBool("hhShortShutter", true) ? 1 : 0)-\(hhShutterDen())" : "")
    }
    private var cfgSig = ""

    func start() -> String? {
        if configured, Camera.settingsSig() != cfgSig {   // toggles changed -> rebuild
            session.inputs.forEach { session.removeInput($0) }
            session.outputs.forEach { session.removeOutput($0) }
            configured = false
        }
        if configured {                  // restart: the session keeps its input/output —
            resumeAuto()                 // re-adding them is what used to error out
            applyRotation()              // orientation may have changed since last run
            DispatchQueue.global().async { self.session.startRunning() }
            return nil
        }
        cfgSig = Camera.settingsSig()
        guard let dev = AVCaptureDevice.default(.builtInWideAngleCamera,
                                                for: .video, position: .back),
              let inp = try? AVCaptureDeviceInput(device: dev) else { return "no camera" }
        device = dev
        let hh = Camera.hhOn()
        let use1 = Camera.camOn(1) || hh         // handheld forces explicit format + 60
        session.beginConfiguration()
        // #1: explicit format via .inputPriority (also implicitly kills preset-videoHDR)
        session.sessionPreset = use1 ? .inputPriority : .hd1920x1080
        guard session.canAddInput(inp) else { return "input rejected" }
        session.addInput(inp)
        let out = AVCaptureVideoDataOutput()
        out.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String:
                             kCVPixelFormatType_32BGRA]
        out.alwaysDiscardsLateVideoFrames = true
        out.setSampleBufferDelegate(self, queue: q)
        guard session.canAddOutput(out) else { return "output rejected" }
        session.addOutput(out)
        session.commitConfiguration()
        try? dev.lockForConfiguration()
        if use1 {
            // #9: 4K — but handheld's hh1080 override wins (4K doubles readout = jello)
            let want4K = Camera.camOn(9) && !(hh && Camera.hhBool("hh1080", false))
            let (ww, wh): (Int32, Int32) = want4K ? (3840, 2160) : (1920, 1080)
            var fmt: AVCaptureDevice.Format?     // 60-capable, prefer binned
            for f in dev.formats {
                let d = CMVideoFormatDescriptionGetDimensions(f.formatDescription)
                guard d.width == ww, d.height == wh,
                      f.videoSupportedFrameRateRanges.contains(where: { $0.maxFrameRate >= 60 })
                else { continue }
                if fmt == nil || f.isVideoBinned { fmt = f }
                if f.isVideoBinned { break }
            }
            if let f = fmt { dev.activeFormat = f }
        }
        if Camera.camOn(2) || hh,                // #2 needs a 60-capable format (#1)
           dev.activeFormat.videoSupportedFrameRateRanges.contains(where: { $0.maxFrameRate >= 60 }) {
            dev.activeVideoMinFrameDuration = CMTime(value: 1, timescale: 60)
            dev.activeVideoMaxFrameDuration = CMTime(value: 1, timescale: 60)
        }
        if Camera.camOn(3) || Camera.camOn(4) {  // #4's hard precondition is this line
            dev.automaticallyAdjustsVideoHDREnabled = false
        }
        if Camera.camOn(3), dev.activeFormat.isVideoHDRSupported {
            dev.isVideoHDREnabled = false
        }
        if Camera.camOn(4), dev.activeFormat.isGlobalToneMappingSupported {
            dev.isGlobalToneMappingEnabled = true
        }
        if Camera.camOn(5), dev.isLowLightBoostSupported {
            dev.automaticallyEnablesLowLightBoostWhenAvailable = false
        }
        if Camera.camOn(6), dev.isGeometricDistortionCorrectionSupported {
            dev.isGeometricDistortionCorrectionEnabled = false
        }
        if dev.isFocusPointOfInterestSupported {
            dev.focusPointOfInterest = CGPoint(x: 0.5, y: 0.5)
        }
        if dev.isFocusModeSupported(.continuousAutoFocus) { dev.focusMode = .continuousAutoFocus }
        dev.unlockForConfiguration()
        if Camera.camOn(7), let c = out.connection(with: .video),   // #7 is a CONNECTION
           c.isVideoStabilizationSupported {                        // property, not device
            c.preferredVideoStabilizationMode = .off
        }
        applyRotation()
        if !rotObserved {                // live-follow rotation for the whole app life
            rotObserved = true
            UIDevice.current.beginGeneratingDeviceOrientationNotifications()
            NotificationCenter.default.addObserver(
                forName: UIDevice.orientationDidChangeNotification, object: nil,
                queue: .main) { [weak self] _ in self?.applyRotation() }
        }
        let set = (1 ... 9).filter { Camera.camOn($0) }.map { "#\($0)" }.joined(separator: "+")
        info = "camera: " + (set.isEmpty ? "all iOS defaults (no toggles)" : set)
        if hh {
            info += "  [HANDHELD: "
                + (Camera.hhBool("hhShortShutter", true) ? "1/\(Camera.hhShutterDen()) shutter, " : "")
                + (Camera.hhBool("hh1080", false) ? "1080p forced" : "") + "]"
        }
        configured = true
        DispatchQueue.global().async { self.session.startRunning() }
        return nil
    }

    var info = ""

    // one-shot autofocus at frame center, WAIT for convergence, then freeze all 3A —
    // locking blind froze whatever the lens happened to be doing (AF park lottery)
    func settleAndLock(_ done: @escaping (String) -> Void) {
        guard let d = device else { done("no device"); return }
        try? d.lockForConfiguration()
        if d.isFocusPointOfInterestSupported { d.focusPointOfInterest = CGPoint(x: 0.5, y: 0.5) }
        if d.isFocusModeSupported(.autoFocus) { d.focusMode = .autoFocus }
        if d.isExposureModeSupported(.continuousAutoExposure) { d.exposureMode = .continuousAutoExposure }
        d.unlockForConfiguration()
        DispatchQueue.global().async {
            let t0 = Date()
            usleep(300_000)                       // let the one-shot AF start moving
            while d.isAdjustingFocus && Date().timeIntervalSince(t0) < 2.5 { usleep(50_000) }
            usleep(200_000)
            try? d.lockForConfiguration()
            if d.isFocusModeSupported(.locked) { d.focusMode = .locked }
            if Camera.hhOn(), Camera.hhBool("hhShortShutter", true),
               d.isExposureModeSupported(.custom) {
                // HANDHELD: blur is linear in exposure — 1/500-1/1000 freezes hand
                // shake outright; ISO pays (clamped), noise survives, blur does not
                let den = Camera.hhShutterDen()
                let cur = CMTimeGetSeconds(d.exposureDuration)
                let iso = min(max(d.iso * Float(cur * Double(den)),
                                  d.activeFormat.minISO), d.activeFormat.maxISO)
                d.setExposureModeCustom(duration: CMTime(value: 1, timescale: CMTimeScale(den)),
                                        iso: iso, completionHandler: nil)
            } else if Camera.camOn(8), d.isExposureModeSupported(.custom),
               CMTimeGetSeconds(d.exposureDuration) > 0.008 {
                // #8: shorten to <=8 ms, ISO scaled up to keep brightness (clamped to
                // the format) — the anti-blend lever for display fps > ~20
                let tgt = CMTime(value: 1, timescale: 125)
                let iso = min(max(d.iso * Float(CMTimeGetSeconds(d.exposureDuration) / 0.008),
                                  d.activeFormat.minISO), d.activeFormat.maxISO)
                d.setExposureModeCustom(duration: tgt, iso: iso, completionHandler: nil)
            } else if d.isExposureModeSupported(.locked) { d.exposureMode = .locked }
            if d.isWhiteBalanceModeSupported(.locked) { d.whiteBalanceMode = .locked }
            let msg = String(format: "3A settled+locked: lens %.3f, %.1f ms / ISO %.0f%@",
                             d.lensPosition,
                             CMTimeGetSeconds(d.exposureDuration) * 1000, d.iso,
                             d.lensPosition > 0.85 ? "  (lens near MACRO limit — move back?)" : "")
            d.unlockForConfiguration()
            done(msg)
        }
    }

    func refocus(_ done: @escaping (String) -> Void) {   // continuous AF -> settle -> relock
        guard let d = device else { done("no device"); return }
        try? d.lockForConfiguration()
        if d.isFocusModeSupported(.continuousAutoFocus) { d.focusMode = .continuousAutoFocus }
        if d.isExposureModeSupported(.continuousAutoExposure) { d.exposureMode = .continuousAutoExposure }
        if d.isWhiteBalanceModeSupported(.continuousAutoWhiteBalance) {
            d.whiteBalanceMode = .continuousAutoWhiteBalance
        }
        d.unlockForConfiguration()
        DispatchQueue.global().asyncAfter(deadline: .now() + 1.5) {
            self.settleAndLock(done)
        }
    }

    func setLens(_ v: Float) {
        guard let d = device, (try? d.lockForConfiguration()) != nil else { return }
        d.setFocusModeLocked(lensPosition: v, completionHandler: nil)
        d.unlockForConfiguration()
    }

    var lensPosition: Float { device?.lensPosition ?? -1 }
    var currentISO: Float { device?.iso ?? -1 }

    func setISO(_ iso: Float) -> Float {     // exposure duration untouched
        guard let d = device, (try? d.lockForConfiguration()) != nil else { return -1 }
        let v = min(max(iso, d.activeFormat.minISO), d.activeFormat.maxISO)
        d.setExposureModeCustom(duration: d.exposureDuration, iso: v, completionHandler: nil)
        d.unlockForConfiguration()
        return v
    }

    func stop() { session.stopRunning() }

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        guard let pb = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        if zeroCopy, let cb = onPixelBuffer {
            cb(pb, CVPixelBufferGetWidth(pb), CVPixelBufferGetHeight(pb))
            return
        }
        CVPixelBufferLockBaseAddress(pb, .readOnly)
        let w = CVPixelBufferGetWidth(pb), h = CVPixelBufferGetHeight(pb)
        let stride = CVPixelBufferGetBytesPerRow(pb)
        let base = CVPixelBufferGetBaseAddress(pb)!.assumingMemoryBound(to: UInt8.self)
        var buf = [UInt8](repeating: 0, count: w * h * 4)
        buf.withUnsafeMutableBufferPointer { p in
            if stride == w * 4 {
                memcpy(p.baseAddress!, base, w * h * 4)
            } else {
                for y in 0 ..< h { memcpy(p.baseAddress! + y * w * 4, base + y * stride, w * 4) }
            }
        }
        CVPixelBufferUnlockBaseAddress(pb, .readOnly)
        onFrame?(buf, w, h)
    }
}
