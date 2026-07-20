// Geometry.swift — find the field homography without OpenCV. Approach (leaner than
// the receiver's contour search, header CRC still arbitrates): adaptive threshold on a
// /4 max-channel image -> flood the border-connected dark region away -> the largest
// ENCLOSED dark blob containing the image center is the field interior (moat + glyphs;
// its outline = the ring's inner edge, same insight as the receiver's hole trick) ->
// coarse corners by diagonal extremes -> sub-pixel edge refinement at full res (the
// refine_quad idea: fit each side to dark->bright transitions, intersect) -> 8
// orientation candidates scored by sync-corner correlation, caller tries top ones.
import Foundation

enum Geometry {
    static var lastInfo = ""          // debug: what the last findQuad call saw

    // returns candidate Hinv matrices (field-crop coords -> image coords), best first
    static func findQuad(_ bgra: [UInt8], _ w: Int, _ h: Int, cols: Int, rows: Int,
                         px: Int, C: Int) -> [[Double]] {
        lastInfo = "no blob"
        let ds = 4, dw = w / ds, dh = h / ds
        var maxc = [Int32](repeating: 0, count: dw * dh)
        bgra.withUnsafeBufferPointer { p in
            for y in 0 ..< dh { for x in 0 ..< dw {
                let o = (y * ds * w + x * ds) * 4
                maxc[y * dw + x] = Int32(max(p[o], max(p[o + 1], p[o + 2])))
            } }
        }
        var integ = [Int64](repeating: 0, count: (dw + 1) * (dh + 1))
        for y in 0 ..< dh {
            var run: Int64 = 0
            for x in 0 ..< dw {
                run += Int64(maxc[y * dw + x])
                integ[(y + 1) * (dw + 1) + x + 1] = integ[y * (dw + 1) + x + 1] + run
            }
        }
        let R = 6                                  // block 13 at /4 ~ block 51 full-res
        var white = [Bool](repeating: false, count: dw * dh)
        for y in 0 ..< dh { for x in 0 ..< dw {
            let x0 = max(0, x - R), x1 = min(dw - 1, x + R)
            let y0 = max(0, y - R), y1 = min(dh - 1, y + R)
            let s = integ[(y1 + 1) * (dw + 1) + x1 + 1] - integ[y0 * (dw + 1) + x1 + 1]
                  - integ[(y1 + 1) * (dw + 1) + x0] + integ[y0 * (dw + 1) + x0]
            let mean = s / Int64((x1 - x0 + 1) * (y1 - y0 + 1))
            white[y * dw + x] = Int64(maxc[y * dw + x]) > mean - 5
        } }
        var w2 = white                             // despeckle: isolated white noise
        for y in 1 ..< dh - 1 {                    // pixels fragment the interior blob
            for x in 1 ..< dw - 1 {                // on RAW frames (debug JPEGs smooth
                let i = y * dw + x                 // them away — measured trap)
                if white[i] {
                    var nb = 0
                    if white[i - 1] { nb += 1 }
                    if white[i + 1] { nb += 1 }
                    if white[i - dw] { nb += 1 }
                    if white[i + dw] { nb += 1 }
                    if nb < 2 { w2[i] = false }
                }
            }
        }
        white = w2
        // flood dark pixels reachable from the border -> "outside"; BFS via stack
        var mark = [UInt8](repeating: 0, count: dw * dh)   // 1 = outside, 2 = visited blob
        var stack = [Int]()
        for x in 0 ..< dw {
            for y in [0, dh - 1] where !white[y * dw + x] && mark[y * dw + x] == 0 {
                stack.append(y * dw + x); mark[y * dw + x] = 1
            }
        }
        for y in 0 ..< dh {
            for x in [0, dw - 1] where !white[y * dw + x] && mark[y * dw + x] == 0 {
                stack.append(y * dw + x); mark[y * dw + x] = 1
            }
        }
        while let i = stack.popLast() {
            let x = i % dw, y = i / dw
            for (nx, ny) in [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)] {
                if nx >= 0 && nx < dw && ny >= 0 && ny < dh {
                    let j = ny * dw + nx
                    if !white[j] && mark[j] == 0 { mark[j] = 1; stack.append(j) }
                }
            }
        }
        // enclosed dark blobs whose bbox holds the center AND stays clear of the image
        // border (the dark laptop BEZEL forms an enclosed ring too — measured trap: it
        // outweighed a noise-fragmented interior; its bbox hugs the frame edge)
        let ctrX = dw / 2, ctrY = dh / 2
        var blobs = [(Int, [Int])]()
        for start in 0 ..< dw * dh where !white[start] && mark[start] == 0 {
            var comp = [start]
            mark[start] = 2
            var head = 0
            var minX = dw, maxX = 0, minY = dh, maxY = 0
            while head < comp.count {
                let i = comp[head]; head += 1
                let x = i % dw, y = i / dw
                minX = min(minX, x); maxX = max(maxX, x)
                minY = min(minY, y); maxY = max(maxY, y)
                for (nx, ny) in [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)] {
                    if nx >= 0 && nx < dw && ny >= 0 && ny < dh {
                        let j = ny * dw + nx
                        if !white[j] && mark[j] == 0 { mark[j] = 2; comp.append(j) }
                    }
                }
            }
            if comp.count > dw * dh / 100,
               ctrX >= minX, ctrX <= maxX, ctrY >= minY, ctrY <= maxY,
               minX > 3, minY > 3, maxX < dw - 4, maxY < dh - 4 {
                blobs.append((comp.count, comp))
            }
        }
        blobs.sort { $0.0 > $1.0 }
        if blobs.isEmpty { lastInfo = "no usable blob"; return [] }
        lastInfo = "blobs " + blobs.prefix(2).map { "\($0.0)" }.joined(separator: "/")

        func maxcFull(_ x: Int, _ y: Int) -> Double {
            let o = (y * w + x) * 4
            return Double(max(bgra[o], max(bgra[o + 1], bgra[o + 2])))
        }
        // dst rect in crop coords (receiver's convention: o = GAP*C/px inside overshoot)
        let sc = Double(C) / Double(px), o = 8.0 * Double(C) / Double(px)
        let W = Double(cols * px) + 16.0, H = Double(rows * px) + 16.0
        let dst = [(-o, -o), (W * sc - o, -o), (W * sc - o, H * sc - o), (-o, H * sc - o)]
        let exp: [Double] = [255, 255, 255, 0, 255, 0, 0, 0, 0, 255, 255, 0]
        var scored = [(Double, [Double])]()

        for (_, best) in blobs.prefix(2) {         // top-2: the header CRC arbitrates
            // coarse corners: diagonal extremes (tilt < ~45 deg), scaled to full res
            var tl = (Int.max, 0), tr = (Int.min, 0), br = (Int.min, 0), bl = (Int.max, 0)
            for i in best {
                let x = i % dw, y = i / dw
                if x + y < tl.0 { tl = (x + y, i) }
                if x - y > tr.0 { tr = (x - y, i) }
                if x + y > br.0 { br = (x + y, i) }
                if x - y < bl.0 { bl = (x - y, i) }
            }
            let coarse = [tl.1, tr.1, br.1, bl.1].map {
                (Double($0 % dw * ds), Double($0 / dw * ds))
            }
            // sub-pixel side refinement: march the outward normal, strongest dark->bright
            var lines = [(Double, Double, Double)]()
            var sideFail = false
            for s in 0 ..< 4 {
                let A = coarse[s], B = coarse[(s + 1) % 4]
                var nx = B.1 - A.1, ny = -(B.0 - A.0)   // outward: corners run CW, y down
                let nl = (nx * nx + ny * ny).squareRoot()
                if nl < 1e-9 {                          // degenerate blob: corners collide
                    lastInfo += ", side \(s) degenerate"
                    sideFail = true
                    break
                }
                nx /= nl; ny /= nl
                var pts = [(Double, Double)]()
                for k in 0 ..< 24 {
                    let t = 0.08 + 0.84 * Double(k) / 23
                    let px0 = A.0 + (B.0 - A.0) * t, py0 = A.1 + (B.1 - A.1) * t
                    var bestG = 0.0, bestD = 0.0
                    var prev = 0.0
                    for step in -16 ... 22 {   // wide: perspective can squeeze a side's
                        let xd = (px0 + nx * Double(step)).rounded()       // moat + ring
                        let yd = (py0 + ny * Double(step)).rounded()
                        if !(xd >= 1 && yd >= 1 && xd < Double(w - 1) && yd < Double(h - 1)) {
                            continue           // double-domain check: NaN/inf fall here
                        }                      // instead of TRAPPING in Int()
                        let x = Int(xd), y = Int(yd)
                        let v = maxcFull(x, y)
                        if step > -16 {
                            let g = v - prev
                            if g > bestG { bestG = g; bestD = Double(step) - 0.5 }
                        }
                        prev = v
                    }
                    if bestG > 12 { pts.append((px0 + nx * bestD, py0 + ny * bestD)) }
                }
                if pts.count < 6 {
                    lastInfo += ", side \(s) only \(pts.count) edge pts"
                    sideFail = true
                    break
                }
                lines.append(fitLine(pts))
            }
            if sideFail { continue }
            var quad = [(Double, Double)]()
            for s in 0 ..< 4 {                      // corner = adjacent side intersection
                let (a1, b1, c1) = lines[(s + 3) % 4], (a2, b2, c2) = lines[s]
                let det = a1 * b2 - a2 * b1
                if abs(det) < 1e-9 { quad = []; break }
                quad.append(((c1 * b2 - c2 * b1) / det, (a1 * c2 - a2 * c1) / det))
            }
            if quad.count != 4 { continue }
            // 8 orientation candidates, scored by sync-corner color correlation
            for flip in [1, -1] {
                for r in 0 ..< 4 {
                    var q = [(Double, Double)]()
                    for i in 0 ..< 4 {
                        let idx = flip == 1 ? (i + r) % 4 : (4 - i + r) % 4
                        q.append(quad[idx])
                    }
                    guard let Hm = perspective(q, dst), let Hinv = invert3(Hm) else { continue }
                    var meas = [Double]()
                    for (tr, tc) in [(0, 0), (0, cols - 1), (rows - 1, 0), (rows - 1, cols - 1)] {
                        let fx = (Double(tc) + 0.5) * Double(C), fy = (Double(tr) + 0.5) * Double(C)
                        let qd = Hinv[6] * fx + Hinv[7] * fy + Hinv[8]
                        let sxd = ((Hinv[0] * fx + Hinv[1] * fy + Hinv[2]) / qd).rounded()
                        let syd = ((Hinv[3] * fx + Hinv[4] * fy + Hinv[5]) / qd).rounded()
                        if !(sxd >= 0 && syd >= 0 && sxd < Double(w) && syd < Double(h)) {
                            meas += [0, 0, 0]; continue   // incl. NaN/inf from a near-
                        }                                 // singular H — Int() TRAPS on those
                        let sx = Int(sxd), sy = Int(syd)
                        let ofs = (sy * w + sx) * 4
                        meas += [Double(bgra[ofs + 2]), Double(bgra[ofs + 1]), Double(bgra[ofs])]
                    }
                    scored.append((corr(meas, exp), Hinv))
                }
            }
        }
        let top = scored.sorted { $0.0 > $1.0 }
        lastInfo += ", corrs " + top.prefix(4).map { String(format: "%.2f", $0.0) }
            .joined(separator: "/")
        return top.prefix(4).map { $0.1 }   // up to 2 per blob — CRC arbitrates
    }

    static func fitLine(_ pts: [(Double, Double)]) -> (Double, Double, Double) {
        func lsq(_ pts: [(Double, Double)]) -> (Double, Double, Double) {
            let n = Double(pts.count)
            let mx = pts.map { $0.0 }.reduce(0, +) / n, my = pts.map { $0.1 }.reduce(0, +) / n
            var sxx = 0.0, sxy = 0.0, syy = 0.0
            for p in pts {
                sxx += (p.0 - mx) * (p.0 - mx); syy += (p.1 - my) * (p.1 - my)
                sxy += (p.0 - mx) * (p.1 - my)
            }
            // normal = eigenvector of smaller eigenvalue of the covariance
            let tr = sxx + syy, dt = ((sxx - syy) * (sxx - syy) + 4 * sxy * sxy).squareRoot()
            let l = (tr - dt) / 2
            var a = sxy, b = l - sxx
            if abs(a) + abs(b) < 1e-12 { a = 1; b = 0 }
            let nl = (a * a + b * b).squareRoot()
            a /= nl; b /= nl
            return (a, b, a * mx + b * my)
        }
        var (a, b, c) = lsq(pts)                  // one trim pass: drop worst quartile
        let res = pts.map { abs(a * $0.0 + b * $0.1 - c) }
        let cut = res.sorted()[res.count * 3 / 4]
        let kept = zip(pts, res).filter { $0.1 <= cut }.map { $0.0 }
        if kept.count >= 4 { (a, b, c) = lsq(kept) }
        return (a, b, c)
    }

    static func perspective(_ src: [(Double, Double)], _ dst: [(Double, Double)]) -> [Double]? {
        var M = [[Double]](); var v = [Double]()   // solve 8x8 for H (h22 = 1)
        for i in 0 ..< 4 {
            let (x, y) = src[i], (u, w_) = dst[i]
            M.append([x, y, 1, 0, 0, 0, -u * x, -u * y]); v.append(u)
            M.append([0, 0, 0, x, y, 1, -w_ * x, -w_ * y]); v.append(w_)
        }
        for col in 0 ..< 8 {                       // Gaussian elimination, partial pivot
            var piv = col
            for r in col + 1 ..< 8 where abs(M[r][col]) > abs(M[piv][col]) { piv = r }
            if abs(M[piv][col]) < 1e-12 { return nil }
            M.swapAt(col, piv); v.swapAt(col, piv)
            for r in 0 ..< 8 where r != col {
                let f = M[r][col] / M[col][col]
                if f != 0 {
                    for j in col ..< 8 { M[r][j] -= f * M[col][j] }
                    v[r] -= f * v[col]
                }
            }
        }
        var hv = (0 ..< 8).map { v[$0] / M[$0][$0] }
        hv.append(1)
        return hv
    }

    static func invert3(_ m: [Double]) -> [Double]? {
        let d = m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6])
              + m[2] * (m[3] * m[7] - m[4] * m[6])
        if abs(d) < 1e-12 { return nil }
        return [(m[4] * m[8] - m[5] * m[7]) / d, (m[2] * m[7] - m[1] * m[8]) / d, (m[1] * m[5] - m[2] * m[4]) / d,
                (m[5] * m[6] - m[3] * m[8]) / d, (m[0] * m[8] - m[2] * m[6]) / d, (m[2] * m[3] - m[0] * m[5]) / d,
                (m[3] * m[7] - m[4] * m[6]) / d, (m[1] * m[6] - m[0] * m[7]) / d, (m[0] * m[4] - m[1] * m[3]) / d]
    }

    static func corr(_ a: [Double], _ b: [Double]) -> Double {
        let n = Double(a.count)
        let ma = a.reduce(0, +) / n, mb = b.reduce(0, +) / n
        var num = 0.0, da = 0.0, db = 0.0
        for i in 0 ..< a.count {
            num += (a[i] - ma) * (b[i] - mb)
            da += (a[i] - ma) * (a[i] - ma); db += (b[i] - mb) * (b[i] - mb)
        }
        let den = (da * db).squareRoot()
        return den < 1e-9 ? 0 : num / den
    }
}
