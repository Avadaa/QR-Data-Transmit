import SwiftUI

func memFootprint() -> Double {          // app's phys_footprint, the number Jetsam uses
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size
                                       / MemoryLayout<integer_t>.size)
    let kr = withUnsafeMutablePointer(to: &info) {
        $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
            task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
        }
    }
    return kr == KERN_SUCCESS ? Double(info.phys_footprint) / 1e9 : 0
}

struct RAMBar: View {
    @State var used = memFootprint()
    @State var avail = Double(os_proc_available_memory()) / 1e9
    let total = Double(ProcessInfo.processInfo.physicalMemory) / 1e9
    let timer = Timer.publish(every: 1, on: .main, in: .common).autoconnect()

    var body: some View {
        // avail = how much MORE this app may allocate before iOS kills it (Jetsam
        // headroom) — the only honest "free RAM" a phone will admit to
        Text(String(format: "RAM: %.2f used | %.2f headroom | %.1f GB device",
                    used, avail, total))
            .font(.system(size: 12, design: .monospaced))
            .foregroundColor(avail < 0.5 ? .red : .secondary)
            .frame(maxWidth: .infinity, alignment: .center)
            .onReceive(timer) { _ in
                used = memFootprint()
                avail = Double(os_proc_available_memory()) / 1e9
            }
    }
}

struct ContentView: View {
    @State var mode = 0
    @State var modelPx = 16          // which px the model buttons act on
    @State var overwriteFactory = false   // resets each launch — factory writes are deliberate
    @AppStorage("scale16r12") var scale16 = false   // canonical-16: r12 tiles read at 16x16
    @AppStorage("handheld") var handheld = false    // HH pipeline: per-frame naming + hh models
    @AppStorage("wirev3") var wireV3 = false        // v3.1 DMT: demapper decodes payloads
    @AppStorage("v3nc") var v3nc = 8                // which NC the v3 model buttons act on
    @ObservedObject var rx = LinkReceiver.shared
    var modelKey: Int {                  // px without native weights is ALWAYS canonical
        modelPx != 16 && (scale16 || wireV3 || modelPx != 12) ? modelPx * 100 + 16 : modelPx
    }
    var modelSet: String { handheld ? "hh" : "stable" }   // buttons follow the toggle

    var body: some View {
        VStack(spacing: 4) {
            RAMBar()
            switch mode {
            case 1: ReceiveView(back: { mode = 0 })
            case 2: DemoView(back: { mode = 0 })
            case 3: FrameView(back: { mode = 0 })
            case 4: SettingsView(back: { mode = 0 })
            default:
                VStack(spacing: 24) {
                    Spacer()
                    Text("QR File Transmit").font(.title2).bold()
                    Button("Receive") { mode = 1 }
                        .buttonStyle(.borderedProminent).controlSize(.large)
                    Button("Core demo / bench") { mode = 2 }
                        .buttonStyle(.bordered)
                    Button("View frames (CAL / debug)") { mode = 3 }
                        .buttonStyle(.bordered)
                    HStack(spacing: 12) {
                        Button("Settings") { mode = 4 }
                            .buttonStyle(.bordered)
                        Toggle("Handheld", isOn: $handheld)
                            .font(.caption).fixedSize()
                    }
                    Picker("", selection: $wireV3) {
                        Text("v2 glyphs").tag(false)
                        Text("v3.1 DMT").tag(true)
                    }
                    .pickerStyle(.segmented).frame(width: 220)
                    Divider().frame(width: 200)
                    Text(wireV3 ? "model (v3.1 demapper set)"
                         : handheld ? "model (HH set — handheld tunes live apart)" : "model")
                        .font(.caption).foregroundColor(.secondary)
                    HStack(spacing: 10) {
                        Picker("", selection: $modelPx) {
                            Text("8").tag(8)
                            Text("12").tag(12)
                            Text("16").tag(16)
                            Text("30").tag(30)
                        }
                        .pickerStyle(.segmented).frame(width: 200)
                        if modelPx == 12, !wireV3 {   // only r12 has a native choice;
                            Toggle("Scale to 16 px", isOn: $scale16)   // 8/30 + v3 are
                                .font(.caption).fixedSize()            // always canonical
                        }
                    }
                    if wireV3 {
                        Picker("", selection: $v3nc) {
                            Text("nc 8").tag(8)
                            Text("nc 12").tag(12)
                        }
                        .pickerStyle(.segmented).frame(width: 140)
                    }
                    HStack(spacing: 12) {
                        Button("Save fork") {
                            wireV3
                                ? LinkReceiver.shared.v3SaveFork(px: modelKey, nc: v3nc,
                                                                 alsoFactory: overwriteFactory)
                                : LinkReceiver.shared.saveFork(px: modelKey, set: modelSet,
                                                               alsoFactory: overwriteFactory)
                        }
                        Button("Load fork") {
                            wireV3
                                ? LinkReceiver.shared.v3LoadFork(px: modelKey, nc: v3nc)
                                : LinkReceiver.shared.loadFork(px: modelKey, set: modelSet)
                        }
                        Button("Factory") {
                            wireV3
                                ? LinkReceiver.shared.v3Factory(px: modelKey, nc: v3nc)
                                : LinkReceiver.shared.factoryModel(px: modelKey, set: modelSet)
                        }
                    }
                    .buttonStyle(.bordered).controlSize(.small)
                    Toggle("overwrite factory too", isOn: $overwriteFactory)
                        .font(.caption).frame(width: 220)
                    Text(rx.log.split(separator: "\n").last.map(String.init) ?? "")
                        .font(.system(size: 11, design: .monospaced))
                        .foregroundColor(.secondary)
                    Spacer()
                }
                .onAppear { UIApplication.shared.isIdleTimerDisabled = true }
            }
        }
    }
}

struct ReceiveView: View {
    let back: () -> Void
    @ObservedObject var rx = LinkReceiver.shared
    @State var tested = false
    @State var pscale: CGFloat = 1.0

    func modeColor(_ m: String) -> Color {
        switch m {
        case "WAITING": return .gray
        case "TRAIN": return .orange
        case "TRAINING": return .blue
        case "CAL": return .pink
        case "ARMED": return .yellow
        case "LIVE": return .green
        case "IDLE": return .secondary
        default: return .purple
        }
    }

    var preview: some View {
        VStack(spacing: 4) {
            if let cg = rx.previewCG {
                Image(uiImage: UIImage(cgImage: cg))
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    .cornerRadius(6)
            } else {
                RoundedRectangle(cornerRadius: 6)
                    .fill(Color.primary.opacity(0.08))
                    .overlay(Text("camera off").font(.caption).foregroundColor(.secondary))
            }
            HStack(spacing: 12) {
                Button { pscale = max(0.5, pscale - 0.25) } label: { Image(systemName: "minus.circle") }
                Button("Z") { rx.zoomPreview.toggle() }
                    .font(.caption.bold())
                    .foregroundColor(rx.zoomPreview ? .blue : .secondary)
                Button { pscale = min(2.5, pscale + 0.25) } label: { Image(systemName: "plus.circle") }
            }
        }
    }

    var main: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Button("< menu") { rx.stop(); back() }.disabled(rx.running)
                Text(rx.mode)
                    .font(.headline).bold()
                    .padding(.horizontal, 12).padding(.vertical, 5)
                    .background(modeColor(rx.mode))
                    .foregroundColor(rx.mode == "ARMED" ? .black : .white)
                    .cornerRadius(8)
                Spacer()
                Button("AF") { rx.camera.refocus { rx.out($0) } }
                    .buttonStyle(.bordered).disabled(!rx.running)
                Button("CAL") { rx.calibrate() }
                    .buttonStyle(.bordered).disabled(!rx.running || rx.mode == "CAL")
                Button("Debug") { rx.start(debug: true) }
                    .buttonStyle(.bordered).disabled(rx.running)
                Button(rx.running ? "Stop" : "Receive") {
                    rx.running ? rx.stop() : rx.start()
                }
                .buttonStyle(.borderedProminent)
            }
            Text(rx.stat)
                .font(.system(size: 13, design: .monospaced))
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(8)
                .background(Color.primary.opacity(0.06))
                .cornerRadius(8)
            ScrollViewReader { pr in
                ScrollView {
                    Text(rx.log)
                        .font(.system(size: 11, design: .monospaced))
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .id("end")
                }
                .onChange(of: rx.log) { pr.scrollTo("end", anchor: .bottom) }
            }
        }
    }

    var body: some View {
        GeometryReader { geo in
            if geo.size.width > geo.size.height {
                HStack(alignment: .top, spacing: 10) {     // landscape: preview right
                    main
                    preview.frame(width: 220 * pscale)
                }
            } else {
                VStack(spacing: 10) {                      // portrait: preview bottom
                    main
                    preview.frame(height: 150 * pscale)
                }
            }
        }
        .padding()
        .onAppear {
            UIApplication.shared.isIdleTimerDisabled = true
            if !tested { tested = true
                DispatchQueue.global(qos: .userInitiated).async { rx.selfTest() }
            }
        }
    }
}

struct SettingsView: View {
    let back: () -> Void
    @AppStorage("zeroCopyCapture") var zeroCopy = true
    @AppStorage("cam1") var cam1 = true      // default: ALL camera options 1-9 on
    @AppStorage("cam2") var cam2 = true      // (was the bisection set #1+#2+#5+#6)
    @AppStorage("cam3") var cam3 = true
    @AppStorage("cam4") var cam4 = true
    @AppStorage("cam5") var cam5 = true
    @AppStorage("cam6") var cam6 = true
    @AppStorage("cam7") var cam7 = true
    @AppStorage("cam8") var cam8 = true
    @AppStorage("cam9") var cam9 = true
    @AppStorage("hhShortShutter") var hhShort = true
    @AppStorage("hhShutter") var hhShutter = 500
    @AppStorage("hh1080") var hh1080 = false   // handheld: DON'T force 1080p by default
    @AppStorage("rtEvery") var rtEvery = 50
    @AppStorage("rtTake") var rtTake = 10

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack {
                Button("< menu") { back() }
                Spacer()
                Text("settings").font(.headline)
            }
            .padding(.bottom, 8)
            ScrollView {
                VStack(alignment: .leading, spacing: 12) {
                    Toggle("Zero-Copy Capture", isOn: $zeroCopy)
                    Text("ON: dedup reads camera frames in place; the full 8 MB copy "
                         + "happens only for committed frames and the 3 fps preview "
                         + "(~20 copies/s instead of 60). OFF: every camera frame is "
                         + "copied out (the original path). CAL and Debug always copy.")
                        .font(.caption).foregroundColor(.secondary)
                    Divider()
                    Text("runtime training (data phase)").font(.caption).bold()
                    HStack {
                        Text("Runtime train every").font(.caption)
                        Slider(value: Binding(get: { Double(rtEvery) },
                                              set: { rtEvery = Int($0) }),
                               in: 0 ... 100, step: 5)
                        Text("\(rtEvery == 0 ? "off" : "\(rtEvery)")")
                            .font(.caption).frame(width: 30)
                    }
                    HStack {
                        Text("Runtime train frames").font(.caption)
                        Slider(value: Binding(get: { Double(rtTake) },
                                              set: { rtTake = Int($0) }),
                               in: 1 ... 50, step: 1)
                        Text("\(rtTake)").font(.caption).frame(width: 30)
                    }
                    Text("Every N decoded frames, harvest the LAST M of them (decoded "
                         + "payload = free labels) and fine-tune a clone on ONE "
                         + "background thread while decode keeps running; the swap "
                         + "re-patches the ANE. Counting restarts after each tune. "
                         + "Tracks thermal channel drift on long transfers. 0 = off.")
                        .font(.caption).foregroundColor(.secondary)
                    Divider()
                    Text("camera (bisection #1-8, docs/iphone_camera_specs.txt — "
                         + "production = 1+2+5+6)")
                        .font(.caption).bold()
                    Toggle("1 · explicit 1080p60 format", isOn: $cam1)
                    Toggle("2 · 60 fps capture (needs 1)", isOn: $cam2)
                    Toggle("3 · videoHDR off", isOn: $cam3)
                    Toggle("4 · global tone mapping ON", isOn: $cam4)
                    Toggle("5 · low-light boost off", isOn: $cam5)
                    Toggle("6 · geo-distortion corr off", isOn: $cam6)
                    Toggle("7 · stabilization (EIS) off", isOn: $cam7)
                    Toggle("8 · short shutter \u{2264}8 ms, ISO up", isOn: $cam8)
                    Toggle("9 · 4K capture (r12: 2x px/tile)", isOn: $cam9)
                    Text("Everything applies at the next Receive start (the camera "
                         + "session reconfigures when toggles changed). 8 kicks in at "
                         + "the 3A lock — retrain after flipping it, the noise "
                         + "distribution shifts.")
                        .font(.caption).foregroundColor(.secondary)
                    Divider()
                    Text("handheld overrides — active ONLY while Handheld is on; "
                         + "they OVERWRITE the toggles above")
                        .font(.caption).bold()
                    Toggle("ultra-short shutter (freezes hand shake)", isOn: $hhShort)
                    if hhShort {
                        Picker("", selection: $hhShutter) {
                            Text("1/250").tag(250)
                            Text("1/500").tag(500)
                            Text("1/1000").tag(1000)
                        }
                        .pickerStyle(.segmented).frame(width: 230)
                    }
                    Toggle("force 1080p (least rolling-shutter jello; overrides 9)",
                           isOn: $hh1080)
                    Text("Also forced in handheld: explicit format + 60 fps capture "
                         + "(1+2). ISO auto-scales with the short shutter (clamped) — "
                         + "brighten the panel to buy it back. Stabilization stays OFF "
                         + "(iPhone OIS comes bundled with EIS frame-warp — parked "
                         + "experiment). Brace your elbows: Z-drift beats angular "
                         + "shake at this depth of field. Retrain after toggling.")
                        .font(.caption).foregroundColor(.secondary)
                }
            }
        }
        .padding()
    }
}

struct FrameView: View {         // in-app viewer for the frames the receiver saved
    let back: () -> Void
    @State var name = "cal.jpg"
    @State var img: UIImage?
    @State var actual = false    // false = fit, true = 1:1 pixels (scrollable)

    func load() {
        img = UIImage(contentsOfFile:
            LinkReceiver.docs.appendingPathComponent(name).path)
    }

    var body: some View {
        VStack(spacing: 8) {
            HStack {
                Button("< menu") { back() }
                Picker("", selection: $name) {
                    Text("CAL").tag("cal.jpg")
                    Text("dbg0").tag("dbg0.jpg")
                    Text("dbg1").tag("dbg1.jpg")
                    Text("dbg2").tag("dbg2.jpg")
                }
                .pickerStyle(.segmented)
                Button(actual ? "fit" : "1:1") { actual.toggle() }
                    .buttonStyle(.bordered)
            }
            if let im = img {
                if actual {
                    ScrollView([.horizontal, .vertical]) { Image(uiImage: im) }
                } else {
                    Image(uiImage: im).resizable().aspectRatio(contentMode: .fit)
                    Spacer()
                }
            } else {
                Spacer()
                Text("no \(name) saved yet").foregroundColor(.secondary)
                Spacer()
            }
        }
        .padding()
        .onAppear { load() }
        .onChange(of: name) { load() }
    }
}

struct DemoView: View {
    let back: () -> Void
    @StateObject var run = Runner()

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Button("< menu") { back() }
                Spacer()
                Text("rx core demo").font(.headline)
            }
            HStack {
                Button("Decode") { run.bg { run.runDecode() } }
                Button("Bench") { run.bg { run.runBench() } }
                Button("Warp") { run.bg { run.runWarpBench() } }
                Button("Train") { run.trainDemo() }
                Button("Reset") { run.reset() }
            }
            .buttonStyle(.bordered).disabled(run.busy)
            if run.busy { ProgressView() }
            ScrollViewReader { pr in
                ScrollView {
                    Text(run.log)
                        .font(.system(size: 11, design: .monospaced))
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .id("end")
                }
                .onChange(of: run.log) { pr.scrollTo("end", anchor: .bottom) }
            }
        }
        .padding()
        .onAppear { run.boot() }
    }
}
