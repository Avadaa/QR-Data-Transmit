 how would you do the rateless fountain? what are the advantages, performance, error rates, would it be worth it? any negative sides? no execution yet.

Short version: at our scale, a true rateless codec (RaptorQ/wirehair) buys almost nothing over what we have — because our dense-XOR scheme is already information-theoretically near-optimal at small K. The thing worth taking from the fountain world isn't the codec, it's two behaviors: repair spread through the stream, and an open-ended repair tail. Both fit our existing 20-line GF(2) math.

Where we actually stand

Our current scheme is a fountain — a fixed-rate one. Each repair frame is a dense random GF(2) combination of all sources, and dense random matrices are full-rank with probability ≈ 1 − 2^−(R−m) (R = repairs received, m = frames lost). Lose m, receive m+10 repairs → 99.9% solve. That "+2 to +10 frames" overhead is what RaptorQ spends its whole precode machinery to achieve — at K = 10⁴–10⁶ symbols. We run K = 100–1500 frames. LT codes are actually worse than dense XOR at our K (robust-soliton overhead is 5–15% at K≈1000, with high variance); RaptorQ matches dense XOR's overhead but pays in complexity. The asymptotics only matter when Gaussian elimination becomes the bottleneck, and ours runs over the missing set only — m ≤ a few hundred, Python-int bitmasks, milliseconds.

Encode cost check for the mac at r12/15fps scale: a 30 MB transfer is ~5000 frames; one dense repair = XOR of ~2500 rows × 6 KB = 15 MB of XOR ≈ a few ms in numpy. Fine even generated on the fly.

What's genuinely wrong with the current scheme

1. The rate is guessed at transmit time. FRAMES//40 covers a 2.5% uniform loss. Channel worse → whole run dead, rerun everything (the 4 MB transfer died exactly this way: 21 lost > 12 repairs). Channel better → wasted airtime (last run: 5 of 12 repairs unused).
2. All repairs sit at the end. A burst late in the run (thermal throttle, AF hiccup, you walking past the rig) can kill sources and the repairs that would have saved them, since they're adjacent in time.
3. No "done" moment. The receiver knows the instant it's solvable, but the transmitter has already stopped.

What I'd do (design only)

Keep the codec. Change the schedule.

- Interleave repairs mid-stream, generation-wise. Split sources into generations of G ≈ 128 frames; after each generation, emit its ~G/40 dense repairs immediately, masks seeded per generation. Burst losses now only threaten one generation's worth of data, decode stays the same tiny elimination per generation, and transmitter memory drops from "whole file" to "one generation" (nice for huge files).
- Rateless tail: after the last generation, instead of END, loop fresh repair frames (repair ordinal keeps incrementing — seed makes them free to generate forever, no state needed) cycling across generations. The receiver flips to done the moment every generation solves, prints DONE + md5, and the human (or later, an audio beep backchannel) stops the mac. That is functionally rateless: the receiver completes from any sufficient subset, and bad-channel runs cost extra seconds instead of a rerun.
- Wire cost: nothing new. Header idx ≥ FRAMES already marks repairs; the ordinal already implies the mask; generations are just idx // G.

Would wirehair/RaptorQ ever be worth it?

Only if K grows past ~10⁴ frames per solve and losses get large enough that elimination over m ≈ thousands hurts — i.e., 100 MB+ transfers at r12/15fps with a hostile channel. Even then, the practical blocker is real: the Surface has no compiler (that's what killed torch.compile), wirehair is C++ with thin, poorly maintained Python bindings, and it would be our only native dependency to cross-build for three machines. Generations solve the same scaling problem for free by capping K at G.

Honest cost/benefit

┌─────────────────────────────┬─────────────────────────┬─────────────────────────────┬──────────────────────────┐
│                             │ now (end-loaded, fixed) │ generations + rateless tail │     wirehair/RaptorQ     │
├─────────────────────────────┼─────────────────────────┼─────────────────────────────┼──────────────────────────┤
│ overhead when channel clean │ ~2.5% fixed             │ ~2.5% + tail you stop early │ ~identical               │
├─────────────────────────────┼─────────────────────────┼─────────────────────────────┼──────────────────────────┤
│ bad channel                 │ run dies, full rerun    │ a few extra tail seconds    │ a few extra tail seconds │
├─────────────────────────────┼─────────────────────────┼─────────────────────────────┼──────────────────────────┤
│ burst robustness            │ poor                    │ good                        │ good                     │
├─────────────────────────────┼─────────────────────────┼─────────────────────────────┼──────────────────────────┤
│ decode cost                 │ ~ms                     │ ~ms                         │ ~ms but native lib       │
├─────────────────────────────┼─────────────────────────┼─────────────────────────────┼──────────────────────────┤
│ code delta                  │ —                       │ ~30 lines both ends         │ new C++ dep, build pain  │
└─────────────────────────────┴─────────────────────────┴─────────────────────────────┴──────────────────────────┘

Negatives to be aware of even for the cheap version: the transmit loop loses its clean "N frames then END" shape (END becomes "receiver says done", which changes the timeout logic and the goodput accounting); repair airtime during the tail is pure overhead on clean runs, so goodput numbers dip unless you stop promptly; and generation size is a new knob to get wrong (too small → per-generation +2-frame overhead multiplies; G=128 keeps that under ~1.5%).

Verdict: worth it, but as ~30 lines of schedule change on our existing math — not as a codec import. And it ranks behind the r12@15fps record-and-decode experiment, since fountain polish only matters once frames are flowing at that rate.