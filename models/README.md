# models/ — base pretrained weights

All files are raw fp32, torch `state_dict` tensors concatenated in declaration
order (no header). The Swift readers (`ios/Sources/Net.swift`,
`ios/Sources/Demap.swift`) and the training exporters agree on this format.

## Glyph nets (wire v2) — 16-shape + 4-color heads

Architecture: conv3×3(3→16) relu pool2 → conv3×3(16→32) relu pool2 →
fc(32·(k/4)² → 64) relu → heads 16 + 4. k = input side.

| file | k | what |
|---|---|---|
| `glyph16_loop_L16_best.bin` | 16 | THE production base (camera-trained across many rig sessions; every other px warm-starts from it) |
| `glyph12_loop_L12n.bin` | 12 | native-12 bootstrap (picture-trained; native r12 is optics-limited — prefer canonical 12c16) |

There are no native 8/30 px models by design: the **canonical-16** scheme
samples ANY tile size at 16×16 through the homography, so keys 816/1216/3016
all start from the 16 px weights and adapt in-session. Camera-tuned per-key
snapshots from the dev phone live in `ios/tuned/` (rig-specific).

## Demappers (wire v3.1) — PAM-2 DCT cells

Architecture: conv3×3(3→32) relu pool2 → conv3×3(32→64) relu pool2 →
fc(1600→512) relu → fc(512 → 3·NC·2). Input: 20×20 canonical crop (cell +
margin) — one model serves 16 px native AND canonical-scaled 12/30 px cells.

| file | NC | provenance |
|---|---|---|
| `demap16_nc8.bin` | 8 | pretrained on synthetic + fine-tuned on 1561 real camera frames (0.40 % BER raw, 0.16 % with distortion correction) |
| `demap16_nc12.bin` | 12 | synthetic pretrain only (0.00 % synthetic; no real NC12 footage yet) |

Regenerate any of these with `training/` (minutes on a GPU).
