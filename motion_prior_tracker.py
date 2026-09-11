"""
Motion-Prior Tracker
====================
A pluggable physics-prior tracking framework.

The tracking loop is fixed; the motion model is swappable (see motion_models.py).
Core loop:
  1. Context-enlarged template matching for robust localization
  2. 50-scale dense bird-body matching + quadratic interpolation for continuous w/h
  3. EMA smoothing on scale for temporal consistency
  4. Motion-model prediction + original-template re-acquisition for recovery

Output: YOLO-format .txt per image (class_id cx cy w h)
"""

import argparse
import os, sys, glob, time, json
import numpy as np
import cv2

from motion_models import make_motion_model

# ==================== Configuration ====================
INPUT_BASE      = r"C:\Users\Lenovo\Desktop\标注1"
OUTPUT_BASE     = r"C:\Users\Lenovo\Desktop\标注1_annotations"
BIRD_CLASS      = 15
CONTEXT_MARGIN  = 2          # context window = bird × (1 + 2*margin)
SEARCH_RADIUS   = 200        # local search radius (pixels)
BIRD_PAD        = 4          # extra padding around bird template
FINE_MARGIN     = 80         # bird fine-search window size
N_SCALES        = 50         # number of dense scales for bird size
SCALE_RANGE     = (0.5, 1.5) # min/max scale for bird size
SCALE_EMA       = 0.35       # EMA smoothing factor for scale
MATCH_THRESH    = 0.35       # context match threshold
BIRD_THRESH     = 0.15       # bird-body match threshold
TEMPLATE_EMA    = 0.3        # template update rate
REACQ_INTERVAL  = 10         # re-acquisition every N consecutive failures
REACQ_SCALE     = 0.25       # coarse full-image search resolution

# Motion prior (pluggable — see motion_models.py)
MOTION_MODEL    = "ConstantVelocity"
# =======================================================

DENSE_SCALES = np.linspace(SCALE_RANGE[0], SCALE_RANGE[1], N_SCALES)


def imread(path):
    """OpenCV imread with Chinese path support."""
    arr = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def clamp_norm(cx, cy, bw, bh):
    return [
        max(0.0003, min(0.9997, cx)),
        max(0.0003, min(0.9997, cy)),
        max(0.0002, min(0.9998, bw)),
        max(0.0002, min(0.9998, bh)),
    ]


def multi_scale_match(search_region, template, base_w, base_h, scales):
    """Template matching at multiple scales. Returns (best_score, x, y, scale)."""
    best_score = -1.0
    best_x, best_y, best_scale = 0, 0, 1.0
    sr_h, sr_w = search_region.shape[:2]

    for scale in scales:
        tw = max(3, int(base_w * scale))
        th = max(3, int(base_h * scale))
        if tw > sr_w or th > sr_h:
            continue
        scaled_tmpl = cv2.resize(template, (tw, th))
        result = cv2.matchTemplate(search_region, scaled_tmpl, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        if max_val > best_score:
            best_score = max_val
            best_x, best_y = max_loc
            best_scale = scale

    return best_score, best_x, best_y, best_scale


def dense_scale_match(search_region, template, base_w, base_h):
    """
    Dense multi-scale matching with quadratic peak interpolation.
    Returns (best_score, x, y, interpolated_scale).
    """
    sr_h, sr_w = search_region.shape[:2]
    scale_scores = []
    best_score = -1.0
    best_idx = 0
    best_loc = (0, 0)

    for j, scale in enumerate(DENSE_SCALES):
        tw = max(3, int(base_w * scale))
        th = max(3, int(base_h * scale))
        if tw > sr_w or th > sr_h:
            scale_scores.append(-1.0)
            continue
        scaled_tmpl = cv2.resize(template, (tw, th))
        result = cv2.matchTemplate(search_region, scaled_tmpl, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        scale_scores.append(max_val)
        if max_val > best_score:
            best_score = max_val
            best_idx = j
            best_loc = max_loc

    # Quadratic interpolation for sub-scale precision
    interp_scale = DENSE_SCALES[best_idx]
    if 1 <= best_idx < N_SCALES - 1:
        s0 = scale_scores[best_idx - 1]
        s1 = scale_scores[best_idx]
        s2 = scale_scores[best_idx + 1]
        denom = 2 * (2 * s1 - s0 - s2)
        if s1 > s0 and s1 > s2 and abs(denom) > 1e-8:
            delta = (s2 - s0) / denom
            step = DENSE_SCALES[1] - DENSE_SCALES[0]
            interp_scale = DENSE_SCALES[best_idx] + delta * step

    interp_scale = max(SCALE_RANGE[0], min(SCALE_RANGE[1], interp_scale))
    return best_score, best_loc[0], best_loc[1], interp_scale


def full_image_search(img, template, base_w, base_h, search_scale):
    """Coarse full-image search for re-acquisition."""
    H, W = img.shape[:2]
    ws = int(W * search_scale)
    hs = int(H * search_scale)
    img_s = cv2.resize(img, (ws, hs))

    tw = max(3, int(base_w * search_scale))
    th = max(3, int(base_h * search_scale))
    tmpl_s = cv2.resize(template, (tw, th))

    result = cv2.matchTemplate(img_s, tmpl_s, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    if max_val > MATCH_THRESH * 0.7:
        return max_val, int(max_loc[0] / search_scale), int(max_loc[1] / search_scale), 1.0
    return max_val, 0, 0, 1.0


def process_folder(folder_path, output_base):
    folder_name = os.path.basename(folder_path)
    images = sorted(glob.glob(os.path.join(folder_path, '*.jpg')))
    if not images:
        print(f"  [{folder_name}] No images, skipping")
        return None

    # ---- Read reference ----
    first_name = os.path.splitext(os.path.basename(images[0]))[0]
    ref_txt = os.path.join(folder_path, first_name + '.txt')
    if not os.path.exists(ref_txt):
        print(f"  [{folder_name}] No reference txt, skipping")
        return None

    with open(ref_txt, 'r') as f:
        parts = f.readline().strip().split()
        ref_bbox = [float(x) for x in parts[1:5]]

    os.makedirs(os.path.join(output_base, folder_name), exist_ok=True)

    # ---- Initialize from reference frame ----
    img0 = imread(images[0])
    H, W = img0.shape[:2]
    cx, cy, bw, bh = ref_bbox
    px_cx = int(cx * W); px_cy = int(cy * H)
    px_w = max(5, int(bw * W)); px_h = max(5, int(bh * H))

    # Bird template (with small padding)
    bx1 = max(0, px_cx - px_w // 2 - BIRD_PAD)
    by1 = max(0, px_cy - px_h // 2 - BIRD_PAD)
    bx2 = min(W, px_cx + px_w // 2 + BIRD_PAD)
    by2 = min(H, px_cy + px_h // 2 + BIRD_PAD)
    bird_tmpl = img0[by1:by2, bx1:bx2].copy()
    orig_bird_tmpl = bird_tmpl.copy()
    ref_tw = bird_tmpl.shape[1]
    ref_th = bird_tmpl.shape[0]

    # Context template (large, for localization)
    ctx_w = max(20, int(px_w * (1 + 2 * CONTEXT_MARGIN)))
    ctx_h = max(20, int(px_h * (1 + 2 * CONTEXT_MARGIN)))
    ctx_x1 = max(0, min(px_cx - ctx_w // 2, W - ctx_w))
    ctx_y1 = max(0, min(px_cy - ctx_h // 2, H - ctx_h))
    bird_off_x = px_cx - (ctx_x1 + ctx_w / 2)
    bird_off_y = px_cy - (ctx_y1 + ctx_h / 2)
    ctx_tmpl = img0[ctx_y1:ctx_y1 + ctx_h, ctx_x1:ctx_x1 + ctx_w].copy()
    orig_ctx_tmpl = ctx_tmpl.copy()

    # Tracking state + motion prior
    init_center = np.array([ctx_x1 + ctx_w / 2, ctx_y1 + ctx_h / 2], dtype=np.float64)
    motion_model = make_motion_model(MOTION_MODEL, init_center)
    smooth_scale = 1.0
    consecutive_fail = 0

    ctx_scales = [0.88, 0.94, 1.0, 1.06, 1.12]

    total = len(images)
    matched = 0; predicted = 0; recovered = 0
    frame_stats = []
    t0 = time.time()

    for idx, img_path in enumerate(images):
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        out_txt = os.path.join(output_base, folder_name, img_name + '.txt')

        img = imread(img_path)
        if img is None:
            continue

        if idx == 0:
            out_cx, out_cy, out_bw, out_bh = ref_bbox
            matched += 1
            consecutive_fail = 0
            frame_stats.append({'frame': idx, 'mode': 'ref', 'score': 1.0, 'scale': 1.0})
        else:
            pred_center = motion_model.predict()
            bird_score = 0.0
            mode = 'pred'

            # ===== PASS 1: Context localization =====
            sx1 = max(0, int(pred_center[0] - ctx_w / 2 - SEARCH_RADIUS))
            sy1 = max(0, int(pred_center[1] - ctx_h / 2 - SEARCH_RADIUS))
            sx2 = min(W, int(pred_center[0] + ctx_w / 2 + SEARCH_RADIUS))
            sy2 = min(H, int(pred_center[1] + ctx_h / 2 + SEARCH_RADIUS))
            ctx_sr = img[sy1:sy2, sx1:sx2]

            ctx_score, ctx_lx, ctx_ly, ctx_scale = multi_scale_match(
                ctx_sr, ctx_tmpl, ctx_w, ctx_h, ctx_scales)

            # ===== Re-acquisition if needed =====
            if ctx_score < MATCH_THRESH and consecutive_fail > 0 and consecutive_fail % REACQ_INTERVAL == 0:
                re_s, re_x, re_y, re_scl = full_image_search(
                    img, orig_ctx_tmpl, ctx_w, ctx_h, REACQ_SCALE)
                if re_s > ctx_score:
                    fm = 100
                    ffx1 = max(0, re_x - fm); ffy1 = max(0, re_y - fm)
                    ffx2 = min(W, re_x + ctx_w + fm); ffy2 = min(H, re_y + ctx_h + fm)
                    ff_sr = img[ffy1:ffy2, ffx1:ffx2]
                    f_s, f_x, f_y, f_scl = multi_scale_match(
                        ff_sr, orig_ctx_tmpl, ctx_w, ctx_h, ctx_scales)
                    if f_s > ctx_score:
                        ctx_score = f_s
                        ctx_lx = ffx1 - sx1 + f_x
                        ctx_ly = ffy1 - sy1 + f_y
                        ctx_scale = f_scl
                        if f_s > MATCH_THRESH:
                            recovered += 1
                            mode = 'recovered'

            if ctx_score > MATCH_THRESH:
                ctx_x = sx1 + ctx_lx
                ctx_y = sy1 + ctx_ly
                approx_bx = ctx_x + ctx_w * ctx_scale / 2 + bird_off_x * ctx_scale
                approx_by = ctx_y + ctx_h * ctx_scale / 2 + bird_off_y * ctx_scale

                # ===== PASS 2: Dense bird size estimation =====
                fx1 = max(0, int(approx_bx - FINE_MARGIN))
                fy1 = max(0, int(approx_by - FINE_MARGIN))
                fx2 = min(W, int(approx_bx + FINE_MARGIN))
                fy2 = min(H, int(approx_by + FINE_MARGIN))
                bird_sr = img[fy1:fy2, fx1:fx2]

                bird_score, blx, bly, raw_scale = dense_scale_match(
                    bird_sr, bird_tmpl, ref_tw, ref_th)

                # EMA smooth the scale
                smooth_scale = SCALE_EMA * raw_scale + (1 - SCALE_EMA) * smooth_scale
                smooth_scale = max(SCALE_RANGE[0], min(SCALE_RANGE[1], smooth_scale))

                bird_w_px = ref_tw * smooth_scale
                bird_h_px = ref_th * smooth_scale
                bird_cx = fx1 + blx + ref_tw * smooth_scale / 2
                bird_cy = fy1 + bly + ref_th * smooth_scale / 2

                if bird_score > BIRD_THRESH:
                    mode = 'matched' if mode != 'recovered' else 'recovered'
                    matched += 1
                    consecutive_fail = 0
                else:
                    consecutive_fail += 1

                # Update context tracker
                new_center = np.array([ctx_x + ctx_w * ctx_scale / 2,
                                       ctx_y + ctx_h * ctx_scale / 2])
                motion_model.update(new_center)

                # Update context template
                cy2 = min(ctx_y + int(ctx_h * ctx_scale), H)
                cx2 = min(ctx_x + int(ctx_w * ctx_scale), W)
                new_ctx = img[ctx_y:cy2, ctx_x:cx2]
                if new_ctx.shape[0] > 5 and new_ctx.shape[1] > 5:
                    ctx_tmpl = cv2.addWeighted(
                        ctx_tmpl, 1 - TEMPLATE_EMA,
                        cv2.resize(new_ctx, (ctx_w, ctx_h)), TEMPLATE_EMA, 0)

                # Update bird template
                by1 = max(0, int(bird_cy - bird_h_px / 2))
                by2 = min(H, int(bird_cy + bird_h_px / 2))
                bx1 = max(0, int(bird_cx - bird_w_px / 2))
                bx2 = min(W, int(bird_cx + bird_w_px / 2))
                new_bird = img[by1:by2, bx1:bx2]
                if new_bird.shape[0] > 0 and new_bird.shape[1] > 0:
                    bird_tmpl = cv2.addWeighted(
                        bird_tmpl, 1 - TEMPLATE_EMA,
                        cv2.resize(new_bird, (ref_tw, ref_th)), TEMPLATE_EMA, 0)
            else:
                bird_cx = pred_center[0] + bird_off_x
                bird_cy = pred_center[1] + bird_off_y
                bird_w_px = ref_tw * smooth_scale
                bird_h_px = ref_th * smooth_scale
                motion_model.update(pred_center)  # carry prediction forward (velocity unchanged)
                consecutive_fail += 1
                predicted += 1

            out_cx = bird_cx / W
            out_cy = bird_cy / H
            out_bw = bird_w_px / W
            out_bh = bird_h_px / H

            frame_stats.append({
                'frame': idx, 'mode': mode,
                'score': max(ctx_score, bird_score),
                'scale': smooth_scale
            })

        # Write output
        out_cx, out_cy, out_bw, out_bh = clamp_norm(out_cx, out_cy, out_bw, out_bh)
        with open(out_txt, 'w', encoding='utf-8') as f:
            f.write(f"{BIRD_CLASS} {out_cx:.6f} {out_cy:.6f} {out_bw:.6f} {out_bh:.6f}\n")

        if (idx + 1) % 100 == 0:
            elapsed = time.time() - t0
            fps = (idx + 1) / elapsed if elapsed > 0 else 0
            eta = (total - idx - 1) / fps if fps > 0 else 0
            print(f"  [{folder_name}] {idx+1}/{total} "
                  f"({fps:.1f} fps, M={matched}, P={predicted}, R={recovered}, ETA {eta:.0f}s)")

    elapsed = time.time() - t0
    print(f"  [{folder_name}] DONE: {total} frames in {elapsed:.0f}s "
          f"({total/elapsed:.1f} fps) — M={matched} P={predicted} R={recovered}")

    # Compute w/h statistics
    ws = [s['scale'] * ref_tw / W for s in frame_stats]
    hs = [s['scale'] * ref_th / H for s in frame_stats]

    return {
        'folder': folder_name,
        'total': total,
        'matched': matched,
        'predicted': predicted,
        'recovered': recovered,
        'match_rate': matched / total if total > 0 else 0,
        'elapsed': elapsed,
        'fps': total / elapsed if elapsed > 0 else 0,
        'w_mean': float(np.mean(ws)), 'w_std': float(np.std(ws)),
        'h_mean': float(np.mean(hs)), 'h_std': float(np.std(hs)),
        'w_unique': len(set(round(x, 8) for x in ws)),
        'h_unique': len(set(round(x, 8) for x in hs)),
        'frame_diffs': float(np.mean(np.abs(np.diff(ws)))),
    }


def main():
    p = argparse.ArgumentParser(
        description="Motion-Prior Tracker — pluggable physics-prior tracking")
    p.add_argument("--input", default=INPUT_BASE,
                   help="folder of image-sequence subfolders")
    p.add_argument("--output", default=OUTPUT_BASE,
                   help="output folder for YOLO-format .txt")
    p.add_argument("--model", default=MOTION_MODEL,
                   help="motion prior name (see motion_models.py)")
    args = p.parse_args()

    # make the chosen model visible to process_folder (reads the module global)
    globals()['MOTION_MODEL'] = args.model
    input_base = args.input
    output_base = args.output

    print("=" * 60)
    print("  Motion-Prior Tracker")
    print(f"  Motion model: {MOTION_MODEL}")
    print("  Dense-scale + interpolation + EMA smoothing")
    print("=" * 60)
    print(f"  Scales: {N_SCALES} ({SCALE_RANGE[0]}-{SCALE_RANGE[1]})")
    print(f"  Scale EMA: {SCALE_EMA}")
    print(f"  Match thresh: {MATCH_THRESH}, Bird thresh: {BIRD_THRESH}")
    print()

    subfolders = sorted([d for d in glob.glob(os.path.join(input_base, '*')) if os.path.isdir(d)])
    print(f"Input:  {input_base}")
    print(f"Output: {output_base}")
    print(f"Folders ({len(subfolders)}): {', '.join(os.path.basename(d) for d in subfolders)}")
    print()

    all_stats = []
    total_frames = 0
    total_start = time.time()

    for folder in subfolders:
        print(f"--- {os.path.basename(folder)} ---")
        stats = process_folder(folder, output_base)
        if stats:
            all_stats.append(stats)
            total_frames += stats['total']
        print()

    total_elapsed = time.time() - total_start

    # Print summary
    print("=" * 60)
    print("  FINAL SUMMARY")
    print("=" * 60)
    print(f"{'Folder':<6} {'Frames':<8} {'Match%':<10} {'w_mean':<12} {'w_std':<12} {'w_unique':<10} {'d(ms)':<10}")
    print("-" * 68)
    for s in all_stats:
        print(f"{s['folder']:<6} {s['total']:<8} {s['match_rate']:<10.1%} "
              f"{s['w_mean']:<12.6f} {s['w_std']:<12.6f} {s['w_unique']:<10} "
              f"{s['frame_diffs']*1e6:<10.1f}")
    print("-" * 68)
    print(f"Total: {total_frames} frames in {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")
    print(f"Annotations saved to: {output_base}")

    # Save stats JSON for later analysis
    stats_path = os.path.join(output_base, '_tracking_stats.json')
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(all_stats, f, indent=2, ensure_ascii=False)
    print(f"Stats saved to: {stats_path}")


if __name__ == '__main__':
    main()
