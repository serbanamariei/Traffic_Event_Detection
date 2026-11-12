"""Detects accidents over many frames (images extracted from videos) and says when they happen.

Examples:
    python main_frames.py --folder frames/ --weights best.pt
    python main_frames.py --folder frames/video1 --weights best.pt --acc-conf 0.4 --min-frames 5
    python main_frames.py --folder frames/ --weights best.pt --video --fps 25 --stride 2 --batch 2

Project commands:
    python YOLOv8/main_frames.py --folder YOLOv8/tumtraf_yolo/val/video_frames --weights YOLOv8/best.pt
    python YOLOv8/main_frames.py --folder YOLOv8/tumtraf_yolo/val/video_frames --weights YOLOv8/best.pt --acc-conf 0.4 --min-frames 5
    python YOLOv8/main_frames.py --folder YOLOv8/tumtraf_yolo/val/video_frames --weights YOLOv8/best.pt --video --fps 25 --stride 2 --batch 2

"""
import os
import warnings

os.environ.setdefault('YOLO_VERBOSE', 'False')
warnings.filterwarnings('ignore')

import argparse
import csv
import logging
import re
from pathlib import Path

IMG_EXT = {'.png', '.jpg', '.jpeg'}
NOT_VEHICLE = {'accident', 'other', 'license_plate_location'}
MAX_BOX_AREA = 0.5   # an expanded accident box larger than this fraction of the image is discarded


def parse_ts(stem):
    nums = re.findall(r'\d+', stem)
    if len(nums) >= 2 and len(nums[0]) >= 9:
        return float(f'{nums[0]}.{nums[1]}')
    return None


def natural_key(p):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', p.name)]


def sort_frames(files):
    ts = [parse_ts(f.stem) for f in files]
    if all(t is not None for t in ts):
        return [f for _, f in sorted(zip(ts, files), key=lambda x: x[0])]
    return sorted(files, key=natural_key)


def collect_sequences(folder):
    root = Path(folder)
    groups = {}
    for p in root.rglob('*'):
        if p.is_file() and p.suffix.lower() in IMG_EXT:
            groups.setdefault(p.parent, []).append(p)
    if not groups:
        raise SystemExit(f'No images found in {folder}')
    return {('__'.join(d.relative_to(root).parts) or root.resolve().name): sort_frames(fs)
            for d, fs in sorted(groups.items())}


def _near(a, b, m):
    return not (a[2] + m < b[0] or b[2] + m < a[0] or a[3] + m < b[1] or b[3] + m < a[1])


def involved_box(acc, vehicles, W, H, margin):
    m = margin * max(acc[2] - acc[0], acc[3] - acc[1])
    inv = [v for v in vehicles if _near(acc, v[2], m)]
    boxes = [acc] + [v[2] for v in inv]
    big = [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)]
    if (big[2] - big[0]) * (big[3] - big[1]) > MAX_BOX_AREA * W * H:
        return list(acc), []
    return big, inv


def analyze(res, conf, acc_conf, margin):
    H, W = res.orig_img.shape[:2]
    names = res.names
    dets = [(names[int(b.cls[0])].lower(), float(b.conf[0]), b.xyxy[0].tolist()) for b in res.boxes]
    dets = [d for d in dets if d[1] >= (acc_conf if d[0] == 'accident' else conf)]
    veh = [d for d in dets if d[0] not in NOT_VEHICLE]
    accs = []
    for d in sorted((d for d in dets if d[0] == 'accident'), key=lambda d: -d[1]):
        big, inv = involved_box(d[2], veh, W, H, margin)
        accs.append((d[1], big, inv))
    return dets, accs


def draw(img, dets, accs):
    import cv2
    im = img.copy()
    s = max(1, round(im.shape[1] / 960))
    inv_ids = {id(v) for _, _, inv in accs for v in inv}
    for d in dets:
        if d[0] != 'accident':
            x1, y1, x2, y2 = map(int, d[2])
            hit = id(d) in inv_ids
            cv2.rectangle(im, (x1, y1), (x2, y2), (0, 140, 255) if hit else (0, 200, 0), 3 * s if hit else s + 1)
    for c, big, inv in accs:
        x1, y1, x2, y2 = map(int, big)
        cv2.rectangle(im, (x1, y1), (x2, y2), (0, 0, 255), 4 * s)
        cv2.putText(im, f'ACCIDENT {c:.2f} | {len(inv)} vehicles', (x1, max(y1 - 10, 24 * s)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7 * s, (0, 0, 255), 2 * s)
    return im


def load_image(path):
    """cv2.imread that also works with non-ASCII paths. Returns None if the file cannot be read."""
    import cv2
    import numpy as np
    try:
        return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        return None


def precision_kwargs(device):
    """FP16 on GPU. New ultralytics (>= 8.4.80) uses quantize=16; older versions only know half=True. FP32 on CPU."""
    import torch
    import ultralytics
    if str(device).lower() == 'cpu' or not torch.cuda.is_available():
        return {}
    ver = tuple(int(x) for x in re.findall(r'\d+', ultralytics.__version__)[:3])
    return {'quantize': 16} if ver >= (8, 4, 80) else {'half': True}


def find_runs(flags, max_gap):
    """Chains flagged frames separated by at most max_gap missed ones -> [(first, last, n_flagged), ...]."""
    runs, cur = [], None
    for i, f in enumerate(flags):
        if not f:
            continue
        if cur and i - cur[1] - 1 <= max_gap:
            cur[1], cur[2] = i, cur[2] + 1
        else:
            if cur:
                runs.append(tuple(cur))
            cur = [i, i, 1]
    if cur:
        runs.append(tuple(cur))
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folder', required=True, help='folder with frames; every sub-folder is one sequence')
    ap.add_argument('--weights', default='best.pt')
    ap.add_argument('--conf', type=float, default=0.25, help='general detection threshold')
    ap.add_argument('--acc-conf', type=float, default=0.35, help='threshold for the accident class')
    ap.add_argument('--imgsz', type=int, default=960)
    ap.add_argument('--min-frames', type=int, default=3, help='accident frames in a row needed for the verdict')
    ap.add_argument('--max-gap', type=int, default=1, help='missed frames tolerated inside a run')
    ap.add_argument('--margin', type=float, default=0.10, help='how close a vehicle must be to the accident box to count as involved')
    ap.add_argument('--batch', type=int, default=4, help='images processed at once (lower = less RAM)')
    ap.add_argument('--stride', type=int, default=1, help='process every N-th frame')
    ap.add_argument('--fps', type=float, default=30.0, help='fps of the frames (used for time when file names have no timestamp)')
    ap.add_argument('--device', default=None, help="e.g. 'cpu' or '0' (default: automatic)")
    ap.add_argument('--save-all', action='store_true', help='save every annotated frame, not only those with an accident')
    ap.add_argument('--out', default='output')
    args = ap.parse_args()

    import cv2
    import torch
    from ultralytics import YOLO
    logging.getLogger('ultralytics').setLevel(logging.ERROR)

    model = YOLO(args.weights)
    prec = precision_kwargs(args.device)
    seqs = collect_sequences(args.folder)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows_csv, summary = [], []
    pred_conf = min(args.conf, args.acc_conf)

    for name, files in seqs.items():
        files = files[::max(1, args.stride)]
        tss = [parse_ts(f.stem) for f in files]
        has_ts = all(t is not None for t in tss)
        when = (lambda i: tss[i] - tss[0]) if has_ts else (lambda i: i * args.stride / args.fps)
        print(f'\nSequence: {name} ({len(files)} frames)', flush=True)

        n = len(files)
        flags, best = [False] * n, [0.0] * n
        done = 0
        for start in range(0, n, args.batch):
            idxs, imgs = [], []
            for j in range(start, min(start + args.batch, n)):
                im = load_image(files[j])
                if im is not None:
                    idxs.append(j)
                    imgs.append(im)
            if imgs:
                with torch.no_grad():
                    results = model.predict(imgs, conf=pred_conf, imgsz=args.imgsz, device=args.device,
                                            verbose=False, **prec)
                for j, res in zip(idxs, results):
                    dets, accs = analyze(res, args.conf, args.acc_conf, args.margin)
                    if accs:
                        flags[j], best[j] = True, accs[0][0]
                    for c, big, inv in accs:
                        rows_csv.append([name, files[j].name, round(when(j), 2), round(c, 3), len(inv)]
                                        + [round(v) for v in big])
                    if accs or args.save_all:
                        (out / name).mkdir(parents=True, exist_ok=True)
                        img = draw(res.orig_img, dets, accs)
                        cv2.imencode('.jpg', img)[1].tofile(str(out / name / (files[j].stem + '.jpg')))
                del imgs, results   # free the chunk before loading the next one
            done = min(start + args.batch, n)
            print(f'  {done}/{n}', flush=True)

        min_frames = max(1, min(args.min_frames, len(files)))
        runs = [r for r in find_runs(flags, args.max_gap) if r[2] >= min_frames]
        for k, (a, b, n) in enumerate(runs, 1):
            print(f'  EVENT {k}: {files[a].name} -> {files[b].name} | +{when(a):.1f}s ... +{when(b):.1f}s | '
                  f'{n} accident frames | peak conf {max(best[a:b + 1]):.2f}')
        print('  ACCIDENT DETECTED' if runs else '  No accident detected.', flush=True)
        summary.append((name, len(files), sum(flags), f'+{when(runs[0][0]):.1f}s' if runs else '-', bool(runs)))

    with open(out / 'detections.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['sequence', 'frame', 'time_s', 'confidence', 'involved_vehicles', 'x1', 'y1', 'x2', 'y2'])
        w.writerows(rows_csv)

    print('\n=== SUMMARY ===')
    print(f"{'sequence':50s} {'frames':>6s} {'detected':>9s} {'first':>8s}  verdict")
    for name, n, hits, first, acc in summary:
        print(f'{name[:50]:50s} {n:6d} {hits:9d} {first:>8s}  {"ACCIDENT" if acc else "no accident"}')
    print(f'\n{len(rows_csv)} accident detections saved to {out.resolve() / "detections.csv"}')


if __name__ == '__main__':
    main()