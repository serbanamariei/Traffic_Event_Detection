#!/usr/bin/env python3
"""
Completeaza etichetele lipsa: ruleaza un YOLOv8 pre-antrenat pe COCO peste imaginile din setul A9,
si adauga cutiile detectate care NU se suprapun cu etichetele existente (cele proiectate din 3D).
Etichetele originale ramân neschimbate (copie de siguranta in labels_orig).

Ruleaza in Colab (GPU), dupa dezarhivarea setului:
  !python merge_autolabels.py --root /content/a9_yolo
"""
import argparse
import shutil
from pathlib import Path

import numpy as np
import yaml

# clase COCO -> numele claselor din setul nostru
COCO2OURS = {0: "PEDESTRIAN", 1: "BICYCLE", 2: "CAR", 3: "MOTORCYCLE", 5: "BUS", 7: "TRUCK"}


def xywhn_to_xyxy(a):
    a = np.asarray(a, dtype=float).reshape(-1, 4)
    return np.stack([a[:, 0] - a[:, 2] / 2, a[:, 1] - a[:, 3] / 2,
                     a[:, 0] + a[:, 2] / 2, a[:, 1] + a[:, 3] / 2], axis=1)


def overlap(det, boxes):
    """IoU si 'continere' (intersectie / aria detectiei) intre o detectie si mai multe cutii (xyxy)."""
    if len(boxes) == 0:
        return 0.0, 0.0
    ix1 = np.maximum(det[0], boxes[:, 0]); iy1 = np.maximum(det[1], boxes[:, 1])
    ix2 = np.minimum(det[2], boxes[:, 2]); iy2 = np.minimum(det[3], boxes[:, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    a_det = (det[2] - det[0]) * (det[3] - det[1])
    a_box = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    iou = inter / np.maximum(a_det + a_box - inter, 1e-9)
    cont = inter / max(a_det, 1e-9)
    return float(iou.max()), float(cont.max())


def merge(gt_xyxy, dets, iou_thr=0.3, cont_thr=0.7):
    """dets = lista de (xyxy, cls). Intoarce lista cutiilor noi (nesuprapuse cu GT sau intre ele)."""
    known = np.array(gt_xyxy, dtype=float).reshape(-1, 4)
    added = []
    for box, c in dets:
        iou, cont = overlap(np.asarray(box, dtype=float), known)
        if iou > iou_thr or cont > cont_thr:
            continue
        added.append((box, c))
        known = np.vstack([known, np.asarray(box, dtype=float)[None]])
    return added


def read_labels(p):
    if not p.exists():
        return []
    rows = [l.split() for l in p.read_text().splitlines() if l.strip()]
    return [(int(r[0]), [float(x) for x in r[1:5]]) for r in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="folderul cu images/, labels/, data.yaml")
    ap.add_argument("--model", default="yolov8x.pt")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--min-px", type=float, default=12, help="latura minima (pixeli, la 1920x1200)")
    ap.add_argument("--iou", type=float, default=0.3)
    ap.add_argument("--cont", type=float, default=0.7)
    ap.add_argument("--preview", type=int, default=12)
    args = ap.parse_args()

    from ultralytics import YOLO
    import cv2

    root = Path(args.root)
    names = yaml.safe_load((root / "data.yaml").read_text())["names"]
    if isinstance(names, list):
        names = dict(enumerate(names))
    ours = {v: int(k) for k, v in names.items()}
    cmap = {coc: ours[n] for coc, n in COCO2OURS.items() if n in ours}
    print("Mapare COCO -> clase proprii:", {COCO2OURS[k]: v for k, v in cmap.items()})

    if not (root / "labels_orig").exists():
        shutil.copytree(root / "labels", root / "labels_orig")
        print("Copie de siguranta: labels_orig")

    model = YOLO(args.model)
    prev_dir = root / "debug_merge"
    prev_dir.mkdir(exist_ok=True)
    prev_left = args.preview
    total_added = 0

    for split in ("train", "val"):
        imgs = sorted((root / "images" / split).glob("*.jpg"))
        n_added = 0
        gen = model.predict(source=[str(p) for p in imgs], imgsz=args.imgsz, conf=args.conf,
                            classes=list(cmap), stream=True, verbose=False, batch=8)
        for p, r in zip(imgs, gen):
            H, W = r.orig_shape
            lab_p = root / "labels" / split / f"{p.stem}.txt"
            gt = read_labels(lab_p)
            gt_xyxy = xywhn_to_xyxy([b for _, b in gt])

            dets = []
            for xyxyn, c in zip(r.boxes.xyxyn.cpu().numpy(), r.boxes.cls.cpu().numpy()):
                if (xyxyn[2] - xyxyn[0]) * W < args.min_px or (xyxyn[3] - xyxyn[1]) * H < args.min_px:
                    continue
                dets.append((xyxyn, cmap[int(c)]))
            new = merge(gt_xyxy, dets, args.iou, args.cont)
            if not new:
                continue

            lines = [f"{c} {' '.join(f'{v:.6f}' for v in b)}" for c, b in gt]
            for (x1, y1, x2, y2), c in new:
                lines.append(f"{c} {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {x2 - x1:.6f} {y2 - y1:.6f}")
            lab_p.write_text("\n".join(lines), encoding="utf-8")
            n_added += len(new)

            if prev_left > 0 and len(new) >= 2:
                im = cv2.imread(str(p))
                for _, b in gt:  # existente = verde
                    x1, y1, x2, y2 = xywhn_to_xyxy([b])[0] * [W, H, W, H]
                    cv2.rectangle(im, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                for (x1, y1, x2, y2), c in new:  # adaugate = rosu
                    cv2.rectangle(im, (int(x1 * W), int(y1 * H)), (int(x2 * W), int(y2 * H)), (0, 0, 255), 2)
                cv2.imwrite(str(prev_dir / f"{p.stem}.jpg"), im)
                prev_left -= 1

        print(f"{split}: {len(imgs)} imagini, cutii adaugate automat: {n_added}")
        total_added += n_added
    print(f"Total adaugat: {total_added}. Verde = etichete originale, rosu = adaugate (vezi {prev_dir})")


if __name__ == "__main__":
    main()