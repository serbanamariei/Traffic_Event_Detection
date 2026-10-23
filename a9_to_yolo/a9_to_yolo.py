#!/usr/bin/env python3
"""
Converteste A9 R2 (etichete 3D OpenLABEL pe LiDAR) in format YOLO (2D, imagini de camera).

Exemplu (recomandat):
  python a9_to_yolo.py --root "C:\\date\\a9" --out "C:\\date\\a9_yolo" --stride 3 --map EMERGENCY_VEHICLE:CAR --drop OTHER --debug 12

--root trebuie sa contina folderele a9_dataset_r02_s01 ... a9_dataset_r02_s04
Dependente: pip install numpy opencv-python   (opencv e folosit doar pentru imaginile debug)
"""
import argparse
import json
import random
import shutil
from collections import Counter
from itertools import product
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


def quat_to_R(qx, qy, qz, qw):
    n = np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ])


SIGNS = np.array(list(product([-1, 1], repeat=3)), dtype=float)  # 8 colturi


def box_corners(val):
    # val = [x, y, z, qx, qy, qz, qw, length, width, height]  (sistemul LiDAR)
    x, y, z, qx, qy, qz, qw, l, w, h = val
    R = quat_to_R(qx, qy, qz, qw)
    local = SIGNS * np.array([l / 2, w / 2, h / 2])
    return local @ R.T + np.array([x, y, z])


def project_box(corners, T, K, W, H, min_depth, min_vis, min_px):
    """Returneaza (x1, y1, x2, y2) in pixeli sau None daca nu e utilizabila."""
    cam = corners @ T[:3, :3].T + T[:3, 3]
    if (cam[:, 2] < min_depth).any():
        return None
    p = cam @ K.T
    uv = p[:, :2] / p[:, 2:3]
    x1, y1 = uv.min(axis=0)
    x2, y2 = uv.max(axis=0)
    full = (x2 - x1) * (y2 - y1)
    if full <= 0:
        return None
    cx1, cy1 = max(x1, 0.0), max(y1, 0.0)
    cx2, cy2 = min(x2, W), min(y2, H)
    if cx2 - cx1 < min_px or cy2 - cy1 < min_px:
        return None
    vis = ((cx2 - cx1) * (cy2 - cy1)) / full
    if vis < min_vis:
        return None
    return cx1, cy1, cx2, cy2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--val-mode", choices=["time", "seq"], default="time",
                    help="time = ultimele cadre din fiecare secventa merg la val; seq = secvente intregi la val")
    ap.add_argument("--val-frac", type=float, default=0.2, help="(time) fractiunea de cadre pentru val")
    ap.add_argument("--gap", type=int, default=10, help="(time) cadre sarite intre train si val")
    ap.add_argument("--val-seqs", default="s04", help="(seq) secvente pentru val, separate prin virgula")
    ap.add_argument("--stride", type=int, default=1, help="pastreaza doar fiecare al N-lea cadru")
    ap.add_argument("--map", nargs="*", default=[], help="unire clase, ex: EMERGENCY_VEHICLE:CAR")
    ap.add_argument("--drop", nargs="*", default=[], help="clase scoase, ex: OTHER")
    ap.add_argument("--min-vis", type=float, default=0.4, help="fractiunea minima vizibila din cutie")
    ap.add_argument("--min-px", type=float, default=6, help="latura minima a cutiei in pixeli")
    ap.add_argument("--min-depth", type=float, default=0.5)
    ap.add_argument("--invert", action="store_true", help="inverseaza matricea camerei")
    ap.add_argument("--debug", type=int, default=0, help="cate imagini cu cutii desenate sa salveze")
    ap.add_argument("--yaml-path", default="/content/a9_yolo", help="calea din data.yaml (cea din Colab)")
    args = ap.parse_args()

    root, out = Path(args.root), Path(args.out)
    val_seqs = [s.strip() for s in args.val_seqs.split(",") if s.strip()]
    cmap = dict(m.split(":") for m in args.map)
    drop = set(args.drop)

    def canon(t):
        if t in drop:
            return None
        t = cmap.get(t, t)
        return None if t in drop else t

    # ---------- pasul 1: incarca JSON-urile si colecteaza clasele ----------
    seqs = {}
    types = set()
    for seq_dir in sorted(root.glob("a9_dataset_r02_s*")):
        lst = []
        for jp in sorted(seq_dir.glob("labels_point_clouds/*/*.json")):
            data = json.loads(jp.read_text(encoding="utf-8"))["openlabel"]
            lst.append(data)
            for fr in data["frames"].values():
                for ob in fr.get("objects", {}).values():
                    c = canon(ob["object_data"]["type"])
                    if c:
                        types.add(c)
        if lst:
            seqs[seq_dir] = lst
    if not seqs:
        raise SystemExit("Nu am gasit nicio eticheta. Verifica --root.")
    classes = sorted(types)
    cls_id = {c: i for i, c in enumerate(classes)}
    print("Clase folosite:", cls_id)

    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
    dbg_dir = out / "debug"
    if args.debug:
        dbg_dir.mkdir(exist_ok=True)

    # ---------- pasul 2: proiectie + scriere ----------
    stats = {"train": Counter(), "val": Counter()}
    n_img = Counter()
    missing = 0
    dbg_left = args.debug
    random.seed(0)

    for seq_dir, lst in seqs.items():
        seq = seq_dir.name.split("_")[-1]  # s01, s02, ...
        n = len(lst)
        cut = int(n * (1 - args.val_frac))
        for idx, data in enumerate(lst):
            if idx % args.stride:
                continue
            if args.val_mode == "seq":
                split = "val" if seq in val_seqs else "train"
            else:
                if idx >= cut:
                    split = "val"
                elif idx < cut - args.gap:
                    split = "train"
                else:
                    continue  # zona tampon dintre train si val
            for fr in data["frames"].values():
                props = fr["frame_properties"]
                for cam in data["streams"]:
                    fname = next((f for f in props["image_file_names"] if cam in f), None)
                    if fname is None:
                        continue
                    src = seq_dir / "images" / cam / fname
                    if not src.exists():
                        missing += 1
                        continue
                    sp = data["streams"][cam]["stream_properties"]["intrinsics_pinhole"]
                    W, H = sp["width_px"], sp["height_px"]
                    K = np.array(sp["camera_matrix_3x4"], dtype=float)[:, :3]
                    T = np.array(data["coordinate_systems"][cam]["pose_wrt_parent"]["matrix4x4"],
                                 dtype=float).reshape(4, 4)
                    if args.invert:
                        T = np.linalg.inv(T)

                    lines, drawn = [], []
                    for ob in fr.get("objects", {}).values():
                        od = ob["object_data"]
                        cname = canon(od["type"])
                        if cname is None:
                            continue
                        box = project_box(box_corners(od["cuboid"]["val"]), T, K, W, H,
                                          args.min_depth, args.min_vis, args.min_px)
                        if box is None:
                            continue
                        x1, y1, x2, y2 = box
                        lines.append(f"{cls_id[cname]} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} "
                                     f"{(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}")
                        drawn.append((box, cname))
                        stats[split][cname] += 1

                    stem = f"{seq}_{Path(fname).stem}"
                    shutil.copy2(src, out / "images" / split / f"{stem}.jpg")
                    (out / "labels" / split / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")
                    n_img[split] += 1

                    if dbg_left > 0 and cv2 is not None and random.random() < 0.05:
                        im = cv2.imread(str(src))
                        for (x1, y1, x2, y2), t in drawn:
                            cv2.rectangle(im, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                            cv2.putText(im, t, (int(x1), max(int(y1) - 4, 12)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        cv2.imwrite(str(dbg_dir / f"{stem}.jpg"), im)
                        dbg_left -= 1

    # ---------- data.yaml ----------
    yaml = [f"path: {args.yaml_path}", "train: images/train", "val: images/val", "names:"]
    yaml += [f"  {i}: {c}" for i, c in enumerate(classes)]
    (out / "data.yaml").write_text("\n".join(yaml) + "\n", encoding="utf-8")

    print(f"\nImagini: train={n_img['train']}  val={n_img['val']}  (lipsa pe disc: {missing})")
    for split in ("train", "val"):
        print(f"Cutii {split}: {sum(stats[split].values())}  ->  {dict(stats[split])}")
    if args.debug and cv2 is None:
        print("opencv lipseste, nu am generat imaginile debug (pip install opencv-python)")
    elif args.debug:
        print(f"Imagini de verificat: {dbg_dir}")


if __name__ == "__main__":
    main()