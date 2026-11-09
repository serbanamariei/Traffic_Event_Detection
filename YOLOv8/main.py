"""Detects vehicles/pedestrians and says whether there is an accident (the 'accident' class from YOLOv8).

Examples:
    python main.py --folder tumtraf_yolo/val/images --weights best.pt   # random image from the folder
    python main.py --image photo.png --weights best.pt --acc-conf 0.4
"""
import argparse
import random
from pathlib import Path

IMG_EXT = {'.png', '.jpg', '.jpeg'}


def pick_image(image, folder):
    if image:
        return Path(image)
    files = [p for p in Path(folder).rglob('*') if p.suffix.lower() in IMG_EXT]
    if not files:
        raise SystemExit(f'No images found in {folder}')
    return random.choice(files)


def summarize(dets, acc_conf):
    counts = {}
    for name, _, _ in dets:
        if name != 'accident':
            counts[name] = counts.get(name, 0) + 1
    acc = [c for name, c, _ in dets if name == 'accident' and c >= acc_conf]
    return counts, bool(acc), (max(acc) if acc else 0.0)


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--image', help='path of an image')
    g.add_argument('--folder', help='pick a random image from this folder')
    ap.add_argument('--weights', default='best.pt')
    ap.add_argument('--conf', type=float, default=0.25, help='general detection threshold')
    ap.add_argument('--acc-conf', type=float, default=0.35, help='threshold for the accident class')
    ap.add_argument('--imgsz', type=int, default=960)
    ap.add_argument('--out', default='result.jpg')
    args = ap.parse_args()

    import cv2
    from ultralytics import YOLO

    model = YOLO(args.weights)
    path = pick_image(args.image, args.folder)
    res = model.predict(str(path), conf=min(args.conf, args.acc_conf), imgsz=args.imgsz, verbose=False)[0]

    dets = [(model.names[int(b.cls)].lower(), float(b.conf), b.xyxy[0].tolist()) for b in res.boxes]
    dets = [d for d in dets if d[1] >= (args.acc_conf if d[0] == 'accident' else args.conf)]
    counts, accident, best = summarize(dets, args.acc_conf)

    print(f'Image: {path}')
    print('Detected objects:', counts if counts else 'none')
    print(f'ACCIDENT DETECTED (confidence {best:.2f})' if accident else 'No accident detected.')

    img = res.orig_img.copy()
    for name, conf, (x1, y1, x2, y2) in dets:
        color = (0, 0, 255) if name == 'accident' else (0, 200, 0)
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 6 if name == 'accident' else 2)
        cv2.putText(img, f'{name} {conf:.2f}', (int(x1), max(int(y1) - 6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    cv2.imwrite(args.out, img)
    print(f'Annotated image saved to {args.out}')


if __name__ == '__main__':
    main()
