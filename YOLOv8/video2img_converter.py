
import glob
import cv2
from pathlib import Path

VIDEO_FOLDER = Path(__file__).parent / 'videos'

def extract_video_frames(step_s=0.1):

    vids = [f for f in glob.glob(str(VIDEO_FOLDER / '**' / '*.*'), recursive=True)
            if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]

    for v in vids:
        cap = cv2.VideoCapture(v)

        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        every = max(1, round(fps * step_s))

        out = Path(v).parent / Path(v).stem / '_images' / 'camera'
        out.mkdir(parents=True, exist_ok=True)

        i = 0

        while True:
            ok, fr = cap.read()

            if not ok:
                break

            if i % every == 0:
                t = 1700000000 + i / fps

                filename = f'{int(t)}_{int((t % 1) * 1e9):09d}_camera.png'

                cv2.imwrite(str(out / filename), fr)

            i += 1

        cap.release()


if __name__ == "__main__":
    extract_video_frames()
