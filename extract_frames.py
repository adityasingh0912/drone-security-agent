"""
extract_frames.py
Extracts 1 frame per second from security_footage.mp4
Saves them to frames/ directory as JPG images.
Run: python extract_frames.py
"""

import cv2
import os

VIDEO_PATH = "security_footage.mp4"
OUTPUT_DIR = "frames"
FPS_SAMPLE  = 1  # extract 1 frame per second


def extract_frames(video_path: str = VIDEO_PATH, output_dir: str = OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return []

    fps        = cap.get(cv2.CAP_PROP_FPS)
    total      = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration   = int(total / fps)
    interval   = int(fps * FPS_SAMPLE)

    print(f"[INFO] Video FPS     : {fps:.1f}")
    print(f"[INFO] Total frames  : {total}")
    print(f"[INFO] Duration      : {duration}s")
    print(f"[INFO] Sampling every: {interval} frames (1/sec)")

    saved     = []
    frame_idx = 0
    second    = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % interval == 0:
            filename  = f"frame_{second:04d}.jpg"
            filepath  = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, frame)
            saved.append({
                "frame_id":   second + 1,
                "time":       f"{second // 60:02d}:{second % 60:02d}",
                "filename":   filename,
                "filepath":   filepath,
                "location":   "Street / Main Gate",
            })
            print(f"  Saved: {filename} at {second}s")
            second += 1

        frame_idx += 1

    cap.release()
    print(f"\n[DONE] Extracted {len(saved)} frames to '{output_dir}/'")
    return saved


if __name__ == "__main__":
    extract_frames()