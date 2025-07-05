import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

VIDEO_PATH = 'input_video.mp4'
OUTPUT_PATH = 'output_tracked.mp4'
TARGET_FRAME = 199  # manually select frame for target reference

# === Load model
model = YOLO('yolov8n.pt')
tracker = DeepSort(max_age=15)

# === Open video
cap = cv2.VideoCapture(VIDEO_PATH)
cap.set(cv2.CAP_PROP_POS_FRAMES, TARGET_FRAME)
ret, ref_frame = cap.read()
if not ret:
    raise RuntimeError("❌ Cannot read target frame.")

# === Manually selected coordinates (you can use selectROI once and hardcode)
# Example: (x, y, w, h)
bbox = (968, 225, 74, 112)
x, y, w, h = bbox
ref_crop = ref_frame[y:y+h, x:x+w]

# === Setup writer
fps = cap.get(cv2.CAP_PROP_FPS)
w_vid = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h_vid = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w_vid, h_vid))

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
target_id = None
entry_frame = None
points = {}
frame_idx = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]
    detections = []
    for r in results.boxes.data.tolist():
        x1, y1, x2, y2, score, cls = r
        if int(cls) == 0:
            detections.append(([x1, y1, x2 - x1, y2 - y1], score, 'person'))

    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        l, t, r, b = map(int, track.to_ltrb())
        cx, cy = (l + r) // 2, (t + b) // 2

        crop = frame[t:b, l:r]
        if crop.size == 0 or ref_crop.size == 0:
            continue

        if target_id is None:
            try:
                crop_r = cv2.resize(crop, (64, 128))
                ref_r = cv2.resize(ref_crop, (64, 128))
                hist_crop = cv2.calcHist([crop_r], [0, 1, 2], None, [8,8,8], [0,256]*3)
                hist_ref  = cv2.calcHist([ref_r], [0, 1, 2], None, [8,8,8], [0,256]*3)
                sim = cv2.compareHist(hist_crop, hist_ref, cv2.HISTCMP_CORREL)

                if sim > 0.9:
                    target_id = track_id
                    entry_frame = frame_idx
                    print(f"[✔] Target matched at frame {frame_idx} (ID {track_id}, sim {sim:.2f})")
            except:
                continue

        if track_id == target_id:
            if track_id not in points:
                points[track_id] = []
            points[track_id].append((cx, cy))

            cv2.rectangle(frame, (l, t), (r, b), (0, 255, 0), 2)
            cv2.putText(frame, f"ID {track_id}", (l, t - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            for i in range(1, len(points[track_id])):
                cv2.line(frame, points[track_id][i - 1], points[track_id][i], (255, 0, 0), 2)

    if entry_frame is not None and frame_idx >= entry_frame:
        out.write(frame)

    frame_idx += 1

cap.release()
out.release()

if entry_frame is not None:
    print(f"\n🎯 Target first seen at frame {entry_frame}")
    print(f"⏱ Timestamp: {entry_frame / fps:.2f} seconds")
else:
    print("❌ Target not found")