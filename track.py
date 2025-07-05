import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

VIDEO_PATH = 'input_video.mp4'
OUTPUT_PATH = 'output_by_appearance2004.mp4'
TARGET_FRAME_NUMBER = 199  # Frame 200 (0-based index)

# === Load YOLO and Deep SORT
model = YOLO('yolov8n.pt')
tracker = DeepSort(max_age=15)

# === Step 1: Load frame 200 for manual selection
cap = cv2.VideoCapture(VIDEO_PATH)
cap.set(cv2.CAP_PROP_POS_FRAMES, TARGET_FRAME_NUMBER)
ret, target_frame = cap.read()
if not ret:
    print("❌ Couldn't read frame 200.")
    exit()

# Resize for screen
display_height = 720
scale = display_height / target_frame.shape[0]
resized = cv2.resize(target_frame, None, fx=scale, fy=scale)

# === Step 2: Let user draw box
bbox_resized = cv2.selectROI("Select the target person (Frame 200)", resized, False, False)
cv2.destroyAllWindows()

# Convert box back to original scale
x_r, y_r, w_r, h_r = bbox_resized
x = int(x_r / scale)
y = int(y_r / scale)
w = int(w_r / scale)
h = int(h_r / scale)
ref_crop = target_frame[y:y+h, x:x+w]

# === Step 3: Rewind video to beginning
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# === Setup video writer
fps = cap.get(cv2.CAP_PROP_FPS)
w_vid = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h_vid = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w_vid, h_vid))

# === Tracking variables
target_id = None
entry_frame = None
flow_line_points = {}
frame_idx = 0

# === Step 4: Start tracking
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]
    detections = []
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, cls = result
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

        try:
            crop_resized = cv2.resize(crop, (64, 128))
            ref_resized = cv2.resize(ref_crop, (64, 128))

            hist_crop = cv2.calcHist([crop_resized], [0, 1, 2], None, [8, 8, 8],
                                     [0, 256, 0, 256, 0, 256])
            hist_ref = cv2.calcHist([ref_resized], [0, 1, 2], None, [8, 8, 8],
                                    [0, 256, 0, 256, 0, 256])

            sim = cv2.compareHist(hist_crop, hist_ref, cv2.HISTCMP_CORREL)

            if sim > 0.9:
                if target_id is None:
                    target_id = track_id
                    entry_frame = frame_idx
                    print(f"[Frame {frame_idx}] 🎯 Matched target as ID {track_id} (similarity: {sim:.2f})")

                if track_id not in flow_line_points:
                    flow_line_points[track_id] = []
                flow_line_points[track_id].append((cx, cy))

                # Draw visuals (bounding box and flow line)
                cv2.rectangle(frame, (l, t), (r, b), (0, 255, 0), 2)
                cv2.putText(frame, f'TARGET ID: {track_id}', (l, t - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                for i in range(1, len(flow_line_points[track_id])):
                    cv2.line(frame, flow_line_points[track_id][i - 1],
                             flow_line_points[track_id][i], (255, 0, 0), 2)

        except Exception as e:
            print("Skipping a crop:", e)
            continue

    out.write(frame)
    frame_idx += 1

cap.release()
out.release()

# === Final terminal output
if entry_frame is not None:
    print(f"\n✅ Target first seen at frame {entry_frame}")
    print(f"🕒 Timestamp: {entry_frame / fps:.2f} seconds")
else:
    print("❌ Target not matched in any frame.")