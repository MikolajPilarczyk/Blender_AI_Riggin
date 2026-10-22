"""
OpenCV-only (CPU), HuggingFace OpenCV/MediaPipe ONNX models:
- Pose: person_detection_mediapipe + pose_estimation_mediapipe (33 keypoints)
- Hands: palm_detection_mediapipe + handpose_estimation_mediapipe (21 keypoints)

Sterowanie:
  ENTER - start / stop nagrywania (każde nagranie trafia do osobnego pliku JSON)
  Q     - wyjście (jeśli trwa nagrywanie, zostanie zapisane)

Install:
  pip install --upgrade opencv-python numpy requests
"""

import os
import json
import cv2
import numpy as np
import time
import requests
import importlib.util
from datetime import datetime

# ============================================================================
# Configuration
# ============================================================================
CAMERA = 0

# Toggle features
DETECT_POSE = True
DETECT_FACE = True
DETECT_HANDS = True

# Thresholds
POSE_PRESENCE_THRESHOLD = 0.8
HAND_CONF_THRESHOLD = 0.7

FONT = cv2.FONT_HERSHEY_DUPLEX
FONT_SCALE = 0.6
FONT_THICKNESS = 2

# Nagrywanie
OUTPUT_DIR = "recordings"          # folder na pliki JSON
ENTER_KEYS = (13, 10)              # Enter (zależnie od systemu 13 lub 10)

# Cache directory for downloaded models
CACHE_DIR = os.path.expanduser("~/.opencv_models_cache")

# ============================================================================
# Download helper
# ============================================================================

def download_file(repo_id: str, filename: str, cache_dir: str = CACHE_DIR) -> str:
    repo_cache_dir = os.path.join(cache_dir, repo_id.replace("/", "_"))
    os.makedirs(repo_cache_dir, exist_ok=True)

    local_path = os.path.join(repo_cache_dir, filename)
    if os.path.exists(local_path):
        return local_path

    url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
    print(f"  Downloading: {filename} from {repo_id}...")

    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))

        with open(local_path, 'wb') as f:
            if total_size == 0:
                f.write(response.content)
            else:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        progress = (downloaded / total_size) * 100
                        print(f"\r    Progress: {progress:.1f}%", end='', flush=True)
                print()

        print(f"  ✓ Downloaded: {filename}")
        return local_path

    except requests.exceptions.RequestException as e:
        print(f"  ✗ Error downloading {filename}: {e}")
        if os.path.exists(local_path):
            os.remove(local_path)  # nie zostawiaj uszkodzonego pliku w cache
        raise

# ============================================================================
# Shared helpers
# ============================================================================

def _load_module_from_path(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod

# ============================================================================
# Pose models (person det + pose estimation)
# ============================================================================

BODY_CONNECTIONS_MP = [
    (11, 12), (11, 23), (12, 24), (23, 24),  # torso
    (11, 13), (13, 15), (12, 14), (14, 16),  # arms
    (23, 25), (25, 27), (24, 26), (26, 28),  # legs
]

FACE_POINTS_MP = {
    0: ("Nos", (0, 255, 255)),
    2: ("L_Eye", (255, 255, 0)),
    5: ("R_Eye", (255, 255, 0)),
    7: ("L_Ear", (255, 0, 255)),
    8: ("R_Ear", (255, 0, 255)),
}
FACE_CONNECTIONS_MP = [
    (2, 0), (5, 0),
    (2, 7), (5, 8),
    (2, 5),
]

def init_pose_models():
    person_det_onnx = download_file(
        repo_id="opencv/person_detection_mediapipe",
        filename="person_detection_mediapipe_2023mar.onnx",
    )
    pose_onnx = download_file(
        repo_id="opencv/pose_estimation_mediapipe",
        filename="pose_estimation_mediapipe_2023mar.onnx",
    )
    mp_persondet_py = download_file(
        repo_id="opencv/pose_estimation_mediapipe",
        filename="mp_persondet.py",
    )
    mp_pose_py = download_file(
        repo_id="opencv/pose_estimation_mediapipe",
        filename="mp_pose.py",
    )

    mp_persondet = _load_module_from_path("mp_persondet", mp_persondet_py)
    mp_pose = _load_module_from_path("mp_pose", mp_pose_py)

    backend = cv2.dnn.DNN_BACKEND_OPENCV
    target = cv2.dnn.DNN_TARGET_CPU

    person_detector = mp_persondet.MPPersonDet(
        modelPath=person_det_onnx,
        nmsThreshold=0.3,
        scoreThreshold=0.5,
        topK=5000,
        backendId=backend,
        targetId=target,
    )

    pose_estimator = mp_pose.MPPose(
        modelPath=pose_onnx,
        confThreshold=0.8,
        backendId=backend,
        targetId=target,
    )

    return person_detector, pose_estimator


def draw_body_from_pose(frame, landmarks_screen, presence_thresh=POSE_PRESENCE_THRESHOLD):
    if landmarks_screen is None or len(landmarks_screen) < 29:
        return

    if landmarks_screen.shape[0] > 33:
        landmarks_screen = landmarks_screen[:33, :]

    pts = landmarks_screen[:, 0:2].astype(np.int32)
    presence = landmarks_screen[:, 4] if landmarks_screen.shape[1] > 4 else np.ones((len(pts),), dtype=np.float32)

    for a, b in BODY_CONNECTIONS_MP:
        if a < len(pts) and b < len(pts) and presence[a] > presence_thresh and presence[b] > presence_thresh:
            cv2.line(frame, tuple(pts[a]), tuple(pts[b]), (0, 255, 0), 2)

    for i in range(11, min(29, len(pts))):
        if presence[i] > presence_thresh:
            cv2.circle(frame, (int(pts[i][0]), int(pts[i][1])), 4, (0, 0, 255), -1)


def draw_face_from_pose(frame, landmarks_screen, presence_thresh=POSE_PRESENCE_THRESHOLD):
    if landmarks_screen is None or len(landmarks_screen) < 9:
        return False

    if landmarks_screen.shape[0] > 33:
        landmarks_screen = landmarks_screen[:33, :]

    pts = landmarks_screen[:, 0:2].astype(np.int32)
    presence = landmarks_screen[:, 4] if landmarks_screen.shape[1] > 4 else np.ones((len(pts),), dtype=np.float32)

    main_idxs = [0, 2, 5]
    if all(presence[i] <= presence_thresh for i in main_idxs):
        return False

    for a, b in FACE_CONNECTIONS_MP:
        if presence[a] > presence_thresh and presence[b] > presence_thresh:
            cv2.line(frame, tuple(pts[a]), tuple(pts[b]), (200, 200, 200), 2)

    for idx, (name, color) in FACE_POINTS_MP.items():
        if idx < len(pts) and presence[idx] > presence_thresh:
            x, y = int(pts[idx][0]), int(pts[idx][1])
            cv2.circle(frame, (x, y), 6, color, -1)
            cv2.circle(frame, (x, y), 8, (255, 255, 255), 1)

            if idx == 0:
                cv2.putText(frame, name, (x - 15, y + 20), FONT, 0.4, color, 1)
            elif idx in [2, 7]:
                cv2.putText(frame, name, (x - 45, y - 5), FONT, 0.4, color, 1)
            else:
                cv2.putText(frame, name, (x + 10, y - 5), FONT, 0.4, color, 1)

    return True


def infer_pose_landmarks(frame, person_detector, pose_estimator):
    """
    Zwraca listę krotek (landmarks_screen, landmarks_world), po jednej na osobę.
    screen = piksele (do rysowania), world = metry względem środka bioder (do nagrywania).
    """
    persons = person_detector.infer(frame)
    results = []

    for person in persons:
        pose = pose_estimator.infer(frame, person)
        if pose is None:
            continue
        # pose: (bbox, landmarks_screen, landmarks_world, mask, heatmap, conf)
        _, landmarks_screen, landmarks_world, _, _, _ = pose
        if landmarks_screen is not None and landmarks_world is not None:
            results.append((landmarks_screen, landmarks_world))

    return results

# ============================================================================
# Hand models (palm det + hand landmarks)
# ============================================================================

def init_hand_models():
    palm_onnx = download_file(
        repo_id="opencv/palm_detection_mediapipe",
        filename="palm_detection_mediapipe_2023feb.onnx",
    )
    hand_onnx = download_file(
        repo_id="opencv/handpose_estimation_mediapipe",
        filename="handpose_estimation_mediapipe_2023feb.onnx",
    )
    mp_palmdet_py = download_file(
        repo_id="opencv/handpose_estimation_mediapipe",
        filename="mp_palmdet.py",
    )
    mp_handpose_py = download_file(
        repo_id="opencv/handpose_estimation_mediapipe",
        filename="mp_handpose.py",
    )

    mp_palmdet = _load_module_from_path("mp_palmdet", mp_palmdet_py)
    mp_handpose = _load_module_from_path("mp_handpose", mp_handpose_py)

    palm_det = mp_palmdet.MPPalmDet(
        palm_onnx,
        scoreThreshold=0.5,
        nmsThreshold=0.3,
        backendId=cv2.dnn.DNN_BACKEND_OPENCV,
        targetId=cv2.dnn.DNN_TARGET_CPU,
    )

    hand_pose = mp_handpose.MPHandPose(
        hand_onnx,
        confThreshold=HAND_CONF_THRESHOLD,
        backendId=cv2.dnn.DNN_BACKEND_OPENCV,
        targetId=cv2.dnn.DNN_TARGET_CPU,
    )

    return palm_det, hand_pose


def draw_hand_landmarks_only(frame, hand_vec):
    lm = hand_vec[4:67].reshape(21, 3)
    pts = lm[:, :2].astype(int)

    edges = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9, 10), (10, 11), (11, 12),
        (0, 13), (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20),
    ]

    for a, b in edges:
        cv2.line(frame, tuple(pts[a]), tuple(pts[b]), (255, 255, 255), 2)

    for (x, y) in pts:
        cv2.circle(frame, (int(x), int(y)), 3, (0, 0, 255), -1)


def detect_and_draw_hands(frame, palm_det, hand_pose):
    """
    Rysuje dłonie i zwraca listę wektorów wyników (po jednym na dłoń).
    Układ wektora (mp_handpose.py):
      [0:4]    bbox
      [4:67]   21*3 landmarki ekranowe
      [67:130] 21*3 landmarki world (metry)
      [130]    handedness
      [131]    conf
    """
    hands = []
    palms = palm_det.infer(frame)
    if palms is None or len(palms) == 0:
        return hands

    for palm in palms:
        hand = hand_pose.infer(frame, palm)
        if hand is None:
            continue
        draw_hand_landmarks_only(frame, hand)
        hands.append(hand)

    return hands

# ============================================================================
# Nagrywanie
# ============================================================================

class Recorder:
    def __init__(self, output_dir=OUTPUT_DIR):
        self.output_dir = output_dir
        self.recording = False
        self.frames = []
        self.start_time = 0.0
        self.last_saved = None  # nazwa ostatnio zapisanego pliku (do komunikatu)
        self.last_saved_time = 0.0

    def start(self):
        self.recording = True
        self.frames = []
        self.start_time = time.time()
        print("● Nagrywanie rozpoczęte")

    def add_frame(self, pose_world, hands):
        """pose_world: tablica Nx5 lub None; hands: lista wektorów dłoni."""
        if not self.recording:
            return

        frame_data = {
            "t": time.time() - self.start_time,
            "pose": pose_world[:33, :5].tolist() if pose_world is not None else None,
            "hands": [],
        }
        for h in hands:
            h = np.asarray(h).flatten()
            if len(h) >= 132:
                frame_data["hands"].append({
                    "world": h[67:130].reshape(21, 3).tolist(),
                    "handedness": float(h[130]),
                    "conf": float(h[131]),
                })
        self.frames.append(frame_data)

    def stop_and_save(self):
        self.recording = False
        if not self.frames:
            print("■ Nagrywanie zatrzymane (brak klatek, nic nie zapisano)")
            return None

        duration = max(1e-6, self.frames[-1]["t"])
        fps = len(self.frames) / duration

        os.makedirs(self.output_dir, exist_ok=True)
        name = datetime.now().strftime("mocap_%Y%m%d_%H%M%S.json")
        path = os.path.join(self.output_dir, name)

        with open(path, "w") as f:
            json.dump({"fps": fps, "frames": self.frames}, f)

        print(f"■ Zapisano {len(self.frames)} klatek ({duration:.1f}s, ~{fps:.1f} FPS) → {path}")
        self.last_saved = path
        self.last_saved_time = time.time()
        self.frames = []
        return path

    def draw_overlay(self, frame):
        h, w = frame.shape[:2]

        if self.recording:
            elapsed = time.time() - self.start_time


            if int(elapsed * 2) % 2 == 0:
                cv2.circle(frame, (w - 150, 28), 10, (0, 0, 255), -1)
            mins, secs = divmod(int(elapsed), 60)

            cv2.putText(frame, f"REC {mins:02d}:{secs:02d}", (w - 130, 35),
                        FONT, 0.7, (0, 0, 255), 2)

            cv2.putText(frame, f"Klatki: {len(self.frames)}  |  ENTER - zakoncz",
                        (10, 90), FONT, FONT_SCALE, (0, 0, 255), 2)


            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 4)
        else:
            cv2.putText(frame, "ENTER - rozpocznij nagrywanie", (10, 90),
                        FONT, FONT_SCALE, (0, 255, 255), 2)


            if self.last_saved and time.time() - self.last_saved_time < 4:
                cv2.putText(frame, f"Zapisano: {os.path.basename(self.last_saved)}",
                            (10, 120), FONT, FONT_SCALE, (0, 255, 0), 2)

# ============================================================================
# Model loader
# ============================================================================

def load_models():
    print("Loading models (CPU mode)...")

    person_detector = None
    pose_estimator = None
    palm_det = None
    hand_pose = None

    if DETECT_POSE or DETECT_FACE:
        print("  → Pose: MediaPipe Pose (OpenCV DNN ONNX)")
        person_detector, pose_estimator = init_pose_models()

    if DETECT_HANDS:
        print("  → Hands: MediaPipe Hands (OpenCV DNN ONNX)")
        palm_det, hand_pose = init_hand_models()

    print("✓ Models loaded\n")
    return person_detector, pose_estimator, palm_det, hand_pose

# ============================================================================
# Main loop
# ============================================================================

def run_detection(camera=0):
    person_detector, pose_estimator, palm_det, hand_pose = load_models()

    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        print("Error: Cannot open webcam")
        return

    enabled = [name for name, flag in [
        ("Pose", DETECT_POSE),
        ("Face Keypoints", DETECT_FACE),
        ("Hands", DETECT_HANDS),
    ] if flag]
    print(f"▶ Running: {', '.join(enabled)}")
    print("  ENTER = start/stop nagrywania, Q = wyjście\n")

    recorder = Recorder()
    frame_count = 0
    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        frame = cv2.flip(frame, 1)

        pose_world = None
        hands = []

        # Pose + Face
        if (DETECT_POSE or DETECT_FACE) and person_detector is not None and pose_estimator is not None:
            results = infer_pose_landmarks(frame, person_detector, pose_estimator)

            for i, (lm_screen, lm_world) in enumerate(results):
                if DETECT_FACE:
                    draw_face_from_pose(frame, lm_screen, presence_thresh=POSE_PRESENCE_THRESHOLD)
                if DETECT_POSE:
                    draw_body_from_pose(frame, lm_screen, presence_thresh=POSE_PRESENCE_THRESHOLD)
                if i == 0:  # nagrywamy tylko pierwszą wykrytą osobę
                    pose_world = lm_world

        # Hands
        if DETECT_HANDS and palm_det is not None and hand_pose is not None:
            hands = detect_and_draw_hands(frame, palm_det, hand_pose)

        # Zapis klatki (tylko gdy trwa nagrywanie)
        recorder.add_frame(pose_world, hands)

        # HUD
        fps = frame_count / max(1e-6, (time.time() - start_time))
        cv2.putText(frame, f'CPU Mode | FPS: {fps:.1f}', (10, 30),
                    FONT, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f'{", ".join(enabled)} | Q - wyjscie', (10, 60),
                    FONT, FONT_SCALE, (255, 255, 255), 2)
        recorder.draw_overlay(frame)

        cv2.imshow('Analiza obrazu', frame)

        key = cv2.waitKey(1) & 0xFF
        if key in ENTER_KEYS:
            if recorder.recording:
                recorder.stop_and_save()
            else:
                recorder.start()
        elif key == ord('q'):
            break

    # Jeśli wyjście w trakcie nagrywania - zapisz to, co jest
    if recorder.recording:
        recorder.stop_and_save()

    cap.release()
    cv2.destroyAllWindows()

    total_time = time.time() - start_time
    print(f"\n✓ Detection stopped")
    print(f"  Frames: {frame_count} | Avg FPS: {frame_count/max(1e-6,total_time):.1f} | Time: {total_time:.1f}s")


if __name__ == "__main__":
    run_detection(CAMERA)