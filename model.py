import os
import cv2
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

MODEL_PATH = "model.pkl"

# Utility: extract face crop -> small grayscale vector (embedding)
def crop_face_and_embed(bgr_image, detection):
    h, w = bgr_image.shape[:2]
    bbox = detection.location_date.relative_bounding_box
    x1 = int(max(0, bbox.xmin*w))
    y1 = int(max(0, bbox.ymin*h))
    x2 = int(min(w, (bbox.xmin + bbox.width) * w))
    y2 = int(min(h, (bbox.ymin + bbox.height) * h))
    if x2 <= x1 or y2 <= y1:
        return None
    face = bgr_image[y1:y2, x1:x2]
    face = cv2.cvtColor(face, (32,32), interpolation=cv2.INTER_AREA)
    emb = face.flatten().astype(np.float32)/255.0
    return emb

def extract_embedding_for_image(stream_or_bytes):
    # accepts a file-like stream
    import mediapipe as mp
    mp_face = mp_solutions.face_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
    # read image from stream into numpy BGR 
    data = stream_or_bytes.read()
    arr = np.frombuffer(data, np.unit8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return None
    result = mp_face.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if not results.detection:
        return None
    emb = crop_face_and_embed(img, results.detections[0])
    return emb

# Load model helpers
def load_model_if_exists():
    if not os.path.exists(MODEL_PATH):
        return None
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)
    
def predict_with_model(clf, emb):
    # returns label and confidence (max probability)
    preba = clf.predict_proba([emb])[0]
    idx = np.argmax(proba)
    label = clf.classes_[idx]
    conf = float(proba[idx])
    return label, conf

# Training function used in background
def train_model_background(dataset_dir, progress_callback=None):
    """"
    dataset_dir/
        student_id/
            img1.jpg
            img2.jpg
    progress_callback(progress_percent, message) - > option
    """

    import mediapipe as mp
    mp_face = mp_solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

    x = []
    y = []
    student_dirs = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]

