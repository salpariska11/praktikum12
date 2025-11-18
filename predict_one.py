import joblib
from utils_facenet import embed_from_path
import numpy as np

# Load classifier
clf = joblib.load("facenet_svm.joblib")

def predict_image(path, unknown_threshold=0.55):
    emb = embed_from_path(path)
    if emb is None:
        return "NO_FACE", 0.0
    
    # Prediksi probabilitas
    proba = clf.predict_proba([emb])[0] 
    idx = int(np.argmax(proba))
    label = clf.classes_[idx]
    conf = float(proba[idx])
    
    # Jika confidence rendah → UNKNOWN
    if conf < unknown_threshold:
        return "UNKNOWN", conf
    
    return label, conf


if __name__ == "__main__":
    # Tentukan foto untuk diuji
    test_img = "data/val/salva/salpa2.jpg"   # sesuaikan dengan folder kamu

    label, conf = predict_image(test_img)
    print(f"Prediksi: {label} (conf={conf:.3f})")