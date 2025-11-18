# verify_pair.py
from utils_facenet import embed_from_path, cosine_similarity

img1 = "data/val/rizal/your_photo1.jpg"
img2 = "data/val/rizal/your_photo2.jpg"

emb1 = embed_from_path(img1)
emb2 = embed_from_path(img2)

if emb1 is None or emb2 is None:
    print("❌ Wajah tidak terdeteksi pada salah satu gambar.")
else:
    sim = cosine_similarity(emb1, emb2)
    threshold = 0.85

    print("Similarity:", sim)
    print("Match?" , "YA" if sim >= threshold else "TIDAK")
