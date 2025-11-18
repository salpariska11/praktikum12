# build_embeddings.py
import os
import glob
import numpy as np
from tqdm import tqdm
from utils_facenet import embed_from_path

def iter_images(root):
    classes = sorted([d for d in os.listdir(root)
                      if os.path.isdir(os.path.join(root, d))])
    for cls in classes:
        for p in glob.glob(os.path.join(root, cls, "*")):
            yield p, cls


def build_matrix(root):
    X, y, bad = [], [], []
    for path, cls in tqdm(list(iter_images(root))):
        emb = embed_from_path(path)
        if emb is None:
            bad.append(path)
            continue
        X.append(emb)
        y.append(cls)
    return np.array(X), np.array(y), bad


if __name__ == "__main__":
    print("📌 Memulai ekstraksi embedding...")
    
    X, y, bad = build_matrix("data/train")

    print("➡ Embeddings:", X.shape)
    print("➡ Labels:", y.shape)
    print("➡ Gagal deteksi:", len(bad))

    np.save("X_train.npy", X)
    np.save("y_train.npy", y)

    print("✅ Selesai! file X_train.npy dan y_train.npy berhasil dibuat.")
