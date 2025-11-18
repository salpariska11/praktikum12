# utils_facenet.py
import torch
import numpy as np
import cv2
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Detector & aligner
mtcnn = MTCNN(image_size=160, margin=20, post_process=True, device=device)

# Embedder (512-dim)
embedder = InceptionResnetV1(pretrained='vggface2').eval().to(device)


def read_img_bgr(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Gagal membaca gambar: {path}")
    return img


def bgr_to_pil(img_bgr):
    return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))


@torch.no_grad()
def face_align(img_bgr):
    """Mengembalikan wajah yang sudah di-align ukuran 160x160, atau None."""
    pil = bgr_to_pil(img_bgr)
    aligned = mtcnn(pil)   # hasil: tensor [3,160,160] atau None
    return aligned


@torch.no_grad()
def embed_face_tensor(face_tensor):
    """Mengubah tensor wajah menjadi embedding FaceNet (512 dim)."""
    if face_tensor is None:
        return None
    face_tensor = face_tensor.unsqueeze(0).to(device)
    emb = embedder(face_tensor)
    return emb.squeeze(0).cpu().numpy()


@torch.no_grad()
def embed_from_path(path):
    """Baca gambar → align wajah → hasil embedding 512-dim."""
    img = read_img_bgr(path)
    face = face_align(img)
    if face is None:
        return None
    return embed_face_tensor(face)


def cosine_similarity(a, b, eps=1e-8):
    """Hitung cosine similarity antara dua embedding."""
    a = a / (np.linalg.norm(a) + eps)
    b = b / (np.linalg.norm(b) + eps)
    return float(np.dot(a, b))
