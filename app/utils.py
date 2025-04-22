import numpy as np
from torchvision import transforms
from sklearn.metrics.pairwise import cosine_similarity
import onnxruntime as ort
from face_alignment import get_aligned_face

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

def get_embedding(pil_img, session, input_name, expected_dtype):
    tensor = transform(pil_img).unsqueeze(0).numpy()
    tensor = tensor.astype(np.float16) if "float16" in expected_dtype.lower() else tensor.astype(np.float32)

    ort_inputs = {input_name: tensor}
    emb = session.run(None, ort_inputs)[0]
    return emb[0] / np.linalg.norm(emb[0])

def recognize_face(face_emb):
    EMB_DIR = "datasets/embeddings"
    THRESHOLD = 0.5
    best_score = -1
    best_match = "new person"
    for person in os.listdir(EMB_DIR):
        person_dir = os.path.join(EMB_DIR, person)
        if not os.path.isdir(person_dir): continue
        for emb_file in os.listdir(person_dir):
            if emb_file.endswith(".npy"):
                db_emb = np.load(os.path.join(person_dir, emb_file))
                score = cosine_similarity([face_emb], [db_emb])[0][0]
                if score > best_score:
                    best_score = score
                    best_match = person if score >= THRESHOLD else "new person"
    return best_match, best_score
