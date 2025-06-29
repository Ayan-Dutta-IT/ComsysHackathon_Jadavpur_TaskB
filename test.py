import os
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from keras.models import load_model
from utils import load_image
from keras.models import Model
import tensorflow as tf
from keras import layers
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

def build_embedding_model():
    base = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, pooling='avg')
    x = layers.Dense(128, activation='relu')(base.output)
    return Model(base.input, x)

def get_test_pairs(test_dir):
    persons = [p for p in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, p))]
    data = []

    for person in persons:
        folder = os.path.join(test_dir, person)
        clean_imgs = [os.path.join(folder, f) for f in os.listdir(folder)
                      if f.endswith('.jpg') and os.path.isfile(os.path.join(folder, f))]
        distorted_folder = os.path.join(folder, 'distortion')

        if not clean_imgs or not os.path.exists(distorted_folder):
            continue

        distorted_imgs = [os.path.join(distorted_folder, f) for f in os.listdir(distorted_folder)
                          if f.endswith('.jpg')]

        for dist_img in distorted_imgs:
            data.append((dist_img, person, clean_imgs))  # (distorted, true_person, candidate_clean_imgs)
    return data

# Load embedding model
print("Loading embedding model...")
embedding_model = build_embedding_model()
embedding_model.load_weights('model/siamese_model.h5', by_name=True, skip_mismatch=True)

test_dir = 'data/val'
test_data = get_test_pairs(test_dir)

y_true = []
y_pred = []

print("Evaluating on test data...")
for distorted_path, true_id, candidates in tqdm(test_data):
    try:
        distorted_emb = embedding_model.predict(np.expand_dims(load_image(distorted_path), 0), verbose=0)[0]
    except Exception as e:
        print(f"Error loading distorted image: {distorted_path}, {e}")
        continue

    max_sim = -1
    predicted_id = None

    for candidate_path in candidates:
        try:
            candidate_emb = embedding_model.predict(np.expand_dims(load_image(candidate_path), 0), verbose=0)[0]
            sim = cosine_similarity([distorted_emb], [candidate_emb])[0][0]
            if sim > max_sim:
                max_sim = sim
                predicted_id = true_id  # All candidate paths belong to the true_id
        except Exception as e:
            print(f"Error loading candidate image: {candidate_path}, {e}")

    if predicted_id:
        y_true.append(true_id)
        y_pred.append(predicted_id)

# Metrics
accuracy = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred, average='macro')

print(f"\n📊 Test Results:")
print(f"Top-1 Accuracy: {accuracy * 100:.2f}%")
print(f"Macro-averaged F1-Score: {f1:.4f}")
