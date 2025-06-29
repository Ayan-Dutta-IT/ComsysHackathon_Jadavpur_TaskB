import os
import cv2
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import img_to_array
import random


def load_image(path, size=(224, 224)):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Failed to read image: {path}")
    img = cv2.resize(img, size)
    img = img[..., ::-1]  # BGR to RGB
    return img_to_array(img) / 255.0


def get_image_pairs(data_dir, max_pairs=None):
    pairs = []
    labels = []
    persons = [p for p in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, p))]
    all_positive = []
    all_negative = []

    for person in persons:
        folder = os.path.join(data_dir, person)
        clean_imgs = [os.path.join(folder, f) for f in os.listdir(folder)
                      if f.endswith('.jpg') and os.path.isfile(os.path.join(folder, f))]
        distorted_folder = os.path.join(folder, 'distortion')

        if not clean_imgs or not os.path.exists(distorted_folder):
            continue

        distorted_imgs = [os.path.join(distorted_folder, f) for f in os.listdir(distorted_folder)
                          if f.endswith('.jpg')]

        for clean_img in clean_imgs:
            # Positive pairs
            for dist_img in distorted_imgs:
                all_positive.append((clean_img, dist_img))

            # Negative pairs
            neg_persons = [p for p in persons if p != person]
            sampled_neg = random.sample(neg_persons, min(len(neg_persons), len(distorted_imgs)))
            for neg in sampled_neg:
                neg_folder = os.path.join(data_dir, neg)
                neg_clean_imgs = [os.path.join(neg_folder, f) for f in os.listdir(neg_folder)
                                  if f.endswith('.jpg') and os.path.isfile(os.path.join(neg_folder, f))]
                if neg_clean_imgs:
                    neg_img = random.choice(neg_clean_imgs)
                    all_negative.append((clean_img, neg_img))

    # Balance and shuffle
    min_len = min(len(all_positive), len(all_negative))
    if max_pairs:
        min_len = min(min_len, max_pairs // 2)
    combined = list(zip(all_positive[:min_len], [1] * min_len)) + \
               list(zip(all_negative[:min_len], [0] * min_len))
    random.shuffle(combined)

    pairs, labels = zip(*combined)
    return list(pairs), list(labels)


class SiameseDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, pairs, labels, batch_size=32, shuffle=True):
        self.pairs = pairs
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.pairs))
        self.on_epoch_end()

    def __len__(self):
        return len(self.pairs) // self.batch_size

    def __getitem__(self, idx):
        indexes = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_pairs = [self.pairs[k] for k in indexes]
        batch_labels = self.labels[indexes]

        A, B = [], []
        for p1, p2 in batch_pairs:
            try:
                A.append(load_image(p1))
                B.append(load_image(p2))
            except Exception as e:
                print(f"Error loading images: {e}")
                continue

        return [np.array(A), np.array(B)], np.array(batch_labels)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)