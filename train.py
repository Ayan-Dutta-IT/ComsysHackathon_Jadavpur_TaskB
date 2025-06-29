import tensorflow as tf
from keras import layers, Model
from utils import load_image, get_image_pairs, SiameseDataGenerator
import numpy as np
import sys


def build_embedding_model():
    base = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, pooling='avg')
    x = layers.Dense(128, activation='relu')(base.output)
    return Model(base.input, x)


def build_siamese_model():
    embedding = build_embedding_model()
    input_a = tf.keras.Input(shape=(224, 224, 3))
    input_b = tf.keras.Input(shape=(224, 224, 3))
    emb_a = embedding(input_a)
    emb_b = embedding(input_b)
    x = layers.Lambda(lambda t: tf.abs(t[0] - t[1]))([emb_a, emb_b])
    output = layers.Dense(1, activation='sigmoid')(x)
    return Model([input_a, input_b], output)


print("Loading pairs...")
train_pairs, train_labels = get_image_pairs('data/train', max_pairs=50000)
val_pairs, val_labels = get_image_pairs('data/val', max_pairs=10000)

train_gen = SiameseDataGenerator(train_pairs, train_labels, batch_size=32)
val_gen = SiameseDataGenerator(val_pairs, val_labels, batch_size=32)

model = build_siamese_model()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print("Starting training...")
model.fit(train_gen, validation_data=val_gen, epochs=1)
model.save('model/siamese_model.h5')