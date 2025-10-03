#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout, MultiHeadAttention, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

# -------------------------------
# 1. Load the data
# -------------------------------
data = pd.read_csv("Data_Cancers.csv")

# Keep only the desired classes
classes_to_keep = [
    'Breast_cancer', 'Gastric_cancer', 'Non-small_cell_lung_cancer',
    'Colorectal_cancer', 'Pancreatic_cancer', 'Acute_myeloid_leukemia',
    'Renal_cell_carcinoma', 'Prostate_cancer', 'Small_cell_lung_cancer',
    'Thyroid_cancer', 'Glioma', 'Hepatocellular_carcinoma', 'Chronic_myeloid_leukemia',
    'Endometrial_cancer', 'Melanoma', 'Bladder_cancer', 'Basal_cell_carcinoma'
]
data = data[data['Class'].isin(classes_to_keep)]

# -------------------------------
# 2. Class weighting
# -------------------------------
class_weights = {
    'Breast_cancer': 0.0757, 'Gastric_cancer': 0.1012, 'Non-small_cell_lung_cancer': 0.1136,
    'Colorectal_cancer': 0.1250, 'Pancreatic_cancer': 0.1263, 'Acute_myeloid_leukemia': 0.1294,
    'Renal_cell_carcinoma': 0.1798, 'Prostate_cancer': 0.2019, 'Small_cell_lung_cancer': 0.2667,
    'Thyroid_cancer': 0.2797, 'Glioma': 0.1227, 'Hepatocellular_carcinoma': 0.1384, 'Chronic_myeloid_leukemia': 0.1844,
    'Endometrial_cancer': 0.2257, 'Melanoma': 0.2492, 'Bladder_cancer': 0.3824, 'Basal_cell_carcinoma': 0.5488
}

# -------------------------------
# 3. Encode Class and Relation columns
# -------------------------------
class_encoder = LabelEncoder()
data['Class'] = class_encoder.fit_transform(data['Class'])

relation_encoder = LabelEncoder()
data['Relation'] = relation_encoder.fit_transform(data['Relation'])

encoded_class_weights = {class_encoder.transform([k])[0]: v for k, v in class_weights.items()}

# -------------------------------
# 4. Tokenize and pad texts
# -------------------------------
tokenizer = Tokenizer()
text_columns = ['Gene1', 'Gene2', 'Gene1_Description', 'Gene2_Description', 'Gene1_Pathways', 'Gene2_Pathways']

for col in text_columns:
    tokenizer.fit_on_texts(data[col])
    data[col] = tokenizer.texts_to_sequences(data[col])

# Maximum length for padding (at least 50)
max_len = max(data[text_columns].apply(lambda col: col.str.len().max()).max(), 50)

# Pad columns and combine into final X matrix
X = np.hstack([pad_sequences(data[col], maxlen=max_len) for col in text_columns])

# -------------------------------
# 5. Split the data
# -------------------------------
y_class = data['Class'].values
y_relation = data['Relation'].values

X_train, X_test, y_train_class, y_test_class, y_train_relation, y_test_relation = train_test_split(
    X, y_class, y_relation, test_size=0.2, random_state=42
)

# Sample weights
sample_weights_class = np.array([encoded_class_weights[cls] for cls in y_train_class])
sample_weights_relation = np.ones_like(y_train_relation)
sample_weights = {'class_output': sample_weights_class, 'relation_output': sample_weights_relation}

# -------------------------------
# 6. Define the model
# -------------------------------
input_layer = Input(shape=(X.shape[1],))

embedding_dim = 128
x = Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=embedding_dim)(input_layer)

for _ in range(5):
    x = LSTM(64, return_sequences=True)(x)
    x = Dropout(0.3)(x)

# Multi-Head Attention
x = MultiHeadAttention(num_heads=4, key_dim=64)(x, x)
x = GlobalAveragePooling1D()(x)

# Class prediction branch
class_branch = Dense(64, activation='relu')(x)
class_branch = Dropout(0.3)(class_branch)
class_output = Dense(len(class_encoder.classes_), activation='softmax', name='class_output')(class_branch)

# Relation prediction branch
relation_output = Dense(len(relation_encoder.classes_), activation='softmax', name='relation_output')(x)

model = Model(inputs=input_layer, outputs=[class_output, relation_output])

# Compile the model
loss_weights = {'class_output': 2.0, 'relation_output': 1.0}
model.compile(
    optimizer='adam',
    loss={'class_output': 'sparse_categorical_crossentropy', 'relation_output': 'sparse_categorical_crossentropy'},
    loss_weights=loss_weights,
    metrics={'class_output': 'accuracy', 'relation_output': 'accuracy'}
)

# -------------------------------
# 7. Set up checkpoints
# -------------------------------
os.makedirs("checkpoints", exist_ok=True)
checkpoint_path = "checkpoints/model_epoch_{epoch:02d}.weights.h5"
checkpoint = ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True,
    save_best_only=False,  # Save all epochs
    verbose=1
)

# -------------------------------
# 8. Train the model (always start from the first epoch)
# -------------------------------
history = model.fit(
    X_train,
    {'class_output': y_train_class, 'relation_output': y_train_relation},
    validation_split=0.1,
    epochs=50,
    initial_epoch=0,  # Start from the first epoch
    batch_size=32,
    sample_weight=sample_weights,
    callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True), checkpoint]
)

# -------------------------------
# 9. Plot the metrics
# -------------------------------
plt.figure(figsize=(12, 6))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['class_output_accuracy'], label='Class Accuracy')
plt.plot(history.history['relation_output_accuracy'], label='Relation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy Metrics')

# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Total Loss')
plt.plot(history.history['class_output_loss'], label='Class Loss')
plt.plot(history.history['relation_output_loss'], label='Relation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss Metrics')

plt.tight_layout()
plt.show()

