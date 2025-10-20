# ============================================
# Hybrid Deep Model (LSTM + GAT-like) for Gene Pair Classification
# ============================================

# --- Silence warnings ---
import os, warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'   # Hide TF info/warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Embedding, LSTM, Dense, Dropout, MultiHeadAttention,
    GlobalAveragePooling1D, Concatenate, Flatten, LeakyReLU, Lambda
)
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# --- Load and filter dataset ---
data = pd.read_csv(r'D:\Outputs\Cleaned_Data_Cancers.csv')
classes_to_keep = [
    'Breast_cancer', 'Gastric_cancer', 'Non-small_cell_lung_cancer',
    'Colorectal_cancer', 'Pancreatic_cancer', 'Acute_myeloid_leukemia',
    'Renal_cell_carcinoma', 'Prostate_cancer', 'Small_cell_lung_cancer',
    'Thyroid_cancer', 'Glioma', 'Hepatocellular_carcinoma',
    'Chronic_myeloid_leukemia', 'Endometrial_cancer', 'Melanoma',
    'Bladder_cancer', 'Basal_cell_carcinoma'
]
data = data[data['Class'].isin(classes_to_keep)].reset_index(drop=True)

# --- Class weighting ---
class_weights = {
    'Breast_cancer': 0.0977, 'Non-small_cell_lung_cancer': 0.0902,
    'Acute_myeloid_leukemia': 0.088, 'Pancreatic_cancer': 0.0872,
    'Gastric_cancer': 0.0791, 'Colorectal_cancer': 0.0784,
    'Hepatocellular_carcinoma': 0.0601, 'Prostate_cancer': 0.0564,
    'Chronic_myeloid_leukemia': 0.0561, 'Glioma': 0.0525,
    'Renal_cell_carcinoma': 0.0492, 'Endometrial_cancer': 0.0435,
    'Small_cell_lung_cancer': 0.0426, 'Melanoma': 0.0413,
    'Thyroid_cancer': 0.0288, 'Bladder_cancer': 0.0268,
    'Basal_cell_carcinoma': 0.0221,
}

# --- Label encoding ---
class_encoder = LabelEncoder()
relation_encoder = LabelEncoder()
data['Class'] = class_encoder.fit_transform(data['Class'])
data['Relation'] = relation_encoder.fit_transform(data['Relation'])
encoded_class_weights = {class_encoder.transform([k])[0]: v for k, v in class_weights.items()}

# --- TF-IDF representation & pairwise similarity ---
data['Gene1_text_full'] = data['Gene1_Description'].fillna('') + ' ' + data['Gene1_Pathways'].fillna('')
data['Gene2_text_full'] = data['Gene2_Description'].fillna('') + ' ' + data['Gene2_Pathways'].fillna('')
tfidf = TfidfVectorizer(max_features=5000)
all_texts = np.concatenate([data['Gene1_text_full'].values, data['Gene2_text_full'].values])
tfidf.fit(all_texts)
tfidf_g1 = tfidf.transform(data['Gene1_text_full'].values)
tfidf_g2 = tfidf.transform(data['Gene2_text_full'].values)
inner = tfidf_g1.multiply(tfidf_g2).sum(axis=1).A1
g1_sq = tfidf_g1.multiply(tfidf_g1).sum(axis=1).A1
g2_sq = tfidf_g2.multiply(tfidf_g2).sum(axis=1).A1
eps = 1e-9
sim_diag = np.clip(inner / (np.sqrt(g1_sq + eps) * np.sqrt(g2_sq + eps)), 0.0, 1.0)
X_adj = sim_diag.reshape(-1, 1)

# --- Tokenization & sequence encoding ---
text_columns = ['Gene1_Description', 'Gene2_Description', 'Gene1_Pathways', 'Gene2_Pathways']
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(data[text_columns].apply(lambda x: ' '.join(x), axis=1))
max_len = max(data[text_columns].apply(lambda col: col.str.len().max()).max(), 50)
vocab_size = len(tokenizer.word_index) + 1
seqs = [pad_sequences(tokenizer.texts_to_sequences(data[col]), maxlen=max_len) for col in text_columns]
X_text = np.hstack(seqs)

# --- Encode genes ---
gene_encoder_1 = LabelEncoder()
gene_encoder_2 = LabelEncoder()
data['Gene1_enc'] = gene_encoder_1.fit_transform(data['Gene1'])
data['Gene2_enc'] = gene_encoder_2.fit_transform(data['Gene2'])
X_gene1 = data['Gene1_enc'].values
X_gene2 = data['Gene2_enc'].values
y_class = data['Class'].values
y_relation = data['Relation'].values

# --- Train/test split ---
X_train_g1, X_test_g1, X_train_g2, X_test_g2, X_train_text, X_test_text, X_train_adj, X_test_adj, y_train_class, y_test_class, y_train_relation, y_test_relation = train_test_split(
    X_gene1, X_gene2, X_text, X_adj, y_class, y_relation, test_size=0.2, random_state=42
)

# --- Sample weights ---
sample_weights_class = np.array([encoded_class_weights[c] for c in y_train_class])
sample_weights_relation = np.ones_like(y_train_relation)
sample_weights = {'class_output': sample_weights_class, 'relation_output': sample_weights_relation}

# --- Model architecture (Hybrid LSTM + GAT-like) ---
gene1_input = Input(shape=(1,), dtype='int32', name='gene1_input')
gene2_input = Input(shape=(1,), dtype='int32', name='gene2_input')
text_input = Input(shape=(X_text.shape[1],), dtype='int32', name='text_input')
adj_input = Input(shape=(1,), dtype='float32', name='adj_input')

gene_embedding_dim = 32
g1_emb = Embedding(len(gene_encoder_1.classes_) + 1, gene_embedding_dim)(gene1_input)
g2_emb = Embedding(len(gene_encoder_2.classes_) + 1, gene_embedding_dim)(gene2_input)
g1_flat, g2_flat = Flatten()(g1_emb), Flatten()(g2_emb)

text_embed = Embedding(vocab_size, 128)(text_input)
x = text_embed
for i in range(5):
    x = LSTM(64, return_sequences=True)(x)
    x = Dropout(0.3)(x)
x = MultiHeadAttention(num_heads=4, key_dim=64)(x, x)
x = GlobalAveragePooling1D()(x)

split_point = 2 * max_len
slice_g1 = Lambda(lambda t: t[:, :split_point, :])(text_embed)
slice_g2 = Lambda(lambda t: t[:, split_point:, :])(text_embed)
g1_text_pool = GlobalAveragePooling1D()(slice_g1)
g2_text_pool = GlobalAveragePooling1D()(slice_g2)
g1_node_feat = Concatenate()([g1_flat, g1_text_pool])
g2_node_feat = Concatenate()([g2_flat, g2_text_pool])

proj_dim = 64
W_node = Dense(proj_dim)
g1_proj, g2_proj = W_node(g1_node_feat), W_node(g2_node_feat)

att_dense = Dense(1)
e12 = LeakyReLU(0.2)(att_dense(Concatenate()([g1_proj, g2_proj])))
e21 = LeakyReLU(0.2)(att_dense(Concatenate()([g2_proj, g1_proj])))

edge_input = Concatenate()([adj_input, g1_text_pool, g2_text_pool, g1_proj, g2_proj])
edge_dense = Dense(32, activation='relu')(edge_input)
edge_sig = tf.keras.activations.sigmoid(Dense(1)(edge_dense))
edge_weight = Lambda(lambda z: 0.1 + 0.9 * z)(edge_sig)

def softmax_pair(a, b):
    stacked = tf.concat([a, b], axis=1)
    sm = tf.nn.softmax(stacked, axis=1)
    return tf.expand_dims(sm[:, 0], axis=1)

e12_w = Lambda(lambda args: args[0] * args[1])([e12, edge_weight])
e21_w = Lambda(lambda args: args[0] * args[1])([e21, edge_weight])
alpha12 = Lambda(lambda t: softmax_pair(t[0], t[1]))([e12_w, e21_w])
alpha21 = Lambda(lambda t: softmax_pair(t[0], t[1]))([e21_w, e12_w])

m1 = Lambda(lambda args: args[0] * args[1])([g2_proj, alpha12])
m2 = Lambda(lambda args: args[0] * args[1])([g1_proj, alpha21])
g_update_dense = Dense(128, activation='relu')
g1_final = g_update_dense(Concatenate()([g1_proj, m1]))
g2_final = g_update_dense(Concatenate()([g2_proj, m2]))

graph_repr = Concatenate()([g1_final, g2_final])
graph_repr = Dropout(0.3)(Dense(128, activation='relu')(graph_repr))
combined = Dropout(0.4)(Dense(128, activation='relu')(Concatenate()([x, graph_repr])))

class_output = Dense(len(class_encoder.classes_), activation='softmax', name='class_output')(combined)
relation_output = Dense(len(relation_encoder.classes_), activation='softmax', name='relation_output')(combined)

model = Model(inputs=[gene1_input, gene2_input, text_input, adj_input],
              outputs=[class_output, relation_output])

# --- Compile & train ---
model.compile(
    optimizer='adam',
    loss={'class_output': 'sparse_categorical_crossentropy', 'relation_output': 'sparse_categorical_crossentropy'},
    loss_weights={'class_output': 8.0, 'relation_output': 1.0},
    metrics=['accuracy'],
    weighted_metrics=[]
)

model.summary()

history = model.fit(
    {'gene1_input': X_train_g1, 'gene2_input': X_train_g2, 'text_input': X_train_text, 'adj_input': X_train_adj},
    {'class_output': y_train_class, 'relation_output': y_train_relation},
    validation_split=0.1,
    epochs=70,
    batch_size=32,
    sample_weight=sample_weights,
    callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)],
    verbose=1
)

# --- Plot metrics ---
plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.plot(history.history.get('class_output_accuracy', []), label='Class Acc')
plt.plot(history.history.get('relation_output_accuracy', []), label='Relation Acc')
plt.legend(); plt.title('Accuracy')

plt.subplot(1,2,2)
plt.plot(history.history.get('class_output_loss', []), label='Class Loss')
plt.plot(history.history.get('relation_output_loss', []), label='Relation Loss')
plt.legend(); plt.title('Loss')
plt.tight_layout()
plt.show()

# --- Evaluate on test data ---
test_results = model.evaluate(
    {'gene1_input': X_test_g1, 'gene2_input': X_test_g2, 'text_input': X_test_text, 'adj_input': X_test_adj},
    {'class_output': y_test_class, 'relation_output': y_test_relation},
    verbose=1
)

print("\n🔹 Test Results:")
for name, val in zip(model.metrics_names, test_results):
    print(f"{name}: {val:.4f}")


# In[ ]:




