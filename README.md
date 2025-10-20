# GLAN: A Pan-Cancer Gene Interaction Atlas for Simultaneous Multi-Cancer Classification

**Hybrid Deep Learning Framework for Multi-Cancer Classification and Gene–Gene Interaction Prediction**

This repository provides the implementation of **GLAN**, a hybrid deep learning model combining **LSTM layers** with a **GAT-inspired attention module** for:  

1. **Cancer type classification**  
2. **Prediction of gene–gene functional relationships**  

GLAN integrates gene identifiers, textual descriptions, and pathway annotations into a unified representation, capturing complex biological dependencies. This framework facilitates accurate modeling of both individual gene features and pairwise interactions, providing a powerful tool for uncovering molecular mechanisms across multiple cancer types.

---

## Dataset

- **File:** `Data_Cancer.csv`  
- **Content:** Gene pairs with associated descriptions, pathways, relations, and cancer types.  
- **Number of classes:** 17 major cancer types were retained, including *Breast cancer, Gastric cancer, Non-small cell lung cancer, Colorectal cancer, Pancreatic cancer, Acute myeloid leukemia, Renal cell carcinoma, Prostate cancer, Small cell lung cancer, Thyroid cancer, Glioma, Hepatocellular carcinoma, Chronic myeloid leukemia, Endometrial cancer, Melanoma, Bladder cancer, and Basal cell carcinoma*.  

**License (Dataset):**  
Creative Commons Attribution 4.0 International (CC BY 4.0)  
- You are free to share, adapt, and build upon this dataset.  
- Proper attribution to the original source must be given.  

---

## Model Architecture


- **Gene Embedding:** Embedding layers for `Gene1` and `Gene2` identifiers  
- **Text Representation:** TF-IDF and tokenized gene descriptions/pathways processed via stacked **5 LSTM layers**  
- **Attention Module:** Multi-Head Attention applied to textual embeddings  
- **Graph Interaction:** GAT-inspired module models pairwise gene interactions with learnable edge weights  
- **Dual Outputs:**  
  - `class_output`: Cancer type prediction  
  - `relation_output`: Gene–gene functional relation prediction  

- **Training Strategy:**  
  - Class-weighted loss to handle imbalance  
  - Early stopping and checkpointing for stable convergence

---

## Results

The performance of the model across two prediction tasks is summarized below:


| Output Type    | Overall Accuracy | Class-weighted Accuracy | F1-Score | Recall | Precision | AUC |
|----------------|----------------|------------------------|----------|--------|-----------|--------|
| Cancer Class   | 64.27%         | 65.06%                 | 0.6560   | 0.6275 | 0.7377    | 0.9557 |
| Relation       | 94.32%         | -                      | 0.8742   | 0.8862 | 0.8754    | 0.9995 |

The results demonstrate **high accuracy in predicting gene–gene relationships** and reliable cancer type classification, highlighting the effectiveness of the hybrid architecture.

---

## Installation & Usage

1. Clone this repository:
   ```bash
   git clone https://github.com/amin-chartab-soltani/PanCancer-Gene-Atlas.git
   cd PanCancer-Gene-Atlas
