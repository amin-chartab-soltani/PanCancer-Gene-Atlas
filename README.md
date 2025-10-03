# PanCancer-Gene-Atlas
Dataset and deep learning model for the paper "A Pan-Cancer Gene Interaction Atlas for Simultaneous Multi-Cancer Classification".

This repository provides the implementation of a **multi-task deep learning model** for predicting both **cancer type classification** and **gene–gene relation prediction**. The project integrates gene information (gene names, descriptions, and pathways) into a unified representation using **LSTM layers with attention mechanism**, enabling effective modeling of biological knowledge.

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

- **Embedding layer** for text representation  
- **5 stacked BiLSTM layers** for sequential modeling  
- **Multi-Head Attention** for capturing dependencies  
- **Dual-branch outputs**:
  - *Cancer class prediction*  
  - *Gene–gene relation prediction*  

The model is trained with **class-weighted loss** to handle class imbalance and employs **early stopping and checkpointing** for robust training.

---

## Results

The performance of the model across two prediction tasks is summarized below:

| Relation      | 93.70%          | _                    | 0.9123   | 0.9370 | 0.8989    | 0.9986|
| Output Type   | Overall Accuracy | Class-weighted Accuracy | F1-Score | Recall | Precision | AUC |
|---------------|-----------------|------------------------|----------|--------|-----------|-------|
| Cancer Class  | 62.73%          | 64.66%                 | 0.6386   | 0.6273 | 0.6983    | 0.9442|


---

## Installation & Usage

1. Clone this repository:
   ```bash
   git clone https://github.com/amin-chartab-soltani/PanCancer-Gene-Atlas.git
   cd PanCancer-Gene-Atlas
