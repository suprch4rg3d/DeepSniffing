# [MAI642-DL] Semester Project Proposal Man

## Project Title

**Olfactory Computation: Predicting Odor Profiles from 1D Molecular Representations via Sequence-Based Deep Learning**

## 1. Abstract & Objective

While computer vision and audio processing have seen massive breakthroughs in deep learning, the digitization of smell remains a largely unexplored frontier. The objective of this project is to build a deep learning model capable of predicting the human-perceived odor profile (e.g., "fruity," "musky," "minty") of a chemical compound based purely on its molecular structure.

To ensure feasibility within a single semester, this project bypasses complex 3D molecular physics and Graph Neural Networks (GNNs). Instead, it treats chemistry as a language problem by utilizing SMILES (Simplified Molecular-Input Line-Entry System) strings. By transforming 3D molecules into 1D text strings, the project will leverage established Natural Language Processing (NLP) architectures to classify olfactory profiles.

## 2. Background & Motivation

The fragrance, food, and chemical industries currently rely on slow, expensive human testing to identify the scent of newly synthesized molecules. Creating a computational model that can accurately predict scent profiles from a structural formula would drastically accelerate the discovery of new flavors and fragrances. By reframing molecular structure as sequential text data, this project applies state-of-the-art deep learning text classification techniques to a novel biochemical domain.

## 3. Methodology

### 3.1. Dataset Selection

The project will utilize the open-source **DREAM Olfaction Prediction Challenge dataset** (or alternatively, the **Pyrfume** database).

- **Input Data:** Molecular structures represented as SMILES strings (e.g., `CC(=O)OC1=CC=CC=C1C(=O)O`).
- **Target Labels:** Multi-label human perceptual descriptors (e.g., sweet, earthy, citrus).

### 3.2. Data Preprocessing

- **Tokenization:** SMILES strings will be parsed at the character level (or using specialized SMILES byte-pair encoding) so the neural network can process the atoms and bonds as sequential tokens.
- **Class Balancing:** Olfactory datasets are often heavily skewed (e.g., many molecules smell "sweet," few smell "fishy"). Techniques like SMOTE or class-weighted loss functions will be applied.

### 3.3. Model Architecture

The project will experiment with and benchmark 1-2 sequence-based architectures commonly used in NLP:

1.  **1D Convolutional Neural Networks (1D CNN):** Fast to train and effective at finding local structural motifs (like a specific benzene ring that triggers a specific smell).
2.  **Recurrent Networks (LSTM / GRU) or Small Transformers:** Capable of understanding the entire context of the SMILES sequence, ensuring long-range chemical dependencies are factored into the odor prediction.

## 4. Scope and Limitations (Semester Constraints)

To guarantee completion and rigorous evaluation within the 3-4 month semester timeline, the project scope is explicitly constrained:

- **Label Limitation:** The model will not attempt to predict all 146+ possible odor descriptors in the dataset. It will be framed as a multi-label classification task predicting only the **Top 5 to 10 most prominent scent categories** (e.g., Floral, Musky, Earthy, Minty, Citrus).
- **Input Limitation:** The model will rely strictly on SMILES strings, intentionally ignoring 3D conformational data to keep computational overhead low enough to train on a standard consumer GPU or cloud instance (e.g., Google Colab).

## 5. Expected Deliverables

1.  **A Cleaned Dataset Pipeline:** Scripts to tokenize and format SMILES data for deep learning.
2.  **Trained Model Weights:** A functional model capable of accepting a novel SMILES string and outputting probability scores for the top scent categories.
3.  **Performance Analysis:** An evaluation of the model using standard metrics (F1-score, Precision, Recall, and ROC-AUC) to determine how well AI can bridge the gap between chemistry and human perception.
4.  **Final Academic Report:** Documenting the architecture, training process, and results.

## 6. Proposed Timeline (14 Weeks) - Να χαμε να λέγαμεν

- **Weeks 1-3:** Literature review, downloading datasets (DREAM/Pyrfume), and finalizing the top 5-10 target odor labels.
- **Weeks 4-6:** Data preprocessing, standardizing SMILES strings, and building the tokenization pipeline.
- **Weeks 7-9:** Constructing and training the baseline model (e.g., 1D CNN).
- **Weeks 10-11:** Hyperparameter tuning, experimenting with an advanced architecture (LSTM/Transformer), and evaluating validation metrics.
- **Weeks 12-14:** Compiling results, generating visualizations (e.g., confusion matrices), and writing the final project report.
