# Check-COVID Dataset Analysis

## 1. Establishment
The **Check-COVID** dataset is a fact-checking benchmark designed to verify claims about COVID-19 using scientific evidence. 

*   **Source**: The dataset contains **1,504 expert-annotated news claims** about the coronavirus.
*   **Evidence**: These claims are paired with **sentence-level evidence** from scientific journal articles (abstracts).
*   **Annotation**: 
    *   The claims are a mix of **extracted** (written by journalists) and **composed** (written by annotators) text.
    *   Each claim is assigned a veracity label: **SUPPORT**, **REFUTE**, or **NOT ENOUGH INFO**.
    *   Annotators identified specific sentences in the scientific, abstracts that provide the rationale for the label.
*   **Purpose**: The goal is to verify claims written in everyday language (news) against formal academic language (scientific journals).

## 2. Content Structure
The data is primarily located in the `Check-COVID/Check-COVID` directory.

### Claims File (`Check-COVID_all.json`)
This file contains the claims to be verified. Each entry is a JSON object with the following key fields:
*   `id`: Unique identifier for the claim.
*   `claim`: The text of the claim (e.g., "One type of COVID-19 test identifies coronavirus proteins in a few seconds.").
*   `cord_id`: A reference ID linking to the specific scientific article in the corpus.
*   `label`: The veracity label (`SUPPORT`, `REFUTE`, or `NOTENOUGHINFO`).
*   `evidence_set`: An array of objects pointing to the specific sentences in the article abstract that serve as evidence. Each object contains:
    *   `sent_index`: The index of the sentence in the abstract (0-indexed).
    *   `type`: The role of the sentence (`PRIMARY` or `SUPPLEMENTARY`).
*   `is_auth`: Boolean flag (likely indicating if the claim is from an authoritative source or author-verified).

### Corpus File (`corpus.json`)
This file constitutes the knowledge base (scientific articles) against which claims are checked. Each entry contains:
*   `cord_id`: Unique identifier matching the `cord_id` in the claims file.
*   `title`: The title of the scientific article.
*   `abstract`: A list of strings, where each string is a sentence of the article's abstract.

**Example Relationship:**
A claim in `Check-COVID_all.json` refers to a `cord_id` "8vp57c1o". The system looks up "8vp57c1o" in `corpus.json` to retrieve the abstract. If the claim's `evidence_set` has `sent_index: 1`, it means the second sentence of that abstract is the evidence.

## 3. Processing Pipeline
The repository (`fact-checking-system`) implements a pipeline to process this data for training machine learning models (specifically Transformer-based models like RoBERTa). The processing is divided into two main tasks:

### Task A: Rationale Selection
**Goal**: Identify which sentences in a scientific abstract provide evidence for a given claim.
*   **Script**: `fact-checking-system/training/rationale_selection/transformer_covid19_uda.py`
*   **Input**: A pair consisting of the **Claim** text and a single **Sentence** from the abstract.
*   **Process**:
    1.  The system iterates through every claim.
    2.  It retrieves the corresponding abstract from `corpus.json` using `cord_id`.
    3.  It pairs the claim with *each* sentence in the abstract.
    4.  **Labeling**: If the sentence index is present in the claim's `evidence_set`, the pair is labeled as `1` (positive evidence); otherwise, it is `0` (negative).
*   **Model Input Format**: `[CLS] Claim [SEP] Sentence [SEP]` (standard BERT/RoBERTa format).

### Task B: Label Prediction (Veracity)
**Goal**: Determine if the claim is Supported, Refuted, or Not Supported by the evidence.
*   **Script**: `fact-checking-system/training/label_prediction/transformer_covid19_uda.py`
*   **Input**: A pair consisting of the **Claim** and the **Concatenated Rationale** (evidence sentences).
*   **Process**:
    1.  The system iterates through every claim.
    2.  It collects all sentences from the abstract identified as evidence (based on `sent_index` in `evidence_set`).
    3.  These sentences are concatenated into a single "Rationale" string.
    4.  **Labeling**: The sample is assigned the label ID corresponding to `SUPPORT` (2), `NOTENOUGHINFO` (1), or `REFUTE` (0).
*   **Model Input Format**: `[CLS] Rationale [SEP] Claim [SEP]`.

### Unsupervised Data Augmentation (UDA)
The scripts also contain logic for **UDA**, utilizing "back-translated" data to augment the training set. This involves translating claims to another language and back to English to create paraphrased variations, making the model more robust.
