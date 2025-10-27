# Technical NER Pipeline with Classical NLP and LLMs

This project implements a Natural Language Processing (NLP) pipeline designed to perform **Named Entity Recognition (NER)** on technical instruction texts. The main objective is to compare the methods of **Classical NLP (SpaCy)** and **Large Language Models (Flan-T5)**, while addressing the core challenge of Global Entity Management.

---

## 1. Approach and Tool Choices 

| Method | Main Tool | Implementation Style |
| :--- | :--- | :--- |
| **Classical NLP** | **SpaCy** (`en_core_web_sm`) | Extracts general entities and provides baseline processing speed. |
| **LLM-based** | **Flan-T5-Small** (Hugging Face) | Uses a small model, deployed locally, via **Prompt Engineering** for structured extraction. |
| **Entity Management** | `update_global_list()` Logic | Maintains a global list, using a normalization function to **link duplicate entities**. |

---

## 2. Handling Context Limitation (Core Challenge)

The core requirement is to provide the evolving entity list as context to the LLM.

* **The Issue:** The growing `global_entities` list could exceed the **Context Window** of the small Flan-T5 model.
* **Implementation Strategy:** In the `generate_prompt` function, we implemented a truncation mechanism:
    * Only the **30 most recent entities** (using `global_entities[-30:]` slicing) from the global list are passed into the prompt as context.
* **Effectiveness:** This strategy ensures the prompt size is controlled while still providing the most relevant recent entities to support the model in determining if an entity is **Known** or **New**.

---

## 3. Comparison Results and Key Insights 

### 3.1. Quantitative Evaluation (P, R, F1-Score)

The table below presents the comparison of extraction quality (Precision, Recall, F1-Score) based on the manually labeled Ground Truth.

| Text | SpaCy F1-Score (P/R) | Flan-T5 F1-Score (P/R) |
| :---: | :--- | :--- |
| **Text 1** | **0.0000** (P: 0.0000, R: 0.0000) | **0.2857** (P: 0.3333, R: 0.2500) |
| **Text 2** | **0.4000** (P: 1.0000, R: 0.2500) | **0.0000** (P: 0.0000, R: 0.0000) |
| **Text 3** | **0.0000** (P: 0.0000, R: 0.0000) | **0.8000** (P: 0.6667, R: 1.0000) |
| **Text 4** | **0.6667** (P: 1.0000, R: 0.5000) | **0.0000** (P: 0.0000, R: 0.0000) |
| **Text 5** | **0.0000** (P: 0.0000, R: 0.0000) | **0.0000** (P: 0.0000, R: 0.0000) |

---

### 3.2. Key Insights

* **Speed and Performance:** **SpaCy is the fastest solution** ($\sim 0.008$ seconds/text) and should be prioritized for high-performance, general entity processing. **Flan-T5 is significantly slower** ($\sim 2$ seconds/text) due to complex inference.
* **Customization (Flexibility):** **Flan-T5** allows for greater semantic control (as seen in Text 3, achieving the highest F1-Score). Using Prompt Engineering, it successfully extracts **custom entity types** (`Function`, `Machine`).
* **Stability:** SpaCy is more stable for standard labels. **Flan-T5 is prone to JSON parsing errors**, leading to several `F1: 0` instances when the model failed to generate syntactically correct output.
* **Entity Management Success:** The `update_global_list` logic worked correctly. LLM entities (e.g., `"M4 bolt"`, `"system_diag()"`) were successfully **linked** and counted (`"count": 5`) instead of creating duplicates.

---

## 4. Usage Guide

To run this pipeline, please follow these steps:

### 4.1. Environment Setup

1.  **Navigate to Project Directory:**
    ```bash
    cd ner_technical_project
    ```

2.  **Create and Activate Virtual Environment (venv):**
    * *Using `venv` (Standard Virtual Environment):*
    ```bash
    python -m venv venv
    # Activation on Windows (PowerShell):
    # .venv/Scripts/Activate.ps1
    # Activation on Linux/macOS/Git Bash:
    # source venv/bin/activate
    ```

### 4.2. Install Dependencies

After activating the virtual environment, install all dependencies (torch, transformers, accelerate, and spacy) from the `requirements.txt` file:

```bash
pip install -r requirements.txt