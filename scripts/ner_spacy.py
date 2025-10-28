# ner_spacy.py

import spacy
# REMOVE: from texts import TECHNICAL_TEXTS (This file should only define functions)

# Load the installed model (at module level)
try:
    nlp_sm = spacy.load("en_core_web_sm")
except OSError:
    print("ERROR: The spaCy model 'en_core_web_sm' is not downloaded. Please run:")
    print("python -m spacy download en_core_web_sm")
    # If the model fails to load, set nlp_sm = None or exit
    nlp_sm = None


def extract_spacy_entities(text):
    """
    Extracts entities using spaCy and standardizes the output to fit the
    entity management pipeline structure.
    """
    if nlp_sm is None:
        return []

    doc = nlp_sm(text)
    entities = []
    
    # Define relevant labels to avoid too much unnecessary information
    relevant_labels = ["ORG", "PRODUCT", "TIME", "QUANTITY", "CARDINAL", "NORP"]

    for ent in doc.ents:
        if ent.label_ in relevant_labels:
            entity_name = ent.text
            # NOTE: Standardize output keys to match the structure in pipeline_manager.py
            entities.append({
                "name": ent.text,            # Entity Name
                "type": ent.label_,          # Entity Type (from spaCy)
                "source_method": "Classical" # Extraction Method
            })
    return entities

# --- TEST RUN BLOCK ---
# Place your result display code here. 
# Code in this block will NOT run when the file is imported.
if __name__ == "__main__":
    from texts import TECHNICAL_TEXTS # Only import when running this file directly
    
    print("--- Classical NER Results (spaCy) ---")
    for i, text in enumerate(TECHNICAL_TEXTS):
        ents = extract_spacy_entities(text)
        print(f"\n[Text {i+1}]")
        print(f"Text: {text.strip()}")
        
        # Display entities
        if ents:
            for ent in ents:
                print(f"- {ent['name']} ({ent['type']})")
        else:
            print("- No entities found.")
