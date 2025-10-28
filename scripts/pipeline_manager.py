# pipeline_manager.py

from texts import TECHNICAL_DATA
from ner_spacy import extract_spacy_entities
from ner_llm import extract_flan_t5_entities # Assume function is defined here
import time
import json
import re
import matplotlib.pyplot as plt
import numpy as np

def clean_text(text):
    """
    Performs basic text preprocessing steps:
    1. Removes newline and tab characters (\n, \t).
    2. Standardizes whitespace.
    3. Removes ASCII control characters.
    """
    # 1. Remove newlines and tabs
    text = text.replace('\n', ' ').replace('\t', ' ')
    
    # 2. Remove ASCII control characters (e.g., null characters)
    text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', text) 
    
    # 3. Standardize excessive whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # (Optional: Add stopwords removal here if needed)
    
    return text

# Utility function for standardization and entity updating (Core Challenge Logic)
def normalize_entity(name):
    """Standardizes entity name for comparison."""
    return name.lower().strip().replace('.', '').replace('-', ' ') # Handles hyphens

def normalize_entity_strict(name):
    """
    Standardizes entity name for strict comparison against Ground Truth.
    (Lower case, strip, remove periods and parentheses).
    """
    return name.lower().strip().replace('.', '').replace('-', ' ').replace('()', '')

def calculate_metrics(extracted_entities, gt_entities):
    """
    Calculates Precision, Recall, and F1-Score based on set comparison.
    """
    
    # 1. Standardize and get sets of GT and Extracted entities
    # Use sets to ensure each entity is counted only once
    gt_set = {normalize_entity_strict(e['name']) for e in gt_entities}
    extracted_set = {normalize_entity_strict(e['name']) for e in extracted_entities}
    
    # 2. Calculate components:
    # TP (True Positives): Correctly extracted entities (match GT)
    tp = len(gt_set.intersection(extracted_set))
    
    # FP (False Positives): Incorrectly extracted entities (in extracted, not in GT)
    fp = len(extracted_set) - tp
    
    # FN (False Negatives): Missed entities (in GT, not in extracted)
    fn = len(gt_set) - tp
    
    # 3. Calculate Metrics:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "P": round(precision, 4),
        "R": round(recall, 4),
        "F1": round(f1_score, 4),
        "TP": tp,
        "FP": fp,
        "FN": fn
    }

def update_global_list(new_entities, global_list):
    """Updates the global list based on newly extracted entities."""
    
    # Create a map of normalized names for quick lookup
    # Use list comprehension to create map from global_list:
    normalized_map = {normalize_entity(e['name']): e for e in global_list}
    
    for entity in new_entities:
        # Ignore entities without a name or empty names
        if not entity.get('name'):
            continue
            
        norm_name = normalize_entity(entity['name'])
        source = entity['source_method']
        
        if norm_name in normalized_map:
            # Case 1: Entity exists -> Link
            existing_entity = normalized_map[norm_name]
            existing_entity['count'] = existing_entity.get('count', 0) + 1
            if source not in existing_entity['source_methods']:
                existing_entity['source_methods'].append(source)
                
        else:
            # Case 2: New Entity -> Add
            new_entry = {
                "name": entity['name'],
                "type": entity.get('type', 'UNKNOWN'),
                "count": 1,
                "source_methods": [source]
            }
            global_list.append(new_entry)
            normalized_map[norm_name] = new_entry # Update map 

def plot_evaluation_results(evaluation_log):
    # 1. CALCULATE GLOBAL AVERAGES
    
    # Get all F1-Scores and Speeds from the log
    all_classical_f1 = [log['classical_metrics']['F1'] for log in evaluation_log]
    all_llm_f1 = [log['llm_metrics']['F1'] for log in evaluation_log]
    all_classical_speed = [log['classical_speed_s'] for log in evaluation_log]
    all_llm_speed = [log['llm_speed_s'] for log in evaluation_log]

    # Calculate global averages
    avg_classical_f1 = np.mean(all_classical_f1)
    avg_llm_f1 = np.mean(all_llm_f1)
    
    avg_classical_speed = np.mean(all_classical_speed)
    avg_llm_speed = np.mean(all_llm_speed)

    methods = ['SpaCy (Classical)', 'Flan-T5 (LLM)']
    
    # =======================================================
    # CHART 1: AVERAGE F1-SCORE COMPARISON
    # =======================================================
    f1_scores = [avg_classical_f1, avg_llm_f1]
    
    plt.figure(figsize=(8, 5))
    plt.bar(methods, f1_scores, color=['skyblue', 'lightcoral'])
    plt.title('Average F1-Score Comparison Across All Texts')
    plt.ylabel('Average F1-Score')
    plt.ylim(0, 1.0) # Set F1 limit from 0 to 1
    
    # Add value labels to bars
    for i, score in enumerate(f1_scores):
        plt.text(i, score + 0.05, f'{score:.4f}', ha='center', va='bottom')
        
    plt.savefig('f1_score_comparison.png')
    plt.close()

    print("\n[Visualization]: Saved F1-Score comparison to f1_score_comparison.png")

    # =======================================================
    # CHART 2: AVERAGE PROCESSING SPEED COMPARISON (Log Scale)
    # =======================================================
    speeds = [avg_classical_speed, avg_llm_speed]
    
    plt.figure(figsize=(8, 5))
    # Use log scale due to huge speed difference (0.00x vs 2.x)
    plt.bar(methods, speeds, color=['lightgreen', 'darkorange']) 
    plt.title('Average Processing Speed Comparison (Seconds)')
    plt.ylabel('Average Speed (seconds - LOG SCALE)')
    plt.yscale('log') # Essential to display both values together
    
    # Add value labels to bars
    for i, speed in enumerate(speeds):
        plt.text(i, speed * 1.5, f'{speed:.4f}s', ha='center', va='bottom')
        
    plt.savefig('speed_comparison.png')
    plt.close()

    print("[Visualization]: Saved Speed comparison to speed_comparison.png")


def run_pipeline():
    global_entities = [] # Global list
    evaluation_log = [] # Log for speed and result recording

    # RUN THE SINGLE LOOP
    for i, data in enumerate(TECHNICAL_DATA):
        
        # 1. Get Raw Data and Ground Truth (GT)
        text_raw = data['text']
        gt_entities = data['gt_entities'] # <-- Get Ground Truth for metric calculation

        # 2. PREPROCESSING: Apply clean_text to the raw text
        text = clean_text(text_raw)
        
        print(f"\n--- Processing Text {i+1}/{len(TECHNICAL_DATA)} ---")
        
        # --- 3. Classical NER (SpaCy) ---
        start_time_classical = time.time()
        classical_entities = extract_spacy_entities(text)
        time_classical = time.time() - start_time_classical
        
        # Calculate Metrics for SpaCy
        classical_metrics = calculate_metrics(classical_entities, gt_entities)

        # --- 4. LLM-based (Flan-T5) ---
        # PASS THE GLOBAL LIST FOR CONTEXT
        start_time_llm = time.time()
        llm_entities = extract_flan_t5_entities(text, global_entities)
        time_llm = time.time() - start_time_llm

        # Calculate Metrics for LLM
        llm_metrics = calculate_metrics(llm_entities, gt_entities)

        # --- 5. Entity Management ---
        # Update the global list with results from both methods
        update_global_list(classical_entities, global_entities)
        update_global_list(llm_entities, global_entities)
        
        # --- 6. Record Evaluation Log ---
        evaluation_log.append({
            "text_id": i + 1,
            "text": text,
            "classical_speed_s": round(time_classical, 4),
            "llm_speed_s": round(time_llm, 4),
            "classical_entities_count": len(classical_entities),
            "llm_entities_count": len(llm_entities),
            
            # STORE METRICS
            "classical_metrics": classical_metrics,
            "llm_metrics": llm_metrics,
            
            "global_entity_count": len(global_entities)
        })

    # --- Final Evaluation Summary ---
    print("\n===================================")
    print("      FINAL EVALUATION SUMMARY     ")
    print("===================================")
    
    # Print global list
    print("\n[GLOBAL ENTITY LIST (Top 10):]")
    print(json.dumps(global_entities[:10], indent=2))
    
    # Print evaluation log (speed, entity count)
    print("\n[PER-TEXT EVALUATION LOG (Metrics):]")
    for log in evaluation_log:
        # Retrieve F1, P, R from the stored metrics dictionary
        classical_f1 = log['classical_metrics']['F1']
        classical_p = log['classical_metrics']['P']
        classical_r = log['classical_metrics']['R']

        llm_f1 = log['llm_metrics']['F1']
        llm_p = log['llm_metrics']['P']
        llm_r = log['llm_metrics']['R']

        print(f"Text {log['text_id']} | SpaCy F1: {classical_f1} (P:{classical_p}, R:{classical_r}) | Flan-T5 F1: {llm_f1} (P:{llm_p}, R:{llm_r}) | Global: {log['global_entity_count']}")

    # INVOKE PLOTTING FUNCTION
    plot_evaluation_results(evaluation_log) 
    
if __name__ == "__main__":
    # Loading the Flan-T5 model in ner_llm.py may take a moment
    print("Starting NER Pipeline...")
    run_pipeline()