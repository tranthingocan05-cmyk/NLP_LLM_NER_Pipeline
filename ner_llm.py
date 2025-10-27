from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import json
# No need to import accelerate; installation is enough.

# The model name you chose
MODEL_NAME = "google/flan-t5-small"
tokenizer = None
model = None # Initialize variable to prevent UnboundLocalError

try:
    print(f"Loading Tokenizer: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    print(f"Loading Model: {MODEL_NAME}...")
    # Use device_map="cpu" after installing 'accelerate'
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME, 
        device_map="cpu"  # Requires the accelerate library
    )
    print("Model loaded successfully.")
    
except Exception as e:
    # If an error occurs, it might be due to missing 'accelerate' or another issue
    print(f"\nFATAL ERROR: Could not load model or tokenizer.")
    print(f"Details: {e}")
    # Ensure you have installed 'accelerate' if using device_map!
    
# Continue with generate_prompt and extract_flan_t5_entities functions
# ...
# -----------------------------------------------------------------

# Function 1: Prompt Engineering Logic
def generate_prompt(text, global_entities, max_context_entities=30):
    # 1. Shorten the entity list (Context Limitation Strategy)
    context_entities_short = [e['name'] for e in global_entities[-max_context_entities:]] 
    
    # 2. Construct the Prompt
    prompt = f"""
    You are a professional technical assistant. Your task is to extract technical named entities 
    (including Parts, Components, Machines, and Functions) from the text below.
    
    Refer to the 'Known List' to determine if an extracted entity is new or known.

    <Known List (Context)>
    {', '.join(context_entities_short)}
    </Known List (Context)>

    <New Text>
    {text.strip()}
    </New Text>

    <Required Output Format>
    Your task is to extract all named entities.
    **THE OUTPUT MUST BE A SINGLE, VALID JSON ARRAY OF OBJECTS, ONLY!**
    
    For each extracted entity, determine its 'status':
    1. **Known**: If it is present in the <Known List (Context)>.
    2. **New**: Otherwise.
    
    Use the following format precisely (array of JSON objects). **NOTE the curly braces {{ }}**:

    [
      {{"name": "M4 bolt", "type": "Parts", "status": "New"}},
      {{"name": "hydraulic pump", "type": "Machine", "status": "New"}},
      {{"name": "system_diag()", "type": "Function", "status": "New"}}
    ]
    </Required Output Format>
    """
    return prompt.strip()


# Function 2: Model Invocation and JSON Parsing Logic
def extract_flan_t5_entities(text, global_entities):
    
    if model is None:
        print("Model not loaded, skipping LLM extraction.")
        return []

    # --- MODEL INVOCATION AND RAW TEXT GENERATION ---
    prompt = generate_prompt(text, global_entities)
    
    # Encode the prompt
    input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids 
    
    # Generate the response
    outputs = model.generate(input_ids, max_new_tokens=256) 
    # Assign the result to result_text
    result_text = tokenizer.decode(outputs[0], skip_special_tokens=True) 
    # -----------------------------------------------------

    # Initialize json_str before the try block for debugging access in except
    json_str = "" 
    
    # Attempt JSON Parsing
    try:
        
        # B1: Extract the JSON array (remove surrounding text)
        start = result_text.find('[')
        end = result_text.rfind(']') + 1
        
        if start == -1 or end <= start:
             # Debug: Print raw output if array structure is not found
             print(f"LLM Output (No array): {result_text}") 
             return []

        json_str = result_text[start:end]
        
        # B2: Attempt to fix common JSON errors (single quotes, trailing commas)
        json_str = json_str.replace("'", '"')
        json_str = json_str.replace(",\n]", "\n]").replace(",]", "]").replace(", }", "}")
        
        # B2.5: Fix common T5 syntax error (missing curly braces)
        
        if json_str.startswith('["name":'):
            # The model generated an invalid array of key-value pairs, not objects: ["k:v", ...]
            
            # 1. Remove outer brackets
            content = json_str.strip('[]').strip()
            
            # 2. Hack: Replace delimiter between entities with proper JSON object closing/opening
            # e.g., '...", "name":...' -> '"}, {"name":...'
            json_str = content.replace('", "name":', '"}, {"name":')
            
            # 3. Add outer array and object braces back
            json_str = '[{' + json_str + '}]'
            
        # B3: Final Parsing attempt on the cleaned/fixed string
        entities_list = json.loads(json_str)
        
        final_entities = []
        for entity in entities_list:
            # B4: Standardize output keys from LLM output
            entity_name = entity.get("name") or entity.get("entity_name") 
            entity_type = entity.get("type") or entity.get("entity_type", "UNKNOWN_LLM")

            if entity_name:
                final_entities.append({
                    "name": entity_name,
                    "type": entity_type,
                    "source_method": "LLM"
                })
        
        return final_entities
        
    except json.JSONDecodeError as e:
        print(f"\n--- LLM JSON DECODE ERROR ---")
        print(f"Error: {e}")
        # Print json_str for debugging the exact string that failed to parse
        print(f"Corrupt JSON String: {json_str[:200]}...") 
        print(f"Raw LLM Output (Full): {result_text}")
        print(f"--- END ERROR ---\n")
        return []
    except Exception as e:
        print(f"General LLM Extraction Error: {e}")
        return []