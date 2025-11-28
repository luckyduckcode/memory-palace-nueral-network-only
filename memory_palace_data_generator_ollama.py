import numpy as np
import pandas as pd
import json
import time
import requests
import random

# --- 1. CONFIGURATION ---
OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "tinyllama"  # Very small, memory-efficient model
EMBEDDING_DIM = 768
CUBE_SIZE = 8

# --- 2. THE CHESS CUBE GEOMETRIC LATTICE (Blueprint) ---
class ChessCubeLattice:
    """Rigorous definition of the 8x8x8 Memory Palace structure."""
    def __init__(self, size=CUBE_SIZE):
        self.size = size
        self.locations = self._generate_coordinates()

    def _generate_coordinates(self):
        locations = {}
        for x in range(1, self.size + 1):
            for y in range(1, self.size + 1):
                for z in range(1, self.size + 1):
                    cell_key = f"{x}_{y}_{z}"
                    locations[cell_key] = {
                        'x': x, 'y': y, 'z': z,
                        'color_parity': (x + y + z) % 2, 
                        'loc_prefix_mapped': None
                    }
        return locations

# --- 3. OLLAMA API CALL FUNCTIONS ---

def call_ollama_api(prompt):
    """Calls local Ollama API."""
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "format": "json"
    }
    
    try:
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()
        return json.loads(result['response'])
    except Exception as e:
        print(f"Ollama API call failed: {e}")
        return None

def generate_loc_and_pao(fact_text, loc_prefixes_str):
    """
    Uses Ollama to generate classification and PAO mnemonic.
    """
    prompt = f"""You are an expert mnemonic generator and Library of Congress cataloger.

Fact: "{fact_text}"

Available LOC categories: {loc_prefixes_str}

Generate a JSON response with:
1. "loc_prefix": Choose the most appropriate 3-digit LOC code from the list
2. "pao_mnemonic": Create a vivid, absurd PAO (Person-Action-Object) sentence where the fact is the Object

Example format:
{{"loc_prefix": "510", "pao_mnemonic": "Albert Einstein (Person) is JUGGLING (Action) a glowing matrix equation (Object)."}}

Your response (JSON only):"""

    return call_ollama_api(prompt)

# --- 4. DATA GENERATION PIPELINE ---

def generate_training_data(raw_facts, loc_map):
    """Orchestrates data generation using Ollama."""
    palace = ChessCubeLattice()
    loc_prefixes = list(loc_map.keys())
    loc_file_mapping = {loc: i + 1 for i, loc in enumerate(loc_prefixes)}
    
    print("Starting structured data generation with Ollama...")
    final_dataset = []
    
    for i, fact in enumerate(raw_facts):
        print(f"Processing Fact {i+1}/{len(raw_facts)}: {fact[:30]}...")
        
        llm_output = generate_loc_and_pao(fact, ", ".join(loc_prefixes))
        
        if not llm_output:
            print(f"Skipping fact due to API failure.")
            continue
            
        loc_prefix = llm_output.get('loc_prefix')
        pao_mnemonic = llm_output.get('pao_mnemonic')
        
        if not loc_prefix or not pao_mnemonic or loc_prefix not in loc_map:
            print(f"Skipping fact: Invalid response: {llm_output}")
            continue

        target_file_x = loc_file_mapping[loc_prefix]
        available_locs = [k for k, v in palace.locations.items() 
                         if v['loc_prefix_mapped'] is None and v['x'] == target_file_x]
        
        if not available_locs:
            print(f"Skipping fact: File X={target_file_x} is full.")
            continue
            
        available_locs.sort(key=lambda k: (palace.locations[k]['z'], palace.locations[k]['y']))
        cell_key = available_locs[0]
        cell_details = palace.locations[cell_key]
        palace.locations[cell_key]['loc_prefix_mapped'] = loc_prefix
        
        final_dataset.append({
            'fact_text': fact,
            'fact_length': len(fact),
            'loc_prefix': loc_prefix,
            'pao_mnemonic': pao_mnemonic,
            'x_coord': cell_details['x'],
            'y_coord': cell_details['y'],
            'z_coord': cell_details['z'],
            'color_parity': cell_details['color_parity'],
            'semantic_vector_dummy': f"vector_{i}" 
        })
        
        if len(final_dataset) >= CUBE_SIZE**3: 
             break

    return pd.DataFrame(final_dataset)

# --- 5. EXECUTION ---

if __name__ == "__main__":
    LOC_MAPPING = {
        '510': 'Mathematics',
        '530': 'Physics',
        '610': 'Medicine/Health',
        '940': 'European History',
        '820': 'English Literature',
        '330': 'Economics',
        '700': 'Arts',
        '100': 'Philosophy'
    }

    RAW_FACTS = [
        "The inverse of a matrix is calculated using the adjoint and the determinant.",
        "Albert Einstein published his theory of general relativity in 1915.",
        "The primary cause of World War I was the assassination of Archduke Franz Ferdinand in 1914.",
    ]

    print("--- Generative Mnemonic Data Generator (Ollama) ---")
    df = generate_training_data(RAW_FACTS, LOC_MAPPING)
    
    print("\n--- FINAL GENERATED TRAINING DATA SAMPLE ---")
    if not df.empty:
        print(df[['fact_text', 'loc_prefix', 'pao_mnemonic', 'x_coord', 'y_coord', 'z_coord', 'color_parity']].head())
        
        output_file = "memory_palace_training_data.csv"
        df.to_csv(output_file, index=False)
        print(f"\nData saved to {output_file}")
    else:
        print("\nError: No data was generated. Make sure Ollama is running: 'ollama serve'")
