import numpy as np
import pandas as pd
import json
import time
import requests
import random
import os

# --- 1. CONFIGURATION ---
# IMPORTANT: Set your Gemini API Key in the environment variable 'GEMINI_API_KEY'
API_KEY = os.environ.get("GEMINI_API_KEY", "") 
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent?key={API_KEY}"
EMBEDDING_DIM = 768 # Standard size for modern embedding models
CUBE_SIZE = 8       # 8x8x8 Chess Cube = 512 locations

# --- 2. THE CHESS CUBE GEOMETRIC LATTICE (Blueprint) ---
class ChessCubeLattice:
    """Rigorous definition of the 8x8x8 Memory Palace structure."""
    def __init__(self, size=CUBE_SIZE):
        self.size = size
        self.locations = self._generate_coordinates()

    def _generate_coordinates(self):
        locations = {}
        for x in range(1, self.size + 1):  # File (A-H)
            for y in range(1, self.size + 1):  # Rank (1-8)
                for z in range(1, self.size + 1):  # Height (Floor 1-8)
                    cell_key = f"{x}_{y}_{z}"
                    locations[cell_key] = {
                        'x': x, 'y': y, 'z': z,
                        # Parity is crucial for Separation Loss (Alternating Property)
                        'color_parity': (x + y + z) % 2, 
                        'loc_prefix_mapped': None # To be mapped later
                    }
        return locations

# --- 3. API CALL FUNCTIONS ---

def call_gemini_api(payload):
    """Handles API calls with retry logic."""
    if not API_KEY:
        print("Error: GEMINI_API_KEY environment variable not set.")
        return None

    headers = {'Content-Type': 'application/json'}
    max_retries = 3
    delay = 1  # 1 second initial delay
    
    for i in range(max_retries):
        try:
            response = requests.post(GEMINI_API_URL, headers=headers, json=payload)
            response.raise_for_status() # Raises an exception for 4xx/5xx errors
            
            result = response.json()
            if result.get('candidates'):
                # Extract the JSON text content
                json_text = result['candidates'][0]['content']['parts'][0]['text']
                # Clean up potential markdown formatting
                json_text = json_text.replace('```json', '').replace('```', '').strip()
                return json.loads(json_text)
            
            print("API returned no candidates. Retrying...")

        except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
            print(f"API call failed: {e}. Retrying in {delay}s...")
            if i < max_retries - 1:
                time.sleep(delay)
                delay *= 2  # Exponential backoff
            else:
                return None
    return None

def generate_loc_and_pao(fact_text, loc_prefixes_str):
    """
    Step 1 & 2: Uses a single API call to get both the structured classification 
    and the PAO mnemonic, enforcing structure via JSON output.
    """
    
    # 1. System Instruction: Strict persona and rules
    system_prompt = (
        "You are an expert mnemonic generator and Library of Congress (LOC) cataloger. "
        "Your task is to analyze a fact and: "
        "1. Assign a 3-digit LOC-style classification (510, 940, 610, etc.) from the list provided. "
        "2. Generate a highly vivid, absurd, and sensory PAO sentence for the fact. "
        "The sentence MUST strictly follow the Person-Action-Object (PAO) structure. "
        "The Fact itself MUST be integrated into the Object component."
    )
    
    # 2. User Prompt: Include the current fact and the available LOC prefixes (for constraint)
    user_query = (
        f"Fact to categorize and generate mnemonic for: '{fact_text}'\n"
        f"Use one of these major LOC prefixes for categorization: {loc_prefixes_str}"
    )
    
    # 3. Structured JSON Schema: Enforces the exact output format
    payload = {
        "contents": [{"parts": [{"text": user_query}]}],
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "OBJECT",
                "properties": {
                    "loc_prefix": {"type": "STRING", "description": "The 3-digit LOC-style classification (e.g., 510, 940)."},
                    "pao_mnemonic": {"type": "STRING", "description": "The vivid, single-sentence PAO mnemonic."},
                }
            }
        }
    }

    response = call_gemini_api(payload)
    if response:
        return response
    return None

# --- 4. DATA GENERATION PIPELINE ---

def generate_training_data(raw_facts, loc_map):
    """
    Orchestrates the data generation, classification, and geometric assignment.
    """
    # Initialize the fixed Chess Cube structure
    palace = ChessCubeLattice()
    
    # Pre-map the Chess Cube Files (x-coordinates) to the major LOC prefixes
    loc_prefixes = list(loc_map.keys())
    
    # Assign File (x) coordinate based on LOC map
    loc_file_mapping = {loc: i + 1 for i, loc in enumerate(loc_prefixes)}
    
    print("Starting structured data generation...")
    final_dataset = []
    
    for i, fact in enumerate(raw_facts):
        print(f"Processing Fact {i+1}/{len(raw_facts)}: {fact[:30]}...")
        
        # Add delay to avoid rate limiting (except for first request)
        if i > 0:
            time.sleep(5)  # Increased to 5 seconds
        
        # Step 1 & 2: Get Classification and Mnemonic from LLM
        llm_output = generate_loc_and_pao(fact, ", ".join(loc_prefixes))
        
        if not llm_output:
            print(f"Skipping fact due to API failure/missing key.")
            # For demonstration purposes if API fails, we mock it so the user sees the structure
            if not API_KEY:
                print("  (Mocking data for demonstration)")
                llm_output = {
                    "loc_prefix": random.choice(loc_prefixes),
                    "pao_mnemonic": f"MOCKED PAO: A giant person is interacting with {fact[:10]}..."
                }
            else:
                continue
            
        loc_prefix = llm_output.get('loc_prefix')
        pao_mnemonic = llm_output.get('pao_mnemonic')
        
        if not loc_prefix or not pao_mnemonic or loc_prefix not in loc_map:
            print(f"Skipping fact: Invalid LOC prefix or mnemonic generated: {llm_output}")
            continue

        # Step 3: Geometric Assignment
        target_file_x = loc_file_mapping[loc_prefix]
        
        # Find an *available* location within that file (x-coordinate)
        available_locs = [k for k, v in palace.locations.items() if v['loc_prefix_mapped'] is None and v['x'] == target_file_x]
        
        if not available_locs:
            print(f"Skipping fact: File X={target_file_x} is full.")
            continue
            
        # Select the next available location
        # We use pop() which takes from the end, effectively filling from Z=8 down or similar depending on dict order
        # To be strict about path, we might want to sort them.
        # For now, let's sort by z, then y to fill bottom-up
        available_locs.sort(key=lambda k: (palace.locations[k]['z'], palace.locations[k]['y']))
        
        cell_key = available_locs[0] # Take the first one
        cell_details = palace.locations[cell_key]
        
        # Mark the cell as used
        palace.locations[cell_key]['loc_prefix_mapped'] = loc_prefix
        
        # Final Row Assembly
        final_dataset.append({
            'fact_text': fact,
            'fact_length': len(fact),
            'loc_prefix': loc_prefix,
            'pao_mnemonic': pao_mnemonic,
            'x_coord': cell_details['x'],
            'y_coord': cell_details['y'],
            'z_coord': cell_details['z'],
            'color_parity': cell_details['color_parity'],
            # Placeholder for semantic vector
            'semantic_vector_dummy': f"vector_{i}" 
        })
        
        if len(final_dataset) >= CUBE_SIZE**3: 
             break

    return pd.DataFrame(final_dataset)

# --- 5. EXECUTION ---

if __name__ == "__main__":
    # Define the major LOC categories mapping to Files 1-8
    LOC_MAPPING = {
        '510': 'Mathematics',        # X=1
        '530': 'Physics',            # X=2
        '610': 'Medicine/Health',    # X=3
        '940': 'European History',   # X=4
        '820': 'English Literature', # X=5
        '330': 'Economics',          # X=6
        '700': 'Arts',               # X=7
        '100': 'Philosophy'          # X=8
    }

    # Example Raw Facts (reduced to 1 for testing)
    RAW_FACTS = [
        "The inverse of a matrix is calculated using the adjoint and the determinant.",
    ]

    print("--- Generative Mnemonic Data Generator ---")
    df = generate_training_data(RAW_FACTS, LOC_MAPPING)
    
    print("\n--- FINAL GENERATED TRAINING DATA SAMPLE ---")
    if not df.empty:
        print(df[['fact_text', 'loc_prefix', 'pao_mnemonic', 'x_coord', 'y_coord', 'z_coord', 'color_parity']].head())
        
        # Save to CSV
        output_file = "memory_palace_training_data.csv"
        df.to_csv(output_file, index=False)
        print(f"\nData saved to {output_file}")
    else:
        print("\nError: No data was generated. Please check your API key and connection.")
