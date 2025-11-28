import numpy as np
import pandas as pd
import json
import time
import requests
import random
import os
import subprocess
from collections import defaultdict

# --- 1. CONFIGURATION ---
# Use llama.cpp server instead of Gemini API
LLAMA_CPP_URL = "http://localhost:8080/completion"  # Default llama.cpp server port
MODEL_PATH = "./models/tinyllama-1.1b-chat-v1.0.Q4_0.gguf"  # Relative path - update if needed

# Target expansion: 64 facts per category → 640 facts per category (10x increase)
# Total: 512 → 5120 facts
TARGET_FACTS_PER_CATEGORY = 640
CURRENT_FACTS_PER_CATEGORY = 64

# --- 2. LLAMA.CPP FUNCTIONS ---

def start_llama_server():
    """Start llama.cpp server if not already running"""
    try:
        # Check if server is already running
        response = requests.get("http://localhost:8080/health", timeout=5)
        if response.status_code == 200:
            print("llama.cpp server is already running")
            return True
    except:
        pass

    print("Starting llama.cpp server...")
    try:
        # Use absolute path to llama-server
        script_dir = os.path.dirname(os.path.abspath(__file__))
        llama_server_path = os.path.join(script_dir, "models", "llama.cpp", "llama-server")
        
        cmd = [
            llama_server_path,
            "--model", MODEL_PATH,
            "--ctx-size", "2048",
            "--threads", "4",
            "--port", "8080",
            "--host", "127.0.0.1"
        ]
        subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(15)  # Wait longer for server to start
        return True
    except FileNotFoundError:
        print(f"ERROR: llama-server not found at {llama_server_path}")
        return False
    except Exception as e:
        print(f"ERROR starting llama.cpp server: {e}")
        return False

def call_llama_cpp(prompt):
    """Call llama.cpp server for text generation"""
    payload = {
        "prompt": prompt,
        "n_predict": 512,
        "temperature": 0.7,
        "top_p": 0.9,
        "stop": ["\n\n", "###"],
        "stream": False
    }

    try:
        response = requests.post(LLAMA_CPP_URL, json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()
        return result.get('content', '').strip()
    except Exception as e:
        print(f"llama.cpp call failed: {e}")
        return None

def generate_additional_facts_llama(category_name, loc_prefix, existing_facts, num_to_generate):
    """
    Generate additional facts using a simpler approach - one fact at a time.
    """
    print(f"Generating {num_to_generate} additional facts for {category_name} ({loc_prefix})...")

    new_facts = []

    for i in range(num_to_generate):
        # Simple prompt for one fact at a time
        prompt = f"Write one interesting fact about {category_name}. Make it concise and accurate."

        response = call_llama_cpp(prompt)
        if response:
            # Clean up the response
            fact = response.strip()
            # Remove common prefixes that models add
            fact = fact.lstrip("123456789. ")
            fact = fact.strip()

            # Basic quality checks
            if (len(fact) > 20 and len(fact) < 200 and
                fact not in existing_facts and
                not fact.startswith("Here") and
                not fact.startswith("One") and
                not fact.startswith("A")):

                new_facts.append(fact)
                print(f"  Generated: {fact[:50]}...")
            else:
                print(f"  Rejected: {fact[:50]}...")
        else:
            print(f"  Failed to generate fact {i+1}")

        # Small delay between requests
        time.sleep(1)

    print(f"Generated {len(new_facts)} valid new facts for {category_name}")
    return new_facts

# --- 3. DATASET EXPANSION PIPELINE ---

def load_existing_facts():
    """Load and categorize existing facts from facts_dataset.txt"""
    categories = defaultdict(list)

    with open('facts_dataset.txt', 'r') as f:
        current_category = None
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith('#'):
                # Extract category info
                if ' - ' in line:
                    category_part = line.split(' - ')[0].replace('# ', '')
                    loc_part = line.split('(')[1].split(')')[0] if '(' in line else ""
                    current_category = (category_part, loc_part)
            elif current_category and not line.startswith('-'):
                categories[current_category].append(line)

    return categories

def expand_dataset():
    """Main function to expand the dataset from 512 to 5120 facts"""
    print("=== FACTS DATASET EXPANSION: 512 → 5120 facts ===")

    # Load existing facts
    categories = load_existing_facts()
    print(f"Loaded {len(categories)} categories with {sum(len(facts) for facts in categories.values())} total facts")

    # Expand each category
    expanded_categories = {}

    for (category_name, loc_prefix), existing_facts in categories.items():
        print(f"\n--- Processing {category_name} ({loc_prefix}) ---")
        print(f"Current facts: {len(existing_facts)}")

        # Calculate how many more facts needed
        current_count = len(existing_facts)
        needed = TARGET_FACTS_PER_CATEGORY - current_count

        if needed <= 0:
            print(f"Category already has {current_count} facts (target: {TARGET_FACTS_PER_CATEGORY})")
            expanded_categories[(category_name, loc_prefix)] = existing_facts
            continue

        # Generate additional facts
        new_facts = []
        batch_size = 10  # Generate in smaller batches

        for i in range(0, needed, batch_size):
            batch_needed = min(batch_size, needed - i)
            batch_facts = generate_additional_facts_llama(category_name, loc_prefix, existing_facts + new_facts, batch_needed)

            new_facts.extend(batch_facts)

            # Add delay between batches
            if i + batch_size < needed:
                print(f"Batch {i//batch_size + 1} complete. Waiting 5 seconds...")
                time.sleep(5)

        # Combine existing and new facts
        all_facts = existing_facts + new_facts
        expanded_categories[(category_name, loc_prefix)] = all_facts

        print(f"Final count for {category_name}: {len(all_facts)} facts")

    # Save expanded dataset
    save_expanded_dataset(expanded_categories)

    # Print summary
    total_facts = sum(len(facts) for facts in expanded_categories.values())
    print("\n=== EXPANSION COMPLETE ===")
    print(f"Total facts: {total_facts} (target: {TARGET_FACTS_PER_CATEGORY * len(categories)})")
    print(f"Categories: {len(expanded_categories)}")

def save_expanded_dataset(expanded_categories):
    """Save the expanded dataset to a new file"""
    with open('facts_dataset_expanded.txt', 'w') as f:
        for (category_name, loc_prefix), facts in expanded_categories.items():
            f.write(f"# {category_name} ({loc_prefix}) - {len(facts)} facts\n")

            for fact in facts:
                f.write(f"{fact}\n")

            f.write("\n")

    print("Expanded dataset saved to 'facts_dataset_expanded.txt'")

# --- 4. EXECUTION ---

if __name__ == "__main__":
    # Start llama.cpp server
    if not start_llama_server():
        print("Failed to start llama.cpp server. Please ensure it's installed and the model path is correct.")
        exit(1)

    expand_dataset()