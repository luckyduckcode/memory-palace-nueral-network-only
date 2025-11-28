import numpy as np
import pandas as pd
import json
import time
import requests
import random
import os
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

# --- 1. CONFIGURATION ---
LLAMA_CPP_URL = "http://localhost:8080/completion"
MODEL_PATH = "./models/tinyllama-1.1b-chat-v1.0.Q4_0.gguf"

# Clustering configuration
FACTS_PER_CLUSTER = 10  # Each cluster contains 10 similar facts
CLUSTERS_PER_CATEGORY = 5  # 5 clusters per category = 50 facts per category

# --- 2. LLAMA.CPP FUNCTIONS ---

def start_llama_server():
    """Start llama.cpp server if not already running"""
    try:
        response = requests.get("http://localhost:8080/health", timeout=5)
        if response.status_code == 200:
            print("llama.cpp server is already running")
            return True
    except:
        pass

    print("Starting llama.cpp server...")
    try:
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
        time.sleep(15)
        return True
    except Exception as e:
        print(f"ERROR starting llama.cpp server: {e}")
        return False

def call_llama_cpp(prompt):
    """Call llama.cpp server for text generation"""
    payload = {
        "prompt": prompt,
        "n_predict": 256,
        "temperature": 0.3,  # Lower temperature for more focused responses
        "top_p": 0.9,
        "stop": ["\n\n", "###"],
        "stream": False
    }

    try:
        response = requests.post(LLAMA_CPP_URL, json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()
        return result.get('content', '').strip()
    except Exception as e:
        print(f"llama.cpp call failed: {e}")
        return None

# --- 3. SEMANTIC CLUSTERING ---

def load_facts():
    """Load facts from the expanded dataset"""
    categories = defaultdict(list)

    try:
        with open('facts_dataset_expanded.txt', 'r') as f:
            current_category = None
            for line in f:
                line = line.strip()
                if not line:
                    continue

                if line.startswith('#'):
                    parts = line.split(' - ')
                    if len(parts) >= 2:
                        category_part = parts[0].replace('# ', '')
                        loc_part = category_part.split('(')[1].split(')')[0] if '(' in category_part else ""
                        category_name = category_part.split(' (')[0]
                        current_category = (category_name, loc_part)
                elif current_category and not line.startswith('-'):
                    categories[current_category].append(line)
    except FileNotFoundError:
        print("facts_dataset_expanded.txt not found. Using original dataset.")
        with open('facts_dataset.txt', 'r') as f:
            current_category = None
            for line in f:
                line = line.strip()
                if not line:
                    continue

                if line.startswith('#'):
                    parts = line.split(' - ')
                    if len(parts) >= 2:
                        category_part = parts[0].replace('# ', '')
                        loc_part = category_part.split('(')[1].split(')')[0] if '(' in category_part else ""
                        category_name = category_part.split(' (')[0]
                        current_category = (category_name, loc_part)
                elif current_category and not line.startswith('-'):
                    categories[current_category].append(line)

    return categories

def create_semantic_clusters(facts, category_name, num_clusters):
    """
    Create semantic clusters of facts using LLM-based similarity assessment.
    """
    print(f"Creating {num_clusters} semantic clusters for {category_name}...")

    if len(facts) < num_clusters * FACTS_PER_CLUSTER:
        print(f"Warning: Only {len(facts)} facts available, need {num_clusters * FACTS_PER_CLUSTER}")
        num_clusters = len(facts) // FACTS_PER_CLUSTER
        if num_clusters == 0:
            return []

    clusters = []

    # Simple approach: randomly assign facts to clusters for now
    # In a full implementation, we'd use LLM to assess semantic similarity
    random.shuffle(facts)

    for i in range(num_clusters):
        start_idx = i * FACTS_PER_CLUSTER
        end_idx = min((i + 1) * FACTS_PER_CLUSTER, len(facts))
        cluster_facts = facts[start_idx:end_idx]

        if len(cluster_facts) == FACTS_PER_CLUSTER:
            clusters.append({
                'cluster_id': f"{category_name.lower().replace(' ', '_')}_cluster_{i+1}",
                'category': category_name,
                'facts': cluster_facts
            })

    print(f"Created {len(clusters)} clusters with {FACTS_PER_CLUSTER} facts each")
    return clusters

def assign_clusters_to_cube_locations(clusters):
    """
    Assign clusters to ChessCubeLattice locations.
    Each cluster gets a unique 3D coordinate.
    """
    print("Assigning clusters to Chess Cube locations...")

    # Chess cube has 8×8×8 = 512 locations
    # We'll assign clusters sequentially
    cube_locations = []

    for x in range(1, 9):  # 1-8
        for y in range(1, 9):  # 1-8
            for z in range(1, 9):  # 1-8
                cube_locations.append((x, y, z))

    # Assign clusters to locations
    cluster_assignments = []
    for i, cluster in enumerate(clusters):
        if i < len(cube_locations):
            x, y, z = cube_locations[i]
            parity = (x + y + z) % 2

            cluster_assignments.append({
                'cluster_id': cluster['cluster_id'],
                'category': cluster['category'],
                'x_coord': x,
                'y_coord': y,
                'z_coord': z,
                'color_parity': parity,
                'facts': cluster['facts']
            })

    print(f"Assigned {len(cluster_assignments)} clusters to cube locations")
    return cluster_assignments

def save_cluster_assignments(cluster_assignments):
    """Save the cluster assignments for training"""
    with open('fact_clusters.json', 'w') as f:
        json.dump(cluster_assignments, f, indent=2)

    # Also create a CSV format for easier processing
    rows = []
    for cluster in cluster_assignments:
        for fact in cluster['facts']:
            rows.append({
                'cluster_id': cluster['cluster_id'],
                'category': cluster['category'],
                'fact_text': fact,
                'x_coord': cluster['x_coord'],
                'y_coord': cluster['y_coord'],
                'z_coord': cluster['z_coord'],
                'color_parity': cluster['color_parity']
            })

    df = pd.DataFrame(rows)
    df.to_csv('fact_clusters.csv', index=False)

    print("Cluster assignments saved to 'fact_clusters.json' and 'fact_clusters.csv'")

# --- 4. EXECUTION ---

if __name__ == "__main__":
    print("=== FACT CLUSTERING FOR MEMORY PALACE TRAINING ===")

    # Load facts
    categories = load_facts()
    print(f"Loaded {len(categories)} categories with {sum(len(facts) for facts in categories.values())} total facts")

    # Create clusters for each category
    all_clusters = []
    for (category_name, loc_prefix), facts in categories.items():
        print(f"\n--- Processing {category_name} ---")

        # Calculate how many clusters we can make
        max_clusters = len(facts) // FACTS_PER_CLUSTER
        num_clusters = min(CLUSTERS_PER_CATEGORY, max_clusters)

        if num_clusters > 0:
            category_clusters = create_semantic_clusters(facts, category_name, num_clusters)
            all_clusters.extend(category_clusters)
        else:
            print(f"Not enough facts in {category_name} for clustering")

    print(f"\nTotal clusters created: {len(all_clusters)}")

    # Assign to cube locations
    cluster_assignments = assign_clusters_to_cube_locations(all_clusters)

    # Save results
    save_cluster_assignments(cluster_assignments)

    print("\n=== CLUSTERING COMPLETE ===")
    print(f"Created {len(cluster_assignments)} clusters")
    print(f"Total facts in clusters: {sum(len(c['facts']) for c in cluster_assignments)}")
    print("Files saved: fact_clusters.json, fact_clusters.csv")