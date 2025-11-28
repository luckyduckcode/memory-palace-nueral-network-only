import torch
import pandas as pd
import numpy as np
from mnemonic_model import DIM_Net, TRHD_MnemonicMapper, ChessCubeLattice

# --- 1. CONFIGURATION ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- 2. MODEL LOADING ---

def load_trained_models():
    """Load the trained DIM-Net and TRHD_MnemonicMapper"""
    print("Loading trained models...")

    # Initialize models
    dim_net = DIM_Net()
    trhd_mapper = TRHD_MnemonicMapper()

    # Load trained weights
    try:
        dim_net.load_state_dict(torch.load('dim_net_final.pth', map_location=DEVICE))
        trhd_mapper.load_state_dict(torch.load('trhd_mapper_final.pth', map_location=DEVICE))
        print("✅ Models loaded successfully")
    except FileNotFoundError:
        print("❌ Model files not found. Please run training first.")
        return None, None

    dim_net.to(DEVICE)
    trhd_mapper.to(DEVICE)
    dim_net.eval()
    trhd_mapper.eval()

    return dim_net, trhd_mapper

# --- 3. INFERENCE FUNCTIONS ---

def create_fact_cluster(fact, cluster_size=10):
    """
    Create a simulated cluster around a single fact.
    In practice, you'd group similar facts together.
    """
    # Create cluster input features (simplified)
    cluster_features = []

    for i in range(cluster_size):
        if i == 0:
            # Main fact
            fact_vector = torch.randn(70)  # Feature vector
            fact_vector[0] = len(fact) / 100.0  # Normalized length
            truth_score = torch.tensor([0.9])  # High confidence for main fact
        else:
            # Supporting facts (simulated)
            fact_vector = torch.randn(70)
            fact_vector[0] = np.random.uniform(0.5, 1.5)  # Varied lengths
            truth_score = torch.rand(1) * 0.4 + 0.6  # 0.6-1.0 range

        fact_feature = torch.cat([fact_vector, truth_score])
        cluster_features.append(fact_feature)

    return torch.stack(cluster_features).unsqueeze(0)  # Add batch dimension

def generate_memory_coordinates(fact, dim_net, trhd_mapper):
    """
    Generate 3D coordinates for a fact using the trained models
    """
    print(f"\n🧠 Processing fact: '{fact[:60]}...'")

    # Create cluster input
    cluster_input = create_fact_cluster(fact).to(DEVICE)

    # Get TRHD_MnemonicMapper prediction
    with torch.no_grad():
        predicted_coords = trhd_mapper(cluster_input).squeeze(0).cpu().numpy()

    # Round to nearest integer coordinates (chess cube positions)
    x_coord = int(np.clip(np.round(predicted_coords[0]), 1, 8))
    y_coord = int(np.clip(np.round(predicted_coords[1]), 1, 8))
    z_coord = int(np.clip(np.round(predicted_coords[2]), 1, 8))

    # Calculate parity
    parity = (x_coord + y_coord + z_coord) % 2

    return {
        'x': x_coord,
        'y': y_coord,
        'z': z_coord,
        'parity': parity,
        'raw_coords': predicted_coords
    }

def generate_pao_mnemonic(fact, coords):
    """
    Generate a PAO mnemonic using the coordinates.
    This is a simplified version - in practice you'd use LLM.
    """
    x, y, z = coords['x'], coords['y'], coords['z']

    # Simple PAO mapping (Person-Action-Object)
    persons = ["Albert Einstein", "Marie Curie", "Isaac Newton", "Leonardo da Vinci",
              "Charles Darwin", "Galileo", "Nikola Tesla", "Ada Lovelace"]
    actions = ["painting", "dancing", "juggling", "building", "flying", "swimming",
              "singing", "writing"]
    objects = ["crystals", "equations", "machines", "books", "stars", "robots",
              "paintings", "musical notes"]

    person = persons[(x-1) % len(persons)]
    action = actions[(y-1) % len(actions)]
    obj = objects[(z-1) % len(objects)]

    mnemonic = f"{person} is {action} {obj} while remembering: {fact[:30]}..."

    return mnemonic

# --- 4. DEMO FUNCTION ---

def demo_memory_palace():
    """Demonstrate the trained memory palace system"""
    print("🎭 AI-POWERED MEMORY PALACE DEMO")
    print("=" * 50)

    # Load models
    dim_net, trhd_mapper = load_trained_models()
    if dim_net is None:
        return

    # Initialize chess cube
    palace = ChessCubeLattice()

    # Test facts - now using mathematical concepts
    test_facts = [
        "Function composition: f(g(x))",
        "Obtuse triangle: one angle > 90°",
        "Closure, associativity, identity, inverses",
        "Word problems with subtraction",
        "Applications in differential equations"
    ]

    print(f"\n🧠 Processing {len(test_facts)} facts through the memory palace...\n")

    results = []
    for fact in test_facts:
        # Generate coordinates
        coords = generate_memory_coordinates(fact, dim_net, trhd_mapper)

        # Generate mnemonic
        mnemonic = generate_pao_mnemonic(fact, coords)

        # Store result
        result = {
            'fact': fact,
            'coordinates': f"({coords['x']}, {coords['y']}, {coords['z']})",
            'parity': 'Black' if coords['parity'] == 1 else 'White',
            'mnemonic': mnemonic,
            'raw_coords': coords['raw_coords']
        }
        results.append(result)

        # Display result
        print(f"📍 Coordinates: {result['coordinates']} ({result['parity']})")
        print(f"🎭 Mnemonic: {result['mnemonic']}")
        print("-" * 80)

    # Summary statistics
    print("\n📊 MEMORY PALACE SUMMARY")
    print(f"Total facts processed: {len(results)}")

    # Coordinate distribution
    coords_list = [r['raw_coords'] for r in results]
    coords_array = np.array(coords_list)

    print("Coordinate Statistics:")
    print(".2f")
    print(".2f")
    print(".2f")

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv('memory_palace_demo_results.csv', index=False)
    print("\n💾 Results saved to 'memory_palace_demo_results.csv'")
    print("\n🎉 Memory Palace Demo Complete!")
    print("Your AI system can now encode facts as 3D spatial memories!")

if __name__ == "__main__":
    demo_memory_palace()