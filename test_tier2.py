from mnemonic_model import MPNN

def test_tier2():
    print("Testing Tier 2 Memory Palace Logic...")
    mpnn = MPNN()
    
    goal_concept = "calculus"
    explanation = "Calculus is the mathematical study of continuous change."
    path = ["math", "calculus"]
    
    print(f"Consolidating memory for '{goal_concept}'...")
    key, coords = mpnn.consolidate_memory(goal_concept, explanation, path)
    
    print(f"Stored at Key: {key}")
    print(f"Coords: {coords}")
    
    # Verify storage
    stored_data = mpnn.tier2.storage[key]
    print(f"Retrieved from Tier 2: {stored_data}")
    
    assert len(stored_data) == 1
    assert stored_data[0]['explanation'] == explanation
    print("Verification Successful!")

if __name__ == "__main__":
    test_tier2()
