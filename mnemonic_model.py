import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import heapq

# --- CONFIGURATION ---
SEMANTIC_DIM = 768  # Dimension of the input Fact Vector (from LLM embedding)
MNEMONIC_DIM = 512  # Dimension of the latent Mnemonic Potential Vector
OUTPUT_DIM = 3      # (x, y, z) coordinates for the Chess Cube
CUBE_SIZE = 8       # Size of the Chess Cube (8x8x8)

# TRHD Configuration
FACT_VECTOR_DIM = 64     # Semantic Vector Size
META_DIM = 7             # 5W1H (6 dim) + Truth Score (1 dim)
TOTAL_INPUT_DIM = FACT_VECTOR_DIM + META_DIM # 71 dimensions per fact
NUM_FACTS_PER_CLUSTER = 10

# --- CHESS CUBE LATTICE ---
class ChessCubeLattice:
    """Rigorous definition of the 8x8x8 Memory Palace structure."""
    def __init__(self, size=CUBE_SIZE):
        self.size = size
        self.coordinates = self._generate_coordinates()

    def _generate_coordinates(self):
        coordinates = {}
        for x in range(1, self.size + 1):  # File (A-H)
            for y in range(1, self.size + 1):  # Rank (1-8)
                for z in range(1, self.size + 1):  # Height (Floor 1-8)
                    cell_key = f"{x}_{y}_{z}"
                    coordinates[cell_key] = {
                        'x': x, 'y': y, 'z': z,
                        # Parity is crucial for Separation Loss (Alternating Property)
                        'color_parity': (x + y + z) % 2,
                        'loc_prefix_mapped': None # To be mapped later
                    }
        return coordinates

class Tier2Lattice(ChessCubeLattice):
    """
    A secondary, 'Shadow' Memory Palace for storing generated knowledge (explanations).
    It has 'less permissions' (read-only for foundational logic) and is lower in hierarchy.
    It maps 1:1 to the primary lattice but stores 'derivative' content.
    """
    def __init__(self, size=CUBE_SIZE):
        super().__init__(size)
        self.storage = {} # Key: coordinate_key, Value: list of explanations

    def store_explanation(self, primary_coords, explanation, path_sequence):
        """
        Stores the explanation in the Tier 2 lattice at the corresponding location.
        """
        # Convert tuple coords to key
        x, y, z = primary_coords
        key = f"{x}_{y}_{z}"
        
        if key not in self.storage:
            self.storage[key] = []
            
        entry = {
            "explanation": explanation,
            "path_origin": path_sequence,
            "timestamp": "now" # In real system, use actual time
        }
        self.storage[key].append(entry)
        return key

    def get_neighbors(self, coords, radius=1):
        """
        Retrieves explanations from neighboring coordinates in 3D space.
        This enables branched retrieval for richer context.
        
        Args:
            coords (tuple): (x, y, z) coordinates
            radius (int): Search radius in each dimension
            
        Returns:
            list: List of explanation entries from neighboring locations
        """
        x, y, z = coords
        neighbors = []
        
        # Generate all coordinates within the radius
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                for dz in range(-radius, radius + 1):
                    # Skip the center coordinate (0,0,0)
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    
                    nx, ny, nz = x + dx, y + dy, z + dz
                    
                    # Check bounds (assuming cube size is accessible)
                    if 0 <= nx < self.size and 0 <= ny < self.size and 0 <= nz < self.size:
                        neighbor_key = f"{nx}_{ny}_{nz}"
                        if neighbor_key in self.storage:
                            neighbors.extend(self.storage[neighbor_key])
        
        return neighbors

class DIM_Net(nn.Module):
    """
    Dimensionality Isomorphism Network (DIM-Net)
    
    A conceptual model designed to map high-dimensional semantic vectors (1D)
    to structured geometric coordinates (3D) while preserving topological relationships.
    """
    def __init__(self, input_dim=SEMANTIC_DIM, output_dim=OUTPUT_DIM):
        super(DIM_Net, self).__init__()
        
        # Encoder: Compresses the semantic vector
        self.encoder = nn.Linear(input_dim, 256)
        
        # Mapping Layers: The "Isomorphism" engine
        # Uses LeakyReLU to allow for non-linear manifold learning
        self.mapping = nn.Sequential(
            nn.Linear(256, 512),
            nn.LeakyReLU(0.01),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.01),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.01),
        )
        
        # Output Layer: Predicts the 3D coordinate (x, y, z)
        self.output_layer = nn.Linear(256, output_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = self.mapping(x)
        return self.output_layer(x)

class MPNN(nn.Module):
    """
    Memory Palace Neural Network (MPNN)
    
    Acts as a Differentiable Neural Computer (DNC) or MANN.
    Grounds problem-solving in the spatially-organized Chess Cube.
    """
    def __init__(self, dim_net=None, lattice=None):
        super(MPNN, self).__init__()
        self.dim_net = dim_net if dim_net else DIM_Net()
        self.dim_net = dim_net if dim_net else DIM_Net()
        self.lattice = lattice if lattice else ChessCubeLattice()
        self.tier2 = Tier2Lattice() # Initialize Tier 2
        
        # Knowledge Graph (Adjacency List)
        # Comprehensive graph connecting mathematical concepts
        self.knowledge_graph = {
            # Calculus
            "calculus": ["differentiation", "integration", "limits", "series"],
            "differentiation": ["power_rule", "chain_rule", "product_rule", "quotient_rule"],
            "power_rule": [],
            "chain_rule": [],
            "product_rule": [],
            "quotient_rule": [],
            "integration": ["integration_by_parts", "substitution", "partial_fractions"],
            "integration_by_parts": [],
            "substitution": [],
            "partial_fractions": [],
            "limits": ["lhopital_rule"],
            "lhopital_rule": [],
            "series": ["taylor_series", "maclaurin_series"],
            "taylor_series": [],
            "maclaurin_series": [],
            
            # Algebra
            "algebra": ["linear_equations", "quadratic_equations", "polynomial_equations", "simplify", "expand", "factor"],
            "linear_equations": ["matrix_operations"],
            "quadratic_equations": ["quadratic_formula", "completing_square"],
            "quadratic_formula": [],
            "completing_square": [],
            "polynomial_equations": ["factor", "expand"],
            "simplify": [],
            "expand": [],
            "factor": [],
            
            # Linear Algebra
            "linear_algebra": ["matrix_operations", "eigenvalues", "determinant"],
            "matrix_operations": ["inverse", "determinant", "transpose"],
            "inverse": [],
            "determinant": [],
            "transpose": [],
            "eigenvalues": ["eigenvectors"],
            "eigenvectors": [],
            
            # Trigonometry
            "trigonometry": ["trig_identities", "trig_simplify"],
            "trig_identities": [],
            "trig_simplify": [],
            
            # Number Theory
            "number_theory": ["prime_factorization", "gcd", "lcm"],
            "prime_factorization": [],
            "gcd": [],
            "lcm": [],
        }
        
        # Map concepts to approximate coordinates (Mocking the trained state)
        self.concept_locations = {
            # Calculus (x=2-3)
            "calculus": (2, 5, 3),
            "differentiation": (2, 5, 4),
            "power_rule": (2, 6, 4),
            "chain_rule": (2, 6, 5),
            "product_rule": (2, 7, 4),
            "quotient_rule": (2, 7, 5),
            "integration": (2, 4, 3),
            "integration_by_parts": (2, 4, 4),
            "substitution": (2, 3, 4),
            "limits": (3, 5, 3),
            "series": (3, 5, 4),
            "taylor_series": (3, 6, 4),
            
            # Algebra (x=1)
            "algebra": (1, 2, 2),
            "linear_equations": (1, 3, 2),
            "quadratic_equations": (1, 3, 3),
            "quadratic_formula": (1, 4, 3),
            "simplify": (1, 2, 3),
            "expand": (1, 2, 4),
            "factor": (1, 2, 5),
            
            # Linear Algebra (x=4)
            "linear_algebra": (4, 3, 3),
            "matrix_operations": (4, 4, 3),
            "inverse": (4, 4, 4),
            "determinant": (4, 5, 3),
            "eigenvalues": (4, 5, 4),
            
            # Trigonometry (x=5)
            "trigonometry": (5, 3, 3),
            "trig_identities": (5, 4, 3),
        }

    def spatial_proximity_search(self, problem_vector):
        """
        Stage B.1: Identifies the target Goal Loci based on conceptual distance.
        Returns:
            goal_concept (str): The identified concept key.
            coords (tuple): The (x, y, z) coordinates.
        """
        # 1. Use DIM-Net to map vector to 3D space
        # Ensure input is a tensor
        if not isinstance(problem_vector, torch.Tensor):
            problem_vector = torch.tensor(problem_vector, dtype=torch.float32)
        
        # If vector is 1D, add batch dim
        if len(problem_vector.shape) == 1:
            problem_vector = problem_vector.unsqueeze(0)
            
        predicted_coords = self.dim_net(problem_vector).detach().numpy()[0]
        
        # 2. Find nearest concept in our "Knowledge Graph" (concept_locations)
        # In a real system, this would be a k-NN search on the Loci vectors
        nearest_concept = None
        min_dist = float('inf')
        
        for concept, loc in self.concept_locations.items():
            dist = np.linalg.norm(predicted_coords - np.array(loc))
            if dist < min_dist:
                min_dist = dist
                nearest_concept = concept
                
        return nearest_concept, self.concept_locations.get(nearest_concept, (0,0,0))

    def derive_path(self, start_concept, goal_concept):
        """
        Stage B.2: Computes optimal logical Path Sequence.
        Uses Dijkstra's algorithm on the Knowledge Graph.
        """
        # Simple BFS/Dijkstra
        queue = [(0, start_concept, [])] # (cost, current, path)
        visited = set()
        
        while queue:
            cost, current, path = heapq.heappop(queue)
            
            if current == goal_concept:
                return path + [current]
            
            if current in visited:
                continue
            visited.add(current)
            
            # Get neighbors
            neighbors = self.knowledge_graph.get(current, [])
            
            # Also check reverse links (undirected for traversal?) 
            # Or assume strict hierarchy. Let's assume strict for now.
            
            for neighbor in neighbors:
                heapq.heappush(queue, (cost + 1, neighbor, path + [current]))
                
        return [] # No path found

    def consolidate_memory(self, goal_concept, explanation, path_sequence):
        """
        Stores the result in the Tier 2 Memory Palace.
        """
        # Get coords of the goal concept
        coords = self.concept_locations.get(goal_concept, (1,1,1))
        
        # Store in Tier 2
        storage_key = self.tier2.store_explanation(coords, explanation, path_sequence)
        return storage_key, coords

    def retrieve_similar_explanations(self, goal_concept, radius=1):
        """
        Retrieves explanations from neighboring locations in Tier 2 for richer context.
        This enables branched retrieval to provide associative memory context.
        
        Args:
            goal_concept (str): The target concept
            radius (int): Search radius for neighboring coordinates
            
        Returns:
            list: List of similar explanations from neighboring locations
        """
        # Get coordinates of the goal concept
        coords = self.concept_locations.get(goal_concept, (1,1,1))
        
        # Get neighboring explanations from Tier 2
        neighbor_explanations = self.tier2.get_neighbors(coords, radius)
        
        # Also get explanations from the exact location
        key = f"{coords[0]}_{coords[1]}_{coords[2]}"
        local_explanations = self.tier2.storage.get(key, [])
        
        # Combine and return all explanations
        all_explanations = local_explanations + neighbor_explanations
        
        return all_explanations

class GenerativeMnemonicModel(nn.Module):
    """
    The Advanced Hybrid Model.
    
    Combines the geometric mapping of DIM-Net with a generative capability
    to produce the 'Mnemonic Potential Vector' (MPV) for the PAO system.
    """
    def __init__(self, semantic_dim=SEMANTIC_DIM, mnemonic_dim=MNEMONIC_DIM):
        super(GenerativeMnemonicModel, self).__init__()
        
        # Encoder: Maps Fact -> Mnemonic Potential Vector (MPV)
        # This MPV is what would be fed into an LLM Decoder to generate the text.
        self.encoder = nn.Sequential(
            nn.Linear(semantic_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, mnemonic_dim),
            nn.Tanh() # Tanh to normalize the latent space
        )
    
    def forward(self, fact_vector):
        mnemonic_potential_vector = self.encoder(fact_vector)
        return mnemonic_potential_vector

class TRHD_MnemonicMapper(nn.Module):
    """Maps a cluster of 10 structured facts to a 3D coordinate."""
    def __init__(self, input_dim=TOTAL_INPUT_DIM, output_dim=OUTPUT_DIM):
        super(TRHD_MnemonicMapper, self).__init__()
        
        # 1. Attention Mechanism (learns how to focus on salient features)
        # We need a small network to calculate the 'query' for the attention scores
        self.query_net = nn.Linear(input_dim, input_dim) 
        
        # 2. Final Mapping Layers (takes the 71-dim Cluster Vector as input)
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), 
            nn.LeakyReLU(0.01),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.01),
            nn.Linear(64, output_dim),
            nn.Tanh()
        )
        self.size = CUBE_SIZE

    # Input: (Batch_Size, 10, TOTAL_INPUT_DIM) 
    def forward(self, input_fact_cluster):
        
        # 1. Calculate Attention Scores (Conceptual)
        # In a real model, this would be QKV attention. 
        # Here, we use the embedded Truth Score as the primary weight source.
        
        # For PoC, let's extract the Truth Score (assumed to be the last column)
        truth_scores = input_fact_cluster[:, :, -1] # Shape: (Batch_Size, 10)

        # Apply Softmax to get the attention weights for the 10 facts in each cluster
        # This ensures the weights sum to 1.
        attention_weights = F.softmax(truth_scores * 10, dim=1).unsqueeze(-1) # Shape: (B, 10, 1)

        # 2. Apply Weighted Sum (Attention Pooling)
        # Cluster Vector is the weighted average of the 10 structured facts
        cluster_vector = torch.sum(input_fact_cluster * attention_weights, dim=1) # Shape: (B, TOTAL_INPUT_DIM)

        # 3. MAPPING STEP
        output = self.net(cluster_vector)
        return (output + 1.0) / 2.0 * self.size # Rescale to cube size

# --- CUSTOM LOSS FUNCTIONS ---

def geometric_loss(predictions, targets, orthogonality_weight=0.001):
    """
    Enforces coordinate accuracy (MSE) and geometric properties (orthogonality).
    
    Args:
        predictions: Tensor of shape (batch_size, 3) -> Predicted (x, y, z)
        targets: Tensor of shape (batch_size, 3) -> Actual Chess Cube (x, y, z)
        orthogonality_weight: Penalty weight for the norm constraint.
    """
    mse_loss = nn.MSELoss()(predictions, targets)
    
    # Orthogonality Constraint (Conceptual):
    # Penalizes the magnitude of the vector to encourage efficient, orthogonal representations
    # In a full implementation, this would involve dot products between different feature dimensions.
    orthogonality_penalty = torch.norm(predictions, p=2) * orthogonality_weight
    
    return mse_loss + orthogonality_penalty

def compression_vividness_loss(predicted_logits, target_tokens, fact_length_norm):
    """
    Optimizes for high vividness (low cross-entropy loss) weighted by 
    high compression (penalizing errors more heavily for simple/short facts).
    
    Args:
        predicted_logits: Output from the LLM Decoder (Batch, Seq_Len, Vocab_Size)
        target_tokens: The tokenized PAO mnemonic string (Batch, Seq_Len)
        fact_length_norm: Normalized length of the original fact (0.0 to 1.0)
    """
    # Standard generation loss
    # Note: CrossEntropyLoss expects (Batch, Class, Seq) or (Batch, Class)
    # We assume flattened for simplicity here
    vividness_loss = F.cross_entropy(predicted_logits.view(-1, predicted_logits.size(-1)), target_tokens.view(-1)) 
    
    # Compression Weighting:
    # 1.0 / (length + epsilon) -> Short facts have HIGH weight.
    # If the model messes up a short, simple fact, it gets a huge penalty.
    compression_weight = 1.0 / (fact_length_norm + 1e-6)
    
    return vividness_loss * compression_weight

def truth_consistency_loss(predicted_coords, color_parity_labels, avg_truth_scores):
    """
    Enforces that the geometric Parity (0/1) matches the semantic Truth (High/Low).
    
    Target Parity Logic: If Avg Truth > 0.7, Target Parity = 0 (White/Certain). 
    Otherwise, Target Parity = 1 (Black/Uncertain).
    """
    # 1. Determine the Ground Truth Parity based on the input data's truth score
    # Note: We assume Parity 0 (White) is the 'Certain' zone.
    target_parity = (avg_truth_scores < 0.7).float() # 1.0 if uncertain (<0.7), 0.0 if certain (>=0.7)
    
    # 2. Determine the Predicted Parity based on the cube's fixed geometry
    # We must match the predicted coordinate (e.g., 1.2, 3.8, 2.1) to the nearest fixed cube's color parity.
    
    # Simple check: round the predicted coordinate and check its parity
    # Note: The ChessCubeLattice class is needed here to look up the fixed parity.
    
    # PoC Simulation: We use a simple distance to the center of the target cube
    
    # This calculation is complex and requires coordinate lookup. For the PoC, 
    # we'll use a simplified check against the *target* parity, assuming a perfect 
    # geometric match is the goal.
    
    # The geometric separation loss already encourages the predicted coordinates 
    # to land near their target coordinates which *already* have the correct parity.
    
    # Re-using the Separation Loss target for the Truth Loss simplifies the PoC:
    # If the network moves the point to a cube of the wrong color (wrong parity label), 
    # it receives a high penalty.
    return nn.MSELoss()(color_parity_labels, target_parity) # Punish if the predicted location's color is wrong

def generate_simulated_trhd_data(num_clusters):
    """
    Generates dummy input/output data for PoC training with 10 facts per cluster.
    """
    palace = ChessCubeLattice(size=CUBE_SIZE)
    target_coords = palace.coordinates
    
    # INPUT: Structured Fact Clusters (Batch, 10, 71)
    input_clusters = torch.randn(num_clusters, NUM_FACTS_PER_CLUSTER, TOTAL_INPUT_DIM)
    
    # Simulate the Truth Score (last dimension)
    # Make some clusters highly certain (avg truth ~ 0.9) and others uncertain (~0.4)
    avg_truth_scores = torch.rand(num_clusters, 1) * 0.5 + 0.25 # 0.25 to 0.75 range
    for i in range(num_clusters):
        if i % 3 == 0: # Make 1/3 of the clusters highly certain
             avg_truth_scores[i] = torch.rand(1) * 0.1 + 0.85 # 0.85 to 0.95
        
        # Set the last dimension (Truth Score) for all 10 facts in the cluster
        input_clusters[i, :, -1] = avg_truth_scores[i].item() + (torch.rand(NUM_FACTS_PER_CLUSTER) - 0.5) * 0.1

    # TARGET: Geometric Labels (Color Parity)
    # The true color parity of the geometric target (from the Chess Cube)
    color_parity_labels = torch.tensor([(i // (CUBE_SIZE**2)) % 2 for i in range(num_clusters)], dtype=torch.float32)

    # Note: target_coords and target_vividness/fact_complexity would be generated similarly.
    return input_clusters, target_coords, color_parity_labels, avg_truth_scores

# --- EXAMPLE USAGE ---
if __name__ == "__main__":
    # 1. Instantiate the models
    dim_net = DIM_Net()
    mpnn = MPNN(dim_net=dim_net)
    
    print("MPNN instantiated successfully.")
    
    # 2. Test Spatial Search
    # Dummy 768-dim vector
    dummy_vec = torch.randn(1, SEMANTIC_DIM)
    concept, loc = mpnn.spatial_proximity_search(dummy_vec)
    print(f"Nearest Concept: {concept} at {loc}")
    
    # 3. Test Path Derivation
    path = mpnn.derive_path("calculus", "power_rule")
    print(f"Derived Path: {path}")
