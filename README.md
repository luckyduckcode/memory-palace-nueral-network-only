# Memory Palace Neural Network (MPNN)

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

A **Neural Network system** that implements a spatially-organized **Memory Palace Neural Network (MPNN)** for enhanced memory and knowledge retrieval. The system organizes information in a structured 3D space, enabling efficient spatial reasoning and mnemonic generation.

## 🌟 Key Features

### Neural Architecture
- **Memory Palace Neural Network (MPNN)**: Spatial reasoning over a 3D knowledge graph
- **DIM-Net**: Neural networks mapping semantics to 3D coordinates
- **Spatial Reasoning**: Lattice-based organization for relationship modeling
- **Custom Loss Functions**: Accuracy, consistency, and geometric optimization
- **PAO Mnemonic Generation**: Vivid mnemonics for facts and concepts
- **Tier 2 Memory Palace**: Hierarchical storage for generated content

### Core Components
- **Mnemonic Model**: Core neural network for memory palace operations
- **Data Generators**: Tools for creating training data from facts
- **Training Scripts**: For training and fine-tuning the model
- **API Server**: Web interface for interacting with the model
- **GUI**: HTML-based interface for visualization

## 🏗️ Architecture Overview

### Memory Palace Pipeline

```
┌──────────────┐
│ User Input   │
│ "Store fact" │
└──────┬───────┘
       │
       ▼
┌─────────────────────────────────────────────────────────┐
│ DIM-Net Encoding                                        │
│ • Maps semantic content to 3D coordinates              │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│ Spatial Storage & Retrieval                             │
│ • Stores at optimal 3D loci                             │
│ • Retrieves via proximity search                       │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│ Mnemonic Generation                                     │
│ • Creates PAO associations                             │
│ • Generates vivid memory aids                          │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│ Tier 2 Consolidation                                    │
│ • Hierarchical storage of generated content            │
└─────────────────────────────────────────────────────────┘
```

## 🔬 Research-Based Refinements

This implementation incorporates recent advancements in AI-enhanced memory techniques:

- **Local LLM Integration**: Mnemonic generation now uses local LLMs (via Ollama) for privacy and offline capability, with fallback to cloud APIs. This aligns with studies on AI-personalized mnemonics (Alshehri, 2025).
- **Adaptive PAO Generation**: Enhanced prompts for vivid, sensory mnemonics based on research showing improved retention through absurdity and multi-sensory cues (Reddy, 2025).
- **Spatial Efficiency**: Neural network optimization for 3D coordinate mapping, inspired by neuroimaging studies on hippocampal efficiency (Wagner, 2021).

Future refinements may include VR/AR interfaces and neuroimaging feedback loops.

## 📦 Installation & Setup

### Prerequisites

- Python 3.8+
- PyTorch 2.0+

### Quick Setup

```bash
# Clone and enter directory
cd /path/to/memory-palace-nn

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Usage

### Quick Start

```bash
python demo_memory_palace.py
```

This will run demo operations for the memory palace NN.

### Python API

```python
from mnemonic_model import MnemonicModel

# Initialize the model
model = MnemonicModel()

# Train or use the model for memory palace operations
# (See train_memory_palace.py for training examples)
```

### Training

```bash
python train_memory_palace.py
```

This will train the memory palace NN on the provided datasets.

## 🧠 Core Components

### 1. Memory Palace Neural Network (MPNN)
**File**: `mnemonic_model.py`

The MPNN maintains a knowledge graph of concepts and facts:
```python
knowledge_graph = {
    "general_knowledge": ["science", "history", "geography"],
    "philosophy_archetypes": ["hero", "mentor", "trickster"],
    "religion_mythology": ["creation_stories", "deities", "rituals"],
    "social_sciences": ["politics", "economics", "sociology"],
    "language_linguistics": ["grammar", "etymology", "phonetics"],
    "science_technology": ["physics", "chemistry", "engineering"],
    "applied_sciences": ["medicine", "agriculture", "computing"],
    "arts_culture": ["literature", "music", "visual_arts"],
    ...
}
```

Each concept is mapped to 3D coordinates in the spatial lattice:
- `science` → (1, 2, 3)
- `physics` → (1, 2, 4)
- `mechanics` → (1, 3, 4)

### 2. DIM-Net Encoder
**File**: `mnemonic_model.py`

Maps semantic content to 3D coordinates:
```python
encoder = DIMNet()
coordinates = encoder.encode("quantum mechanics")
# Returns: (x, y, z) coordinates
```

### 3. Tier 2 Memory Palace
**File**: `mnemonic_model.py` (Tier2Lattice class)

A hierarchical storage system for generated content:
- **Tier 1**: Foundational concepts and facts (read-only)
- **Tier 2**: Generated associations and mnemonics (writable)

Each Tier 2 entry includes:
- The generated content
- Associated concepts
- Timestamp metadata

### 4. Data Generators
**Files**: `memory_palace_data_generator.py`, `expand_facts_dataset.py`

Tools for creating training data from fact databases and generating expanded datasets for training.

## 📊 Example Output

```
--- Processing Query: 'Find the derivative of x^2' ---
[Stage A] Parsing Intent...
  Intent: differentiation
  Domain: calculus
  Expression: x**2
[Stage B] Searching Memory Palace...
  Goal Loci: differentiation at (2, 5, 4)
  Deriving Path from 'calculus' to 'differentiation'...
  Path Sequence: ['calculus', 'differentiation']
[Stage C] Executing Symbolic Path...
  Symbolic Result: 2*x
[Stage D] Generating Explanation...
[Consolidation] Storing result in Tier 2 Memory Palace...
  Stored at Tier 2 Loci: 2_5_4 (linked to differentiation)

--- FINAL OUTPUT ---
Query: Find the derivative of x^2
Result: 2*x
Tier 2 Location: 2_5_4

Explanation:
To solve this differentiation problem:
  • Initial State: x**2
  • Entered Domain: calculus
  • Applied differentiation: 2*x

Final Result: 2*x
```

## 🔬 Research Background

This system implements the architecture described in "A Cognitive Architecture for Robust Symbolic Reasoning: The Memory Palace Neuro-Symbolic RAG Replacement."

### Key Innovations

1. **Structured Knowledge Graph**: Replaces flat vector search with hierarchical concept relationships
2. **Logical Pathing**: Guarantees valid sequences of mathematical operations
3. **Symbolic Verification**: Every step is mathematically verified (no hallucinations)
4. **Spatial Organization**: Leverages geometric intuition for concept retrieval

### Advantages Over Standard RAG

| Feature | Standard RAG | Memory Palace RAG |
|---------|-------------|-------------------|
| Knowledge Store | Unstructured text corpus | Structured knowledge graph |
| Retrieval | Vector similarity search | Spatial proximity + path derivation |
| Output | Text snippets (unverified) | Symbolic templates (pre-verified) |
| Reasoning | Statistical pattern matching | Logical path sequences |

## 🧪 Testing

```bash
# Test Tier 2 Memory Palace
python test_tier2.py

# Test individual components
python mnemonic_model.py  # Test MPNN
python symbolic_solver.py  # Test SymPy integration
python sllm_wrapper.py     # Test Ollama (requires ollama serve)
```

## 📁 Project Structure

```
memory-ai/
├── neuro_symbolic_rag.py          # Main pipeline
├── mnemonic_model.py               # MPNN, DIM-Net, Tier2Lattice
├── symbolic_solver.py              # SymPy integration
├── sllm_wrapper.py                 # Ollama interface (optional)
├── test_tier2.py                   # Tier 2 tests
├── requirements.txt                # Dependencies
├── README.md                       # This file
└── [legacy files]                  # Original data generation scripts
```

## 🛣️ Roadmap

- [x] Neuro-Symbolic 4-stage pipeline
- [x] Tier 2 Memory Palace
- [x] Differentiation and integration
- [ ] Equation solving
- [ ] Expand knowledge graph (100+ concepts)
- [ ] Train DIM-Net on mathematical corpus
- [ ] Multi-variable calculus
- [ ] Linear algebra operations
- [ ] Proof verification

## 🤝 Contributing

Contributions are welcome! Areas of interest:
- Expanding the knowledge graph
- Adding new mathematical domains
- Improving the symbolic solver
- Training the DIM-Net

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- **SymPy**: Symbolic mathematics library
- **PyTorch**: Neural network framework
- **Ollama**: Local LLM inference (optional)
- Research paper: "Memory Palace Neuro-Symbolic RAG Replacement"

---

**Last Updated**: November 27, 2025  
**Version**: 4.0.0 - Neuro-Symbolic Architecture
