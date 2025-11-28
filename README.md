# Neuro-Symbolic Memory Palace: Mathematical Reasoning AI

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![SymPy](https://img.shields.io/badge/SymPy-Symbolic-green.svg)](https://www.sympy.org/)

A **Neuro-Symbolic AI system** that replaces traditional Vector-Search RAG with a spatially-organized **Memory Palace Neural Network (MPNN)**. The system grounds mathematical problem-solving in a structured knowledge graph (Chess Cube lattice), ensuring symbolically correct, contextually grounded, and linguistically explainable derivations.

## 🌟 Key Features

### Neuro-Symbolic Architecture
- **4-Stage Pipeline**: Intent Parsing → Memory Palace Search → Symbolic Execution → Explanation Generation
- **Memory Palace Neural Network (MPNN)**: Spatial reasoning over a 8×8×8 Chess Cube knowledge graph
- **Symbolic Solver (SymPy)**: Guarantees mathematical rigor with verified symbolic execution
- **Tier 2 Memory Palace**: Hierarchical storage for generated explanations and worked examples
- **Path Derivation**: Logical sequences through concept space (e.g., Calculus → Differentiation → Power Rule)

### Traditional Features (Legacy)
- **DIM-Net**: Neural networks mapping mathematical semantics to 3D coordinates
- **Spatial Reasoning**: Chess cube lattice for mathematical relationship modeling
- **Custom Loss Functions**: Mathematical accuracy, logical consistency, and geometric optimization
- **PAO Mnemonic Generation**: Vivid mnemonics for mathematical formulas and theorems

## 🏗️ Architecture Overview

### Neuro-Symbolic RAG Pipeline

```
┌──────────────┐
│ User Query   │
│ "Find d/dx   │
│  of x²"      │
└──────┬───────┘
       │
       ▼
┌─────────────────────────────────────────────────────────┐
│ Stage A: sLLM Interpreter (Intent Parsing)              │
│ • Extracts: intent=differentiation, expr=x²             │
│ • Generates: Problem Vector (P-Vec)                     │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│ Stage B: Memory Palace Neural Network (MPNN)            │
│ • Spatial Proximity Search → Goal Loci: "power_rule"   │
│ • Path Derivation: [calculus, differentiation]         │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│ Stage C: Symbolic Solver (SymPy)                        │
│ • Executes path: diff(x², x) → 2x                      │
│ • Verifies each step symbolically                       │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│ Stage D: sLLM Explainer (Natural Language)              │
│ • Generates pedagogical explanation                     │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│ Tier 2 Memory Palace (Consolidation)                    │
│ • Stores explanation at coords (2,5,4)                  │
│ • Links to path: [calculus, differentiation]           │
└─────────────────────────────────────────────────────────┘
```

## 📦 Installation & Setup

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- SymPy (for symbolic math)
- Ollama (optional, for LLM features)

### Quick Setup

```bash
# Clone and enter directory
cd /path/to/memory-ai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Optional: Install Ollama for LLM features
# Visit https://ollama.ai and run: ollama pull tinyllama
```

## 🚀 Usage

### Quick Start (No LLM Required)

```bash
python neuro_symbolic_rag.py
```

This will run demo queries:
- Find the derivative of x²
- Find the derivative of x³ + 2x
- Integrate 2x

### Python API

```python
from neuro_symbolic_rag import NeuroSymbolicRAG

# Initialize (works without Ollama)
rag = NeuroSymbolicRAG(use_ollama=False)

# Solve a problem
result = rag.solve("Find the derivative of x^3")

print(f"Result: {result['result']}")  # 3*x**2
print(f"Explanation: {result['explanation']}")
print(f"Tier 2 Location: {result['tier2_location']}")  # 2_5_4
```

### Supported Operations

- **Differentiation**: `"Find the derivative of x^2 + 3*x"`
- **Integration**: `"Integrate 2*x + 1"`
- **More coming soon**: Equation solving, simplification, etc.

## 🧠 Core Components

### 1. Memory Palace Neural Network (MPNN)
**File**: `mnemonic_model.py`

The MPNN maintains a knowledge graph of mathematical concepts:
```python
knowledge_graph = {
    "calculus": ["differentiation", "integration", "limits"],
    "differentiation": ["power_rule", "chain_rule", "product_rule"],
    "algebra": ["linear_equations", "quadratic_equations"],
    ...
}
```

Each concept is mapped to 3D coordinates in the Chess Cube:
- `calculus` → (2, 5, 3)
- `differentiation` → (2, 5, 4)
- `power_rule` → (2, 6, 4)

### 2. Symbolic Solver
**File**: `symbolic_solver.py`

Executes mathematical operations using SymPy:
```python
solver = SymbolicSolver()
result, history = solver.execute_path("x**2", ["calculus", "differentiation"])
# Result: "2*x"
# History: ["Initial State: x**2", "Applied differentiation: 2*x"]
```

### 3. Tier 2 Memory Palace
**File**: `mnemonic_model.py` (Tier2Lattice class)

A "shadow" memory palace that stores generated knowledge:
- **Tier 1**: Foundational concepts and axioms (read-only)
- **Tier 2**: Generated explanations and worked examples (writable)

Each Tier 2 entry includes:
- The explanation text
- The path sequence that generated it
- Timestamp metadata

### 4. sLLM Wrapper (Optional)
**File**: `sllm_wrapper.py`

Interfaces with local Ollama for:
- Natural language intent parsing
- Explanation generation
- Embedding generation

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
