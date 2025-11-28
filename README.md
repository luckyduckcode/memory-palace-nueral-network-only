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

### Data Generation

```bash
python memory_palace_data_generator.py
```

This generates training data by:
1. Classifying facts into 8 knowledge domains
2. Assigning 3D coordinates in the chess cube
3. Creating PAO mnemonics using LLM
4. Storing in Tier 2 for consolidation

### API Usage

```python
from api_server import app
# Run with: python api_server.py

# Or use requests:
import requests
response = requests.post('http://localhost:5000/generate_mnemonic', 
                        json={'fact': 'Your fact here'})
```

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
--- Processing Fact: 'The Earth orbits the Sun once every 365.25 days.' ---
[Stage A] Encoding...
  Semantic Vector: [0.12, 0.45, ...]
  Category: General Knowledge (LOC 000)
[Stage B] Spatial Mapping...
  Assigned Coordinates: (1, 3, 5)
  Color Parity: 1 (Alternating Property)
[Stage C] Mnemonic Generation...
  PAO Mnemonic: "A giant astronaut heroically orbits a blazing sun while juggling 365 colorful balls."
[Stage D] Tier 2 Storage...
  Stored in Tier 2 at (1, 3, 5)

--- FINAL OUTPUT ---
Fact: The Earth orbits the Sun once every 365.25 days.
Coordinates: (1, 3, 5)
Mnemonic: A giant astronaut heroically orbits a blazing sun while juggling 365 colorful balls.
Category: General Knowledge
```

## 🔬 Research Background

This system implements a neural network-enhanced memory palace for general knowledge organization, drawing from traditional mnemonic techniques and modern AI research.

### Key Innovations

1. **3D Spatial Knowledge Graph**: Organizes concepts in a chess cube lattice for intuitive retrieval
2. **AI-Generated Mnemonics**: Uses LLMs to create vivid, personalized PAO associations
3. **Hierarchical Storage**: Tier 1 for facts, Tier 2 for generated content
4. **Adaptive Categorization**: Library of Congress-inspired classification for broad knowledge domains

### Advantages Over Standard Methods

| Feature | Traditional Memory Palace | MPNN System |
|---------|---------------------------|-------------|
| Knowledge Scope | Limited domains | Broad categories (8 major) |
| Mnemonic Generation | Manual | AI-assisted PAO |
| Retrieval | Mental navigation | Spatial + neural search |
| Scalability | 10-50 loci | 512+ 3D locations |
| Persistence | Practice-dependent | Neural consolidation |

## 🧪 Testing

```bash
# Run demo with sample facts
python demo_memory_palace.py

# Generate training data
python memory_palace_data_generator.py

# Test Tier 2 Memory Palace
python test_tier2.py

# Test individual components
python -c "from mnemonic_model import MnemonicModel; print('Model loaded successfully')"
```

### Data Generation

```bash
# Generate expanded facts dataset
python expand_facts_dataset.py

# Create training data with LLM mnemonics
python memory_palace_data_generator.py
```

### API Server

```bash
# Start the web API
python api_server.py

# Access at http://localhost:5000
```

### GUI

Open `gui.html` in a web browser for visualization.

## 📁 Project Structure

```
memory-palace-nn/
├── mnemonic_model.py                    # Core MPNN, DIM-Net, Tier2Lattice
├── train_memory_palace.py               # Training script
├── demo_memory_palace.py                # Demo script
├── memory_palace_data_generator.py      # Data generation with LLM
├── expand_facts_dataset.py              # Dataset expansion
├── cluster_facts.py                     # Fact clustering
├── api_server.py                       # Flask API server
├── gui.html                             # Web GUI
├── sllm_wrapper.py                      # Local LLM wrapper (Ollama)
├── test_tier2.py                        # Tier 2 tests
├── requirements.txt                     # Python dependencies
├── setup_llama.sh                       # Ollama setup script
├── facts_dataset.txt                    # Base facts
├── facts_dataset_expanded.txt           # Expanded facts
├── fact_clusters.json                   # Clustered facts
├── memory_palace_training_data.csv      # Training data
├── memory_palace_demo_results.csv       # Demo results
├── training_history.csv                 # Training logs
├── dim_net_final.pth                    # Trained DIM-Net model
├── trhd_mapper_final.pth                # Trained mapper
├── checkpoint_epoch_*.pth               # Training checkpoints
├── README.md                            # This file
└── .gitignore                           # Git ignore rules
```

## 📚 Documentation

- **[RESEARCH_OVERVIEW.md](RESEARCH_OVERVIEW.md)**: Comprehensive research document explaining the MPNN architecture, methodology, and scientific foundations
- **[ADVANCED_GUIDE.md](ADVANCED_GUIDE.md)**: Advanced usage and technical details

## 🛣️ Roadmap

- [x] 3D Chess Cube Lattice (8x8x8)
- [x] DIM-Net for semantic encoding
- [x] Tier 2 Memory Palace
- [x] PAO Mnemonic generation with LLM
- [x] Expanded knowledge categories (8 domains)
- [ ] VR/AR interface for immersive palaces
- [ ] Adaptive user feedback loops
- [ ] Multi-modal mnemonics (images, audio)
- [ ] Large-scale knowledge graph (1000+ concepts)
- [ ] Real-time neural efficiency monitoring
- [ ] Cross-domain concept linking
- [ ] Mobile app integration

## 🤝 Contributing

Contributions are welcome! Areas of interest:
- Expanding fact datasets across domains
- Improving LLM prompts for better mnemonics
- Adding new neural network architectures
- Enhancing the GUI and API
- Research integration (VR, neuroimaging)

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- **PyTorch**: Neural network framework
- **Ollama**: Local LLM inference
- **Flask**: Web API framework
- **spaCy & NLTK**: Natural language processing
- **Method of Loci**: Ancient mnemonic technique
- Research on AI-enhanced memory (Alshehri, Reddy, Wagner, 2021-2025)

---

**Last Updated**: November 28, 2025  
**Version**: 2.0.0 - General Knowledge MPNN
