# Advanced Mathematical Calculator - User Guide

## 🎯 Overview

The Neuro-Symbolic Memory Palace is now a **sophisticated general mathematics calculator** capable of:

- ✅ **Calculus**: Derivatives, integrals, limits, series expansions
- ✅ **Algebra**: Simplification, expansion, factoring, equation solving
- ✅ **Linear Algebra**: Matrix operations, eigenvalues, determinants
- ✅ **Formula Creation**: Define and store custom formulas for future use
- ✅ **Symbolic Computation**: Exact symbolic answers (not numerical approximations)

## 📚 Supported Operations

### Calculus

#### Differentiation
```
Find the derivative of x^2 + 3*x
Find the derivative of sin(x)*cos(x)
d/dx e^(x^2)
```

#### Integration
```
Integrate 2*x + 1
Integrate x^2*sin(x)
∫ e^x dx
```

#### Limits
```
Find the limit of (x^2 - 1)/(x - 1) as x approaches 1
Limit of sin(x)/x as x approaches 0
```

#### Series Expansions
```
Taylor series of e^x
Series expansion of sin(x) around 0
```

### Algebra

#### Simplification
```
Simplify: (x + 1)^2 - (x - 1)^2
Simplify: (a + b)^3
```

#### Expansion
```
Expand: (x + 2)*(x + 3)
Expand: (a + b + c)^2
```

#### Factoring
```
Factor: x^2 - 5*x + 6
Factor: x^3 - 8
```

#### Equation Solving
```
Solve equation: x^2 - 4 = 0
Solve equation: 2*x + 5 = 11
Solve: x^3 - 6*x^2 + 11*x - 6 = 0
```

### Linear Algebra

```
Matrix multiplication
Find determinant of [[1,2],[3,4]]
Find eigenvalues of matrix
Matrix inverse
```

### Word Problems

The system now supports natural language word problems:

```
A farmer has 5 boxes with 3 kittens each. How many kittens are there?
There were 7 apples. 2 were eaten. How many remain?
John has 12 marbles. He gives 5 to Mary. How many does he have left?
```

**Features:**
- **AI-Powered Parsing**: Uses DistilBERT for operation classification
- **Entity Extraction**: Identifies numbers, nouns, and relationships
- **Direct Solving**: Immediate answers for common patterns
- **Symbolic Fallback**: Complex problems use symbolic computation

## 🔬 Advanced Features

### 1. Custom Formula Creation

Create your own formulas and store them for future use:

```python
# In Python API
solver.create_formula(
    name="kinetic_energy",
    expression="0.5*m*v**2",
    description="Kinetic energy formula"
)

# Apply the formula
result = solver.apply_formula("kinetic_energy", m=10, v=5)
# Result: 125.0
```

### 2. Multi-Variable Support

The system automatically detects variables:

```
Find the derivative of x*y + z^2  # Differentiates with respect to x
Integrate t^2 + 3*t  # Integrates with respect to t
```

### 3. Symbolic Results

All results are exact symbolic expressions:

```
∫ x^2 dx = x^3/3  (not 0.333...)
√2 stays as √2 (not 1.414...)
```

### 4. Tier 2 Knowledge Base

Every solved problem is stored in the Tier 2 Memory Palace:
- **Location**: 3D coordinates in the Chess Cube
- **Path**: The conceptual journey taken
- **History**: Step-by-step execution log
- **Reusable**: Can be retrieved for similar problems

### 5. Branched Retrieval (NEW)

Enhanced memory palace with associative retrieval:
- **Neighbor Search**: Retrieves explanations from adjacent 3D locations
- **Contextual Enrichment**: Provides richer explanations using related concepts
- **Associative Memory**: Connects similar problems for better understanding
- **Configurable Radius**: Search neighboring locations within specified distance

### 1. Custom Formula Creation

Create your own formulas and store them for future use:

```python
# In Python API
solver.create_formula(
    name="kinetic_energy",
    expression="0.5*m*v**2",
    description="Kinetic energy formula"
)

# Apply the formula
result = solver.apply_formula("kinetic_energy", m=10, v=5)
# Result: 125.0
```

### 2. Multi-Variable Support

The system automatically detects variables:

```
Find the derivative of x*y + z^2  # Differentiates with respect to x
Integrate t^2 + 3*t  # Integrates with respect to t
```

### 3. Symbolic Results

All results are exact symbolic expressions:

```
∫ x^2 dx = x^3/3  (not 0.333...)
√2 stays as √2 (not 1.414...)
```

### 4. Tier 2 Knowledge Base

Every solved problem is stored in the Tier 2 Memory Palace:
- **Location**: 3D coordinates in the Chess Cube
- **Path**: The conceptual journey taken
- **History**: Step-by-step execution log
- **Reusable**: Can be retrieved for similar problems

## 🚀 Usage Examples

### Example 1: Polynomial Calculus
```
Query: "Find the derivative of x^3 + 2*x^2 - 5*x + 1"

Result: 3*x^2 + 4*x - 5

Path: calculus → differentiation → power_rule
Tier 2 Location: 2_5_4
```

### Example 2: Trigonometric Integration
```
Query: "Integrate sin(x)*cos(x)"

Result: sin(x)^2/2

Path: calculus → integration → substitution
Tier 2 Location: 2_4_3
```

### Example 3: Algebraic Factoring
```
Query: "Factor: x^2 - 5*x + 6"

Result: (x - 2)*(x - 3)

Path: algebra → factor
Tier 2 Location: 1_2_5
```

### Example 4: Equation Solving
```
Query: "Solve equation: x^2 - 4 = 0"

Result: [-2, 2]

Path: algebra → quadratic_equations → quadratic_formula
Tier 2 Location: 1_3_3
```

### Example 5: Word Problems
```
Query: "A farmer has 5 boxes with 3 kittens each. How many kittens are there?"

Result: 15

Path: word_problem_parsing
Tier 2 Location: word_problem_arithmetic
Branched Context: Related multiplication concepts from neighboring locations
```

## 🎨 GUI Features

Access the web interface at `file:///home/duck/Documents/memory-ai/gui.html`

Features:
- **Quick Examples**: Click buttons for common operations
- **Real-time Pipeline**: Watch the 4-stage process
- **Execution Log**: See every step taken
- **Path Visualization**: View the conceptual journey
- **Tier 2 Storage**: See where results are stored

## 🔧 Python API

For programmatic access:

```python
from neuro_symbolic_rag import NeuroSymbolicRAG

# Initialize
rag = NeuroSymbolicRAG(use_ollama=False)

# Solve a problem
result = rag.solve("Find the derivative of x^3")

print(result['result'])  # 3*x**2
print(result['path'])    # ['calculus', 'differentiation']
print(result['tier2_location'])  # 2_5_4
```

## 📈 Future Capabilities

The system can be extended to support:

- **Differential Equations**: ODEs, PDEs
- **Complex Analysis**: Complex numbers, contour integration
- **Probability**: Distributions, statistical inference
- **Optimization**: Lagrange multipliers, gradient descent
- **Numerical Methods**: When symbolic solutions don't exist
- **Proof Verification**: Check mathematical proofs
- **Formula Discovery**: Learn patterns from solved problems
- **Advanced Word Problems**: Multi-step reasoning, geometry, physics applications

## 💡 Tips for Best Results

1. **Be Specific**: "Find the derivative of x^2" works better than "derivative"
2. **Use Standard Notation**: `x^2` for powers, `*` for multiplication
3. **Check the Path**: The conceptual path shows how the problem was solved
4. **Review Tier 2**: See what formulas have been stored
5. **Create Formulas**: Store frequently-used expressions
6. **Word Problems**: Use natural language; the AI parser handles common patterns
7. **Branched Context**: More complex problems benefit from associative retrieval

## 🎯 Next Steps

To make this even more sophisticated:

1. **Train DIM-Net**: Better concept mapping
2. **Expand Knowledge Graph**: Add more mathematical domains
3. **Add Numerical Solvers**: For when symbolic fails
4. **Implement Proof System**: Verify mathematical statements
5. **Multi-Step Problems**: Chain multiple operations
6. **Formula Discovery**: Learn new formulas from patterns
7. **Enhanced AI Parsing**: Support for more complex word problem types

---

**Current Version**: 4.2.0 - AI-Enhanced Word Problem Solver with Branched Memory
**Last Updated**: November 27, 2025
