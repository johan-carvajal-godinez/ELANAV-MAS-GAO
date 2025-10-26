# Multi-Agent Topology Optimization for Space Software Architectures

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

> **Genetic Algorithm-based optimization for multi-agent system (MAS) topologies in space applications**

This repository contains the implementation of a genetic algorithm approach for optimizing communication topologies in multi-agent software architectures, specifically designed for space missions. The algorithm minimizes communication costs while satisfying organizational and reliability constraints.

## 📋 Overview

The efficiency of multi-agent software architectures fundamentally depends on the organization of agents and their interaction strategies. This implementation addresses the NP-hard problem of finding optimal topologies for multi-agent systems by using genetic algorithms to identify configurations that minimize communication costs while ensuring rapid consensus and maintaining critical interactions.

**Key Features:**
- **80%+ improvement** in solution time compared to brute-force approaches
- Graph-based modeling of agent interactions using adjacency matrices
- Support for fixed and flexible interaction constraints
- Hierarchical agent organization (Management, Coordination, Functional)
- Energy-optimized communication topologies for space applications

## 🚀 Use Cases

This implementation was developed for the **ELANaV Project** (Code 1360059) to enable embedded software for navigation in critical mission space systems. Primary application: **sample-return rover missions** operating in hostile environments (Moon, Mars).

### Operational Scenarios
- **Safe Mode**: 10 agents with minimal constraints
- **Nominal Mode**: 20 agents with standard operational constraints
- **Critical Mode**: 30 agents with enhanced fault tolerance requirements

## 🏗️ Architecture

### System Model

The multi-agent system is represented as an undirected graph **G = (V, E, I)** where:
- **V**: Set of agent nodes
- **E**: Communication links between agents
- **I**: Fixed interactions prescribed by design

The topology is encoded as a vector **x** representing the adjacency matrix with dimensions **n(n-1)/2** for **n** agents.

### Agent Roles

1. **Management Agent (AMS)**: Maintains communication with all agents
2. **Coordination Agents**: Organize hierarchical structures and teams
3. **Functional Agents**: Execute specific mission tasks

### Optimization Objective

Minimize total communication cost:

\[
C_I = \sum_{i=1}^{n-1} \sum_{j=i+1}^{n} c_{ij} A_{ij}(x^*)
\]

where \(c_{ij}\) represents the communication cost between agents \(i\) and \(j\), and \(A_{ij}\) indicates connectivity (0 or 1).

## 🛠️ Implementation

### Genetic Algorithm Components

#### 1. Chromosome Encoding
- Each topology is encoded as a binary vector (chromosome)
- **Golden Genes**: Immutable genes representing fixed constraints
- Dimension: **n(n-1)/2** for **n** agents

#### 2. Fitness Function

\[
\text{Fitness}(x^*) = 
\begin{cases} 
0 & \text{if constraints not satisfied} \\
\frac{\sum_{j=1}^{s} F_j(x^*)}{s} & \text{otherwise}
\end{cases}
\]

#### 3. Selection Method
Tournament selection with configurable tournament size (TSS)

#### 4. Genetic Operators
- **Crossover**: Probability-based (PC parameter)
- **Mutation**: Balanced exploration rate
- **Elitism**: Preserves top-performing solutions

### Constraints

1. **Fixed Interactions**: Management agent connects to all others
2. **Hierarchical Structures**: Predefined team organizations
3. **Degree Constraints**:
   - Management: deg(AMS) = n-1
   - Coordination: deg(C) ≤ (n-1)/2 + 1
   - Functional: deg(F) ≤ 3 + round(n/10)

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/mas-topology-optimization.git
cd mas-topology-optimization

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements
- Python 3.8+
- NumPy
- NetworkX
- Matplotlib
- SciPy

## 🎯 Usage

### Basic Example

```python
from mas_optimizer import GeneticAlgorithmOptimizer

# Define system parameters
num_agents = 10
cost_matrix = np.ones((num_agents, num_agents))  # Symmetric unit costs
hierarchies = [(1, [2, 3, 4])]  # Agent 1 manages agents 2, 3, 4
teams = [(2, [3, 5])]  # Agent 2 coordinates team with agents 3 and 5

# Initialize optimizer
optimizer = GeneticAlgorithmOptimizer(
    num_agents=num_agents,
    cost_matrix=cost_matrix,
    hierarchies=hierarchies,
    teams=teams,
    population_size=100,
    elite_size=10,
    tournament_size=5,
    mutation_rate=0.05
)

# Run optimization
best_topology, best_cost, generations = optimizer.optimize(max_generations=1000)

# Visualize result
optimizer.visualize_topology(best_topology)
```

### Advanced Configuration

```python
# Configure GA parameters
config = {
    'population_size': 100,      # Number of candidate solutions
    'elite_chromosomes': 10,     # Top solutions preserved
    'tournament_size': 5,        # Selection pressure
    'mutation_rate': 0.05,       # Exploration factor
    'crossover_prob': 0.8,       # Exploitation factor
    'max_generations': 1000,     # Stopping criterion
    'convergence_threshold': 0.01  # Early stopping
}

optimizer = GeneticAlgorithmOptimizer(**config)
```

## 📊 Performance

### Results Summary

| Scenario | Agents | Generations (Range) | Solution Time | Communication Cost Reduction |
|----------|--------|---------------------|---------------|------------------------------|
| Safe     | 10     | 4-83                | < 1s          | 20-24 power units            |
| Nominal  | 20     | 15-150              | 2-5s          | 40-50 power units            |
| Critical | 30     | 30-250              | 5-15s         | 60-75 power units            |

### Key Findings

- **No significant difference** in optimal cost across parameter configurations (validated by two-sample t-test)
- **Consistent convergence** to global solutions regardless of GA configuration
- **Variance increases** with problem size due to larger feasible solution space
- **Solution quality improves** with extended evolutionary runs

## 🔬 Experimental Design

Parameter variations tested:
- **Constraints**: Minimal (full connectivity) vs. Nominal (scenario-specific)
- **Hierarchies**: Minimal (single fixed interaction) vs. Nominal (full structure)
- **Team Structures**: Minimal vs. Nominal
- **Agent Counts**: 10, 20, 30

Each configuration was tested **30 times** to ensure statistical significance.

## 📈 Example Output

```
Optimization Results:
=====================
Best Communication Cost: 8 power units
Fitness Score: 0.92
Generations Required: 47
Execution Time: 0.34 seconds

Topology Summary:
- Total Connections: 12
- Management Connections: 4
- Coordination Connections: 6
- Functional Connections: 2
```

## 🔧 Configuration Files

### Cost Matrix Format
```json
{
  "agents": 10,
  "cost_matrix": [
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
    ...
  ]
}
```

### Constraint Definition
```json
{
  "fixed_interactions": {
    "management_agent": 1,
    "connected_to": [2, 3, 4, 5, 6, 7, 8, 9, 10]
  },
  "hierarchies": [
    {"coordinator": 2, "team": [3, 5]}
  ],
  "degree_constraints": {
    "management": "n-1",
    "coordination": "floor((n-1)/2) + 1",
    "functional": "3 + round(n/10)"
  }
}
```

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/

# Run specific test suite
python -m pytest tests/test_genetic_algorithm.py

# Run with coverage
python -m pytest --cov=mas_optimizer tests/
```

## 📚 Documentation

Detailed documentation is available in the `/docs` directory:
- **Algorithm Details**: Theory and mathematical formulation
- **API Reference**: Complete function and class documentation
- **Tutorial**: Step-by-step guide with examples
- **Case Studies**: Rover mission scenarios

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/YourFeature`)
3. Commit your changes (`git commit -m 'Add YourFeature'`)
4. Push to the branch (`git push origin feature/YourFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

This work was supported by **Project ELANaV (Code 1360059)** with funding from the Research Vice-Chancellor's office of the **Costa Rica Institute of Technology (TEC)**.

### Author
**Johan Carvajal-Godinez**  
Space Systems Laboratory (SETECLab)  
School of Electronic Engineering  
Costa Rica Institute of Technology  
Email: johcarvajal@itcr.ac.cr

## 📖 Citation

If you use this implementation in your research, please cite:

```bibtex
@inproceedings{carvajal2025multiagent,
  title={Multi-Agent Topology Optimization for Space Software Architectures Using Genetic Algorithms},
  author={Carvajal-Godinez, Johan},
  booktitle={Proceedings of [Conference Name]},
  year={2025},
  organization={Costa Rica Institute of Technology}
}
```

## 🔗 Related Work

- [Thesis: Agent-based architectures supporting fault-tolerance in small satellites (2021)](https://repository.tudelft.nl/)
- [ELANaV Project](https://www.tec.ac.cr/investigacion)

## 📞 Contact & Support

- **Issues**: Please use the [GitHub issue tracker](https://github.com/yourusername/mas-topology-optimization/issues)
- **Email**: johcarvajal@itcr.ac.cr
- **Research Group**: [SETECLab - Space Systems Laboratory](https://www.tec.ac.cr/)

## 🗺️ Roadmap

- [ ] Implement Particle Swarm Optimization (PSO) comparison
- [ ] Sensitivity analysis for parameter optimization
- [ ] Real-world communication cost matrix estimation
- [ ] Integration with FreeFlyer/GMAT mission planning tools
- [ ] Multi-objective optimization framework
- [ ] Real-time reconfiguration capabilities
- [ ] Hardware-in-the-loop testing with embedded systems

---

**Project Status**: Active Development | **Version**: 1.0.0 | **Last Updated**: October 2025
