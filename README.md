# FairIM-GNN: Fair & Diverse Influence Maximization

**FairIM-GNN** is a multi-attribute, graph-based framework designed to solve the problem of Influence Maximization (IM) while ensuring demographic fairness and structural robustness. By integrating adversarial learning with high-order relational modeling, this model selects a diverse set of "seed" nodes that maximize information spread without propagating systemic biases.

## 🚀 Key Features

### Phase 1: The Fairness Foundation
* **Neighborhood-Aware Encoding**: Replaces standard dense layers with **Graph Convolutional (GCN)** and **Graph Attention (GAT)** layers to capture the actual "shape" of the social network.
* **Adversarial Fairness (GRL)**: Uses a **Gradient Reversal Layer** to create a mathematical tug-of-war, forcing the encoder to "scrub" sensitive demographic traits from node embeddings.
* **Covariance Constraint**: Acts as a mathematical safety net by penalizing linear correlations between learned embeddings and sensitive attributes.
* **Graph Contrastive Learning (GCL)**: Enhances robustness by training the model on "distorted" versions of the graph, ensuring stable embeddings even under structural noise.

### Phase 2: Advanced Strategy & Selection
* **Hypergraph Modeling**: Captures group-level influence pathways (e.g., shared college clubs or majors) that traditional pairwise edges miss.
* **Fair K-Means**: Implements a constrained clustering algorithm to ensure every potential seed group is demographically balanced.
* **Maximal Marginal Relevance (MMR)**: A selection strategy that balances individual node influence with spatial diversity to maximize network coverage and reduce redundancy.

## 📂 Project Structure
* `Model.ipynb`: The primary implementation containing the GNN architecture, adversarial training loops, and evaluation metrics.
* `edges.txt`: The social network topology (source and target node pairs).
* `attr.txt`: Node attribute data used for fairness auditing.

## 🛠️ Installation & Requirements
The project is built using **TensorFlow 2.x** and the **Spektral** GNN library.
```bash
pip install tensorflow spektral networkx pandas numpy scikit-learn
```

## 📊 How It Works
1.  **Data Loading**: The model loads the graph and attributes, pruning nodes that fall outside the target demographic focus.
2.  **Generic Pre-training**: The GAT Encoder is trained to understand graph topology, while the Discriminator is pre-trained to identify sensitive traits.
3.  **Adversarial Training**: The unified model runs a minimax optimization. The Encoder tries to maintain graph structure while the GRL/Covariance layers remove bias.
4.  **Seed Selection**: Using Fair K-Means and MMR, the model identifies the top influential nodes across diverse communities.
