# Feature Flow: MoE for OOD Graph Learning

This document outlines the complete feature flow for building the Mixture-of-Experts (MoE) model for Out-of-Distribution (OOD) generalization in graph classification.

---

## Overview
The feature flow is designed to allow modular development and controlled evaluation. Each stage builds upon the previous one, enabling ablation and integration of advanced techniques such as expert-specific augmentation and gating strategies.

---

## Stage 1: Baseline GNN
- **Objective:** Establish a strong single-model baseline
- **Components:**
  - Single GNN architecture (GIN, GCN, or GraphSAGE)
  - Trained on GOOD benchmark datasets
- **Outputs:** In-distribution and OOD accuracy for each dataset

---

## Stage 2: Basic MoE (Uniform Aggregation)
- **Objective:** Introduce mixture-of-experts without additional enhancements
- **Components:**
  - Multiple GNN experts with identical architecture
  - Uniform or fixed expert aggregation
  - No augmentation or diversity loss
- **Configurable Parameters:**
  - `num_experts`
  - `aggregation: uniform`
- **Outputs:** Performance comparison with baseline GNN

---

## Stage 3: Gating Mechanism
- **Objective:** Replace uniform aggregation with learned gating
- **Components:**
  - Shared or expert-specific encoders
  - Soft attention or learned voting for expert weighting
- **Configurable Parameters:**
  - `gating: soft_attention | learned_voting`
- **Outputs:** Change in accuracy and gate distribution analysis

---

## Stage 4: Expert Diversity Induction
- **Objective:** Promote specialization among experts
- **Components:**
  - Orthogonality or disagreement regularization loss
  - Diversity metrics for embedding separation
- **Configurable Parameters:**
  - `diversity_loss: true`
- **Outputs:** Expert variance, embedding visualization, improved robustness

---

## Stage 5: Shared Data Augmentation
- **Objective:** Test the effect of standard augmentation on MoE
- **Components:**
  - Node feature masking, edge perturbation, or learned policy
- **Configurable Parameters:**
  - `augmentation.enable: true`
  - `augmentation.strategy: node_masking | edge_perturbation | learned`
- **Outputs:** Performance with and without augmentation

---

## Stage 6: Expert-Specific Augmentation
- **Objective:** Assign a unique augmentation path to each expert
- **Components:**
  - Each expert receives a different view of the input graph
  - Augmentations may be static or learned
- **Configurable Parameters:**
  - `augmentation.expert_specific: true`
- **Outputs:** Enhanced specialization, more diverse predictions

---

## Stage 7: Meta-Regularization and Invariant Learning
- **Objective:** Improve generalization through representation learning
- **Components:**
  - Contrastive loss or information maximization
  - Applied across expert outputs or time steps
- **Configurable Parameters:**
  - `meta_regularization: contrastive | info_max`
- **Outputs:** Improved OOD performance, lower representation overlap

---

## Stage 8: Full Model Integration
- **Objective:** Combine all components for a fully-featured MoE
- **Components:**
  - Expert-specific augmentation
  - Diversity loss
  - Gating mechanism
  - Meta-regularization
- **Configurable Parameters:** All of the above
- **Outputs:** Best-performing model across all GOOD benchmark splits

---

## Stage 9: Extended Experiments
- **Objective:** Probe limits and interactions between modules
- **Examples:**
  - What happens when diversity loss is applied without augmentation?
  - Can learned voting outperform fixed aggregation on specific datasets?
  - Are expert-specific augmentations most useful when combined with gating?

---

This flow ensures systematic evaluation and modular growth of the method, enabling empirical justification for each design choice.