# Mixture-of-Experts (MoE) for OOD Graph Learning

This repository implements a Mixture-of-Experts (MoE) architecture for improving Out-of-Distribution (OOD) generalization in graph learning tasks, with a focus on graph classification.

## Overview

The project explores how MoE architectures can improve OOD generalization by:
- Using multiple expert GNNs with specialized behaviors
- Implementing various gating mechanisms for expert selection
- Applying diversity-promoting regularization
- Supporting expert-specific augmentations
- Incorporating meta-regularization techniques