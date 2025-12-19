# Sparse SVM for Text Classification

## Motivation
I wanted to understand how proximal optimization behaves in high-dimensional
text classification problems. In particular, I was interested in how
L1 regularization induces sparsity in linear models and how this affects
generalization.

This project is a simplified implementation aimed at building intuition
around sparsity-inducing optimization techniques rather than achieving
state-of-the-art performance.

## Overview
- TF-IDF features for text representation
- Linear SVM with hinge loss
- L1 regularization implemented via proximal updates
- Evaluation using F1 score

## Notes
The focus of this project is on understanding learning dynamics and
optimization behavior in sparse models.

