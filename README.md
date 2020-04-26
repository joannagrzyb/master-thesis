# Imbalanced data stream classification with the changing prior probabilities

Master thesis implementation 

## Setup

### Data

1. Generated from stream-learn 

   * Number of class: 2
   * Number of streams: 10 (random state has changed in range 1000 - 1550)
   * Number of concept drifts: 5
   * Types of concept drifts: sudden, incremental
   * Stationary imbalance ratio: 10%, 20%, 30%, 40%, 50%
   * Dynamically imbalance ratio: 10%, 20%, 30%, 40%
   * Number of samples: 10000 (where number of chunks: 200 and chunk size: 500)
   * Number of features: 20 (where informative: 15 and redundant: 5)

### Methods

1. HDWE (Hellinger Distance Weighted Ensemble) - own contribution
2. AWE (Accuracy-Weighted Ensemble)
3. Gaussian Naive Bayes - base classifier

### Evaluation

1. Test Then Train

### Metrics

1. Balanced accuracy score
2. F1 score
3. Recall
4. Specificity

## Experiments

1. Compare 2 methods HDWE and AWE with different imbalance ratio.

## Example results

### Dynamically imbalance

![p_gen_sudden_d_ir(2, 5, 0.6)_f1_score_rs1000.png](https://github.com/joannagrzyb/master-thesis/blob/master/results/plots/gen/sudden/f1_score/p_gen_sudden_d_ir(2,%205,%200.6)_f1_score_rs1000.png?raw=true)