# Behavior-Aware Optimization

Experimental exploration of convolutional neural networks (CNNs) combined with lightweight attention and behavioral fitness optimization on structured sensor data.

---

## Overview

This repository documents an exploratory research prototype focused on understanding how different architectural and behavioral design choices affect model robustness and decision reliability.

The work began as a personal effort to deeply understand CNN behavior beyond standard classification pipelines. The system was gradually extended with attention mechanisms and a simple evolutionary search strategy to study how architectural variations influence worst-case performance.

Rather than optimizing only for average accuracy, the model evaluates configurations using behavioral metrics such as worst-case loss and loss variance.

---

## Motivation

Typical CNN experiments focus primarily on:

* Accuracy
* Loss minimization
* Benchmark performance

This project instead explores:

* How architectural choices influence **failure behavior**
* Whether simple evolutionary search can discover more stable configurations
* How pooling strategies affect robustness
* How attention layers provide global context for spatial sensor grids

The goal is learning-driven experimentation, not production deployment.

---

## Core Ideas Explored

### 1. CNN Feature Extraction

* Small spatial sensor grid treated as structured input
* Experiments with:

  * Kernel sizes
  * Strides
  * Channel depth

### 2. Alternative Pooling Behavior

Instead of only traditional max/avg pooling:

* Average pooling
* Minimum-based pooling
* Behavioral impact comparison

This was used to study how pooling affects sensitivity to local spikes and noise.

### 3. Lightweight Attention Layer

A small multi-head attention block is used before convolution to:

* Capture global context across sensors
* Reduce reliance on purely local receptive fields

### 4. Behavioral Fitness Objective

Model configurations are evaluated using:

* Worst-case sample loss
* Loss variance

This shifts optimization from:

> "best average accuracy"
> to
> "more reliable decision behavior"

### 5. Genetic Algorithm Architecture Search

A small evolutionary loop explores:

* Channel count
* Kernel size
* Pooling type

The objective is behavioral stability rather than only accuracy.

---

## Repository Structure

```
data/        Synthetic structured sensor dataset generation
models/      Attention + CNN + behavioral optimization pipeline
analysis/    Training and evaluation visualizations
outputs/     Generated datasets, artifacts, and saved models
```

---

## Dataset

A synthetic sensor grid is generated with:

* Spatial correlation
* Gaussian noise
* Random spikes
* Dropout events

Three behavioral regimes are simulated to study decision boundaries under noise.

The dataset is intended for controlled experimentation, not real-world deployment.

---

## Evaluation Approach

Instead of only reporting accuracy, the system tracks:

* Worst-case loss
* Loss variance
* 95th percentile loss
* Regret relative to baseline configuration

This allows comparison of **behavioral robustness** across architectures.

---

## Status

Research prototype.

This repository represents an ongoing exploration into:

* CNN behavior under noisy structured inputs
* Hybrid attention–convolution pipelines
* Behavior-aware optimization strategies

---

## Citation

If you use this code, methodology, or ideas in academic work or research prototypes, please cite this repository and provide proper attribution.

---

## License

MIT License © 2026 M.H. Karthik
