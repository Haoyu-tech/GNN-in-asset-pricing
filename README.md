# Autoencoder Asset Pricing with SIC-based Graph Neural Networks

Reproduction and extension of the **Autoencoder Asset Pricing** framework with **Graph Neural Networks (GNN)** using **SIC industry classification**.

This project reproduces the deep asset pricing model proposed by **Kelly, Pruitt, and Su** and extends it by incorporating **industry graph structures** through **Graph Neural Networks** to improve cross-sectional asset pricing.

---

# Overview

Recent advances in machine learning have enabled flexible nonlinear asset pricing models. One influential approach is the **Autoencoder Asset Pricing framework**, which uses deep neural networks to estimate **latent pricing factors** and **firm-specific loadings**.

However, most deep asset pricing models treat firms as **independent observations**, ignoring economic relationships such as:

- industry membership
- supply chain linkages
- product market similarity
- macro exposure clustering

This project introduces a **graph-based extension** where firms are connected based on **SIC industry classification**, and a **Graph Neural Network (GNN)** learns cross-sectional dependencies between firms.

The project contains two main components:

1. **Reproduction of Autoencoder Asset Pricing**
2. **GNN-based extension using SIC industry graphs**

---

# Model Framework

## Baseline: Autoencoder Asset Pricing

The baseline follows the deep latent factor model structure.

Input:

- firm characteristics \( z_{i,t} \)

Encoder:

- nonlinear network mapping characteristics to factor loadings

Latent factors:

- low-dimensional pricing factors

Return reconstruction:

\[
\hat{r}_{i,t+1} = \beta_{i,t}^\top f_{t+1}
\]

where

- \( \beta_{i,t} \) is predicted from firm characteristics
- \( f_t \) are latent pricing factors

The model learns:

- nonlinear mapping from characteristics to betas
- latent factor structure in asset returns

---

## Extension: Graph Neural Network Pricing Model

We construct a **firm graph** using SIC industry classification.

Graph structure:

- Node: firm
- Edge: firms sharing the same SIC industry
- Node features: firm characteristics

The GNN aggregates information across firms in the same industry:

\[
h_i^{(k+1)} = \sigma \left( \sum_{j \in N(i)} W h_j^{(k)} \right)
\]

This allows the model to learn **industry-aware firm representations**.

Possible implementations include:

- GCN
- GraphSAGE
- GAT
- Temporal GNN

---

# Project Goals

The main research questions are:

1. Can the Kelly-style autoencoder asset pricing model be successfully reproduced?
2. Does incorporating **industry graph structure** improve pricing accuracy?
3. Can **GNN-based firm embeddings** enhance cross-sectional return prediction?
4. Does graph-based pricing reduce pricing errors relative to the baseline model?

---

# Data

The project is designed for standard empirical asset pricing datasets.

Typical data sources include:

- CRSP monthly returns
- Compustat accounting data
- Firm characteristics (e.g. Gu, Kelly & Xiu dataset)
- Risk-free rate
- SIC industry classification

Example firm characteristics:

- size
- book-to-market
- momentum
- profitability
- investment
- volatility
- liquidity
- leverage

Industry graph construction:

- SIC 2-digit
- SIC 3-digit
- SIC 4-digit

---

# Methodology

## Step 1 — Data preprocessing

- merge CRSP and Compustat
- construct firm characteristic panel
- align firm-month observations
- normalize characteristics cross-sectionally
- winsorize extreme values
- split train / validation / test sets

---

## Step 2 — Reproduce autoencoder asset pricing

Implement baseline deep asset pricing model.

Components:

- encoder network
- latent factor estimation
- return reconstruction
- training via reconstruction loss

---

## Step 3 — Construct SIC industry graph

Possible graph structures:

Binary graph

- firms connected if same SIC

Hierarchical graph

- stronger edges for closer SIC levels

Dynamic graph

- update graph over time

---

## Step 4 — Train GNN asset pricing model

Node features:

- firm characteristics

Edges:

- SIC-based industry relations

Outputs:

- firm embeddings
- factor loadings
- predicted returns

---

## Step 5 — Model evaluation

Models are evaluated using both **statistical** and **economic** metrics.

Statistical metrics:

- Mean Squared Error
- Mean Absolute Error
- Out-of-sample \(R^2\)

Asset pricing metrics:

- pricing error
- factor alpha
- cross-sectional fit

Economic metrics:

- Sharpe ratio
- Sortino ratio
- maximum drawdown
- long-short portfolio returns

---

# Repository Structure
