Kelly Autoencoder in Asset Pricing with SIC-based GNN Extension
Overview

This project reproduces the autoencoder-based asset pricing framework proposed by Kelly, Pruitt, and Su, and extends it by replacing or augmenting the latent factor extraction module with a Graph Neural Network (GNN) based on SIC industry classification.

The core idea is to combine:

Autoencoder asset pricing for latent factor discovery,

Firm characteristics as cross-sectional predictors,

SIC-based industry graph structure to model economic linkages across firms,

and GNN-based representation learning for improved cross-sectional asset pricing.

This project aims to answer the following questions:

Can the Kelly-style autoencoder framework be successfully reproduced on standard firm-characteristic data?

Can a GNN built on SIC industry relations improve latent representation learning in asset pricing?

Does incorporating industry graph structure enhance pricing accuracy, factor interpretability, and portfolio performance?

Motivation

Traditional linear factor models, such as CAPM or Fama-French, impose strong structure on the pricing kernel. In contrast, the autoencoder in asset pricing framework uses deep learning to estimate latent pricing factors and firm-specific loadings in a flexible nonlinear way.

However, most existing autoencoder asset pricing models treat firms as independent cross-sectional units, ignoring the economic relationships among firms. In reality, companies are connected through:

industry membership,

supply chains,

product market similarity,

and macro exposure clustering.

This project introduces a graph-based asset pricing extension, where firms are linked according to SIC industry classification, and a Graph Neural Network is used to learn richer firm embeddings for pricing.

Project Objectives

The project has two main components:

1. Reproduction of Kelly-style Autoencoder Asset Pricing

We first reproduce the benchmark autoencoder asset pricing model, including:

data preprocessing,

firm characteristic normalization,

latent factor extraction,

beta/loading estimation,

return reconstruction,

and out-of-sample evaluation.

2. GNN-based Extension with SIC Industry Graph

We then extend the framework by constructing a graph over firms using SIC codes and applying a GNN to learn cross-sectional dependence.

Possible graph definitions include:

same 2-digit SIC industry,

same 3-digit SIC industry,

weighted adjacency based on SIC distance,

dynamic graph by year/month.

The GNN replaces or complements the original encoder network to produce firm embeddings or factor loadings used for return prediction and pricing tests.

Model Architecture
Baseline: Autoencoder in Asset Pricing

The baseline model follows the logic of deep latent factor pricing:

Input: firm characteristics 
𝑧
𝑖
,
𝑡
z
i,t
	​


Encoder: maps characteristics into latent representations

Latent factors: low-dimensional pricing factors

Decoder / pricing layer: reconstructs expected returns through factor loadings and latent factors

In simplified form:

𝑟
^
𝑖
,
𝑡
+
1
=
𝛽
𝑖
,
𝑡
⊤
𝑓
𝑡
+
1
r
^
i,t+1
	​

=β
i,t
⊤
	​

f
t+1
	​


where:

𝛽
𝑖
,
𝑡
β
i,t
	​

 is a nonlinear function of firm characteristics,

𝑓
𝑡
+
1
f
t+1
	​

 is a set of latent pricing factors.

Extension: SIC-based GNN for Asset Pricing

In the GNN extension:

each firm is a node,

edges are defined using SIC industry proximity,

node features are firm characteristics,

a GNN aggregates information from economically related firms.

The pipeline becomes:

Construct firm graph using SIC codes

Use node features 
𝑥
𝑖
,
𝑡
x
i,t
	​

 = firm characteristics

Apply GNN layers to obtain firm embeddings

Map embeddings to factor loadings or return forecasts

Evaluate pricing performance against the baseline autoencoder

This allows the model to exploit both:

firm-level characteristics, and

cross-firm industry structure.

Data

This project is designed for standard empirical asset pricing datasets, such as:

CRSP monthly stock returns

Compustat accounting variables

Firm characteristics (e.g. Gu, Kelly, and Xiu style characteristic zoo)

Risk-free rate / benchmark factors

SIC codes for industry classification

Typical variables used

monthly stock returns

market equity

book-to-market

momentum

profitability

investment

volatility-related variables

liquidity variables

leverage

and other anomaly predictors

Industry graph inputs

SIC 2-digit / 3-digit / 4-digit codes

industry adjacency matrix

firm-year or firm-month industry mapping

Methodology
Step 1: Data preprocessing

merge returns, characteristics, and SIC codes

align firm-month observations

winsorize or clip extreme characteristic values

standardize characteristics cross-sectionally

handle missing values

create train / validation / test splits using rolling windows

Step 2: Reproduce the autoencoder baseline

implement encoder-decoder pricing network

estimate latent factors

reconstruct returns

evaluate in-sample and out-of-sample fit

Step 3: Build SIC-based graph

Graph construction options:

Binary graph: connect firms in the same SIC industry

Hierarchical graph: stronger weights for same 3-digit or 4-digit SIC

Temporal graph: update graph monthly/yearly as firm universe changes

Step 4: Train GNN pricing model

Candidate GNN architectures:

GCN

GraphSAGE

GAT

Temporal GNN variants

Possible replacement strategies:

GNN as encoder replacement
Replace the autoencoder encoder with GNN layers

GNN + autoencoder hybrid
Use GNN embeddings together with original firm characteristics

GNN for beta estimation
Predict firm-specific loadings using graph-aware embeddings

Step 5: Evaluation

We compare the baseline and GNN-enhanced models using:

out-of-sample 
𝑅
2
R
2

pricing errors

cross-sectional prediction accuracy

latent factor interpretability

portfolio sorting performance

Sharpe ratio of characteristic-managed strategies
