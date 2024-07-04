# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %load_ext autoreload

# %autoreload 2

# # Simulations
#
# This notebook is for running the simulations. 
# Includes several steps: 
# 1. Generating Different Random Graphs
# 2. Simulating different Complex and Simple Contagion Processes on these Graphs and their 2nd Degree Neighborhood
# 3. Obtaining Results for the Simulations
# 4. Validating this on real data
#
# Each of these steps has different open questions
#
# 1. Which Random Graphs?
# - Directed and Undirected? 
# - Preferential Attachment, SBM, Erdos-Reny (here we could even proof what happens)
# - Other considerations that might mimick twitter, such as the configuration model?
#
# 2. Complex and Simple Contagion Processes: Just completely random, or fix the initial conditions?
# - Trying the same seeds?
# - Varying the threshold parameter to not be the same for all users
#
# 3. What kind of results?
# - Epidemic Size (Ratio of infected nodes)
# - Epidemic Length 

import networkx as nx
from src.graph_utils import second_degree_neighbor
# generate exampel graph (like in the paper)
sizes = [100, 100, 100]
probs = [[0.3, 0.05, 0.05], [0.05, 0.3, 0.05], [0.05, 0.05, 0.30]]
G = nx.stochastic_block_model(sizes, probs, seed=0, directed=True)
# generate the second degree neighborhood graph 
G_ext = second_degree_neighbor(G)

print(nx.density(G), nx.density(G_ext))#okay thats a bit problematic

from src.contagion_models import run_simulations
simple = run_simulations(graph=G)
squared = run_simulations(graph=G_ext)

# !pip install matplotlib seaborn

# +
import matplotlib.pyplot as plt
import seaborn as sns
fig, ax = plt.subplots(1,2, figsize=(9,5))

sns.lineplot(sim_df, x="infection_probability", y="epidemic_length", ax=ax[0])
sns.lineplot(sim_df, x="infection_probability", y="epidemic_size", ax=ax[1], c="r")
plt.show()
