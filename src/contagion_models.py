import random
from tqdm import tqdm
import pandas as pd
import numpy as np
import networkx as nx
from scipy.stats import truncnorm


#models for simple diffusion
def si_step(graph, infected_node, prob):
    contact_list = graph.edges(infected_node)
    for infecter, contact in contact_list:
        if graph.edges[(infecter, contact)]["evaluated"] is False:
            if random.random() < prob:
                nx.set_node_attributes(graph, {contact: {"infected": True}})
            nx.set_edge_attributes(graph, {(infecter, contact): {"evaluated": True}})


def si_full(graph, prob):
    patient_zero = random.choice(range(len(graph.nodes)))
    cur_infected = 1
    nx.set_node_attributes(graph, {patient_zero: {"infected": True}})
    it = 1
    while True:
        infected_nodes = [n for n, v in graph.nodes.data() if v['infected'] == True]
        for infected in infected_nodes:
            si_step(graph, infected, prob)

        new_infected = sum(dict(graph.nodes.data("infected")).values())

        if cur_infected == new_infected:
            break
        else:
            it += 1
            cur_infected = new_infected

    s = new_infected / len(graph)
    return it, s


def add_spreading_params(G):

    nx.set_edge_attributes(G, False, "evaluated")
    nx.set_node_attributes(G, False, "infected")


def one_simulation(graph, prob):
    add_spreading_params(graph)
    return si_full(graph, prob)


def run_simulations(graph, runs=100):
    ps = np.arange(0, 1, 0.01)

    return_dict = {"infection_probability": ps,
                   "epidemic_length": [],
                   "epidemic_size": []}
    for p in tqdm(ps, desc="Probablity", position=0):
        l_list = []
        s_list = []
        for _ in range(runs):
            l, s = one_simulation(graph, p)
            l_list.append(l)
            s_list.append(s)
        return_dict["epidemic_length"].append(np.mean(l_list))
        return_dict["epidemic_size"].append(np.mean(s_list))

    return pd.DataFrame(return_dict)



def threshold_diffusion(graph, lam, optimal_pair):
    # Initialize converted nodes list
    converted_list = optimal_pair[:]

    # Generate threshold values
    num_nodes = len(graph.nodes)
    threshold = truncnorm.rvs(a=0, b=np.inf, loc=lam, scale=0.5, size=num_nodes)

    for period in range(4):
        # Initialize list for newly converted nodes
        new_converted = []

        for node in graph.nodes:
            if node not in converted_list:
                weight = 0
                # Calculate weight from neighbors
                for neighbor in graph.neighbors(node):
                    if neighbor in converted_list:
                        weight += 1

                if weight > threshold[node]:
                    new_converted.append(node)

        converted_list.extend(new_converted)

        if set(converted_list) == set(optimal_pair):
            break

    conversion_rate = len(converted_list) / num_nodes
    return conversion_rate, converted_list


