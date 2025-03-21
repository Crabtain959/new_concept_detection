import random
random.seed(42)
import torch
import numpy as np
from tqdm.auto import tqdm
import json
import pandas as pd
import copy
from collections import defaultdict 
import umap
from umap.parametric_umap import ParametricUMAP
from skimage.feature import blob_log
import time
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import backoff
from bokeh.plotting import figure, show, output_notebook
from bokeh.layouts import gridplot
from scipy.ndimage import gaussian_filter

    
import random

dims = (384,)

def embedding_over_period_non_parametric(split_baseline_embs, n_components, v=True):
        
    reducer = umap.UMAP(
        n_components=n_components,
        # n_neighbors=25,
        # min_dist=0.2,
        verbose=v
    )

    combined_emb_transformed = reducer.fit_transform(np.concatenate(split_baseline_embs))


    periods_emb = np.split(combined_emb_transformed, np.cumsum([len(e) for e in split_baseline_embs[:-1]]))

    return periods_emb


def create_heatmaps(periods, sentences, embeddings, normalize, bins=400, show_progress=True):
    
    all_periods = np.concatenate(periods)
    ranges = list(zip(all_periods.min(0), all_periods.max(0)))
    
    del all_periods

    _, edges = np.histogramdd(
        np.array(periods[0]),
        bins=bins,
        range=ranges,
    )
    
    num_components = np.array(periods[0]).shape[1]
    bins_dicts = []
    for period, sentence, embedding in tqdm(zip(periods, sentences, embeddings), total=len(periods), desc='Creating mappings from bins to sentences', disable=not show_progress):
        bins_dict = {}
        dim_bins = [np.digitize(period[:, i], edges[i])-1 for i in range(num_components)]

        for i in range(len(period)):
            bin_key = tuple([dim_bins[j][i] for j in range(num_components)])
            if bin_key not in bins_dict:
                bins_dict[bin_key] = []
            bins_dict[bin_key].append((sentence[i], tuple(embedding[i])))
        bins_dicts.append(bins_dict)
    
    ref_len = len(periods[0])
    
    if normalize:
        histograms = [
            np.histogramdd(
                p,
                bins=bins,
                range=ranges,
            )[0]/(len(p)/ref_len)
            for p in tqdm(periods, desc='Creating normalized heatmaps for each period', disable=not show_progress)
        ]
    else:
        histograms = [
            np.histogramdd(
                p,
                bins=bins,
                range=ranges,
            )[0]
            for p in tqdm(periods, desc='Creating heatmaps for each period', disable=not show_progress)
        ]
    
    return histograms, bins_dicts

def smooth_histograms(histograms):
    return [gaussian_filter(h, sigma=1) for h in tqdm(histograms, desc='Smoothing each heatmap')]


def plot_heatmaps(data):
    
    plots = []
    for i in range(data.shape[0]):
        
        if len(data[i].shape)==3:

            data_2d = np.mean(data[i], axis=0)
            data_2d[data_2d<0]=0
            p = figure(title=f"Plot {i+1}", x_range=(0, data.shape[2]), y_range=(0, data.shape[3]))
            p.image(image=[data_2d], x=0, y=0, dw=data.shape[2], dh=data.shape[3], palette="Spectral11")
        elif len(data[i].shape)==2:
            data[data<0]=0

            p = figure(title=f"Plot {i+1}", x_range=(0, data.shape[1]), y_range=(0, data.shape[2]))
            p.image(image=[data[i]], x=0, y=0, dw=data.shape[1], dh=data.shape[2], palette="Spectral11")
        else:
            assert False, 'Should be either 2 or 3 dimensions'


        plots.append(p)

    # Create a grid layout with 3 plots per row
    grid = gridplot(plots, ncols=3)

    show(grid)


def is_similar(coord1, coord2, max_dist=1):
    """Check if two coordinates are similar based on the 1 unit difference criterion."""
    return all(abs(c1 - c2) <= max_dist for c1, c2 in zip(coord1, coord2))

class Graph:
    def __init__(self, start_coord, start_index, start_sents):
        """
        Initialize a new graph with a starting coordinate, its index, and associated sentences.

        Args:
            start_coord (tuple): The starting coordinate of the graph.
            start_index (int): The index in the original list of the starting coordinate.
            start_sents (list): A list of sentences associated with the starting coordinate.
        """
        self.nodes = [(start_coord, start_index)]
        self.last_update = 0  # Tracks the index of the last list this graph was updated from.
        self.start = start_index
        self.length = 1
        self.sents = start_sents

    def add_coord(self, coord, list_index, sents_to_add=[]):
        """
        Add a new coordinate and its original list index to the graph along with associated sentences.

        Args:
            coord (tuple): The coordinate to add.
            list_index (int): The index of the list from which the coordinate is added.
            sents_to_add (list): The sentences to associate with the coordinate.
        """
        if (coord, list_index) in self.nodes:
            assert (coord, list_index) == self.nodes[-1], ((coord, list_index), self.nodes)
            assert self.last_update == list_index
            return
        self.nodes.append((coord, list_index))
        self.last_update = list_index
        self.length += 1
        self.sents.extend(sents_to_add)
        
    def get_last_coord(self):
        return self.nodes[-1][0]

    def is_similar_and_eligible(self, coord, list_index, n):
        """
        Check if the new coordinate is similar to the last node and eligible to be added.

        Args:
            coord (tuple): The coordinate to check.
            list_index (int): The index of the list from which the coordinate is being considered.
            n (int): The maximum allowed distance in list indices for adding a new coordinate.

        Returns:
            bool: True if the coordinate can be added, False otherwise.
        """
        last_coord, _ = self.nodes[-1]  # Compare only coordinates, ignoring their original indices.
        return (list_index - self.last_update <= n) and is_similar(last_coord, coord)
    
    def summarize(self, sents=None):
        if not sents:
            sents = random.sample(self.sents, min(50, len(self.sents)))
            if type(sents[0]) == tuple:
                sents = [x[0][:1000] for x in sents]
        # print(len(sents))
        text = "\n".join(sents)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-1106",
            messages=[
                {
                    "role": "system",
                    "content": "You are an assistent who is good at summarizing a list of sentences.",
                },
                {
                    "role": "user",
                    "content": f"""
                I will give you a list of sentences and I need you to summarize them. Summarize in 1 or 2 sentences. 
                Sentences: 
                {text}
                Summarization:
                """,
                },
            ],
        )
        self.summarization=response.choices[0].message.content
        return self.summarization
        

    def show(self):
        """Display the graph with coordinates, their original indices, and a sample of associated sentences."""
        for coord, idx in self.nodes:
            print(f"Coord: {coord}, Index in Original List: {idx}")
        return random.sample(self.sents, min(10, len(self.sents)))
    
    

def get_nearby_coords_single(coord, n):
    if n == 0:
        return [coord]
    
    if len(coord) == 2:
        nearby_keys = [
            (coord[0] + dx, coord[1] + dy)
            for dx in range(-n, n+1)
            for dy in range(-n, n+1)
        ]
    elif len(coord) == 3:
        nearby_keys = [
            (coord[0] + dx, coord[1] + dy, coord[2] + dz)
            for dx in range(-n, n+1)
            for dy in range(-n, n+1)
            for dz in range(-n, n+1)
        ]
    else:
        raise Exception(f'Dimension should be either 2 or 3, got dimension {len(coord)}')
    return [
        x for x in nearby_keys 
        if all(0 < component < 400 for component in x)
    ]

def get_nearby_coords_multi(coords, n):
    result_coords = set()
    for coord in coords:
        result_coords.update(get_nearby_coords_single(coord, n))
    return list(result_coords)

def preprocess_blobs(blobs, max_dist):
    coords = [tuple(x[:-1]) for x in blobs]
    sorted_coords = sorted(coords, key = lambda x:x[0])
    
    groups = []
    visited = set()
    
    for i in tqdm(range(len(sorted_coords)), desc='Group blobs for a single period', disable=True):
        if i in visited:
            continue
        group = [sorted_coords[i]]
        max_0 = group[-1][0]
        visited.add(i)
        for j in range(i+1, len(sorted_coords)):
            # if j == i:
            #     continue
            if sorted_coords[j][0]-max_0 > max_dist:
                groups.append(group)
                break
    
            if any(is_similar(sorted_coords[j], c, max_dist) for c in group):
                if j in visited:
                    for g in groups:
                        if sorted_coords[j] in g:
                            group = list(set(group + g))
                            max_0 = max([c[0] for c in group])
                            groups.remove(g)
                else:
                    group.append(sorted_coords[j])
                    visited.add(j)
                    max_0 = max(max_0, sorted_coords[j][0])
            
    if group:
        groups.append(group)
    
    return groups

def get_sentences(bins_dict, grouped_coord, extend_range = 0):
    sents = []
    for c in get_nearby_coords_multi(grouped_coord, extend_range):
        sents.extend([x[0] for x in bins_dict.get(c, [])])
    return sents

def process_lists_with_graph_class(all_blobs, bins_dicts, max_dist=5, show_progress=False, extend_range=1):
    """
    Process lists of coordinates, forming graphs for similar and sequentially close coordinates.

    Args:
        list_of_lists (list of list of tuples): The lists of coordinates to process.
        bins_dicts (list of dicts): A list of dictionaries mapping coordinates to associated sentences.
        n (int): The maximum distance in list indices for adding a new coordinate to an existing graph.

    Returns:
        list of Graph: The list of formed graphs.
    """
    assert len(all_blobs)==len(bins_dicts), f'all_blobs {len(all_blobs)} and bins_dicts {len(bins_dicts)} have difference length!'
    graph_lookup = defaultdict(list)
    for list_index, current_blobs in tqdm(enumerate(all_blobs), total=len(all_blobs), desc='Linking blobs into graphs across periods', disable=not show_progress):
        bins_dict = bins_dicts[list_index]
        
        preprocessed_coords = preprocess_blobs(current_blobs, max_dist)
        
        for grouped_coord in preprocessed_coords: # grouped_coord: list of coords that should be considered as a single coord
            sents_to_add = get_sentences(bins_dict, grouped_coord, extend_range=extend_range)
            
            if list_index == 0:  # Initial list setup
                g = Graph(grouped_coord, list_index, sents_to_add)
                for coord in grouped_coord:
                    assert type(coord)==tuple, coord
                    graph_lookup[coord].append(g)
            
            else:
                added_to_existing_graph = False
                for nearby_coord in get_nearby_coords_multi(grouped_coord, max_dist):
                    
                    curr_graphs = copy.copy(graph_lookup[nearby_coord])
                    for g in curr_graphs:
                        
                        for coord in g.get_last_coord():
                            graph_lookup[coord].remove(g)
                            
                        g.add_coord(grouped_coord, list_index, sents_to_add)
                        
                        for coord in g.get_last_coord():
                            graph_lookup[coord].append(g)
                            
                        added_to_existing_graph = True
            
                if not added_to_existing_graph:
                    g = Graph(grouped_coord, list_index, sents_to_add)
                    for coord in grouped_coord:
                        graph_lookup[coord].append(g)

        for key in list(graph_lookup.keys()): 
            if not graph_lookup[key]:  
                del graph_lookup[key]
                
    resulting_graphs = []
    for graph_lst in graph_lookup.values():
        resulting_graphs.extend([g for g in graph_lst if g.length>1])
    
    return resulting_graphs
