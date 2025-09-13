"""
Visualization utilities for NEAT Trader package.

This module provides functions for visualizing NEAT neural networks,
evolution statistics, and trading performance.
"""

import warnings
import os
from typing import Optional, Dict, Any
import graphviz
import matplotlib.pyplot as plt
import numpy as np

from ..config import GRAPHVIZ_PATH, DEFAULT_VIZ_FORMAT, DEFAULT_VIZ_VIEW
from ..exceptions import VisualizationError


def plot_stats(statistics: Any, ylog: bool = False, view: bool = DEFAULT_VIZ_VIEW, 
               filename: str = 'avg_fitness.svg') -> None:
    """
    Plot the population's average and best fitness over generations.
    
    Args:
        statistics: NEAT statistics object
        ylog: Whether to use logarithmic scale for y-axis
        view: Whether to display the plot
        filename: Output filename for the plot
        
    Raises:
        VisualizationError: If plotting fails
    """
    if plt is None:
        warnings.warn("This display is not available due to a missing optional dependency (matplotlib)")
        return

    try:
        generation = range(len(statistics.most_fit_genomes))
        best_fitness = [c.fitness for c in statistics.most_fit_genomes]
        avg_fitness = np.array(statistics.get_fitness_mean())
        stdev_fitness = np.array(statistics.get_fitness_stdev())

        plt.figure(figsize=(10, 6))
        plt.plot(generation, avg_fitness, 'b-', label="average")
        plt.plot(generation, avg_fitness - stdev_fitness, 'g-.', label="-1 sd")
        plt.plot(generation, avg_fitness + stdev_fitness, 'g-.', label="+1 sd")
        plt.plot(generation, best_fitness, 'r-', label="best")

        plt.title("Population's average and best fitness")
        plt.xlabel("Generations")
        plt.ylabel("Fitness")
        plt.grid()
        plt.legend(loc="best")
        if ylog:
            plt.gca().set_yscale('symlog')

        plt.savefig(filename, dpi=300, bbox_inches='tight')
        if view:
            plt.show()

        plt.close()
        
    except Exception as e:
        raise VisualizationError(f"Failed to plot statistics: {str(e)}")


def plot_spikes(spikes: list, view: bool = DEFAULT_VIZ_VIEW, filename: Optional[str] = None, 
                title: Optional[str] = None) -> Optional[plt.Figure]:
    """
    Plot the trains for a single spiking neuron.
    
    Args:
        spikes: List of spike data tuples (t, I, v, u, f)
        view: Whether to display the plot
        filename: Output filename for the plot
        title: Optional title for the plot
        
    Returns:
        Figure object or None if view=True
        
    Raises:
        VisualizationError: If plotting fails
    """
    try:
        t_values = [t for t, I, v, u, f in spikes]
        v_values = [v for t, I, v, u, f in spikes]
        u_values = [u for t, I, v, u, f in spikes]
        I_values = [I for t, I, v, u, f in spikes]
        f_values = [f for t, I, v, u, f in spikes]

        fig = plt.figure(figsize=(12, 10))
        
        plt.subplot(4, 1, 1)
        plt.ylabel("Potential (mv)")
        plt.xlabel("Time (in ms)")
        plt.grid()
        plt.plot(t_values, v_values, "g-")

        if title is None:
            plt.title("Izhikevich's spiking neuron model")
        else:
            plt.title(f"Izhikevich's spiking neuron model ({title})")

        plt.subplot(4, 1, 2)
        plt.ylabel("Fired")
        plt.xlabel("Time (in ms)")
        plt.grid()
        plt.plot(t_values, f_values, "r-")

        plt.subplot(4, 1, 3)
        plt.ylabel("Recovery (u)")
        plt.xlabel("Time (in ms)")
        plt.grid()
        plt.plot(t_values, u_values, "r-")

        plt.subplot(4, 1, 4)
        plt.ylabel("Current (I)")
        plt.xlabel("Time (in ms)")
        plt.grid()
        plt.plot(t_values, I_values, "r-o")

        if filename is not None:
            plt.savefig(filename, dpi=300, bbox_inches='tight')

        if view:
            plt.show()
            plt.close()
            return None

        return fig
        
    except Exception as e:
        raise VisualizationError(f"Failed to plot spikes: {str(e)}")


def plot_species(statistics: Any, view: bool = DEFAULT_VIZ_VIEW, 
                 filename: str = 'speciation.svg') -> None:
    """
    Visualize speciation throughout evolution.
    
    Args:
        statistics: NEAT statistics object
        view: Whether to display the plot
        filename: Output filename for the plot
        
    Raises:
        VisualizationError: If plotting fails
    """
    if plt is None:
        warnings.warn("This display is not available due to a missing optional dependency (matplotlib)")
        return

    try:
        species_sizes = statistics.get_species_sizes()
        num_generations = len(species_sizes)
        curves = np.array(species_sizes).T

        fig, ax = plt.subplots(figsize=(12, 8))
        ax.stackplot(range(num_generations), *curves)

        plt.title("Speciation")
        plt.ylabel("Size per Species")
        plt.xlabel("Generations")

        plt.savefig(filename, dpi=300, bbox_inches='tight')

        if view:
            plt.show()

        plt.close()
        
    except Exception as e:
        raise VisualizationError(f"Failed to plot species: {str(e)}")


def draw_net(config: Any, genome: Any, view: bool = DEFAULT_VIZ_VIEW, filename: Optional[str] = None, 
             node_names: Optional[Dict[int, str]] = None, show_disabled: bool = True, 
             prune_unused: bool = False, node_colors: Optional[Dict[int, str]] = None, 
             fmt: str = DEFAULT_VIZ_FORMAT) -> Optional[graphviz.Digraph]:
    """
    Draw a neural network with arbitrary topology from a NEAT genome.
    
    Args:
        config: NEAT configuration object
        genome: NEAT genome to visualize
        view: Whether to display the network
        filename: Output filename for the network diagram
        node_names: Dictionary mapping node IDs to names
        show_disabled: Whether to show disabled connections
        prune_unused: Whether to prune unused nodes
        node_colors: Dictionary mapping node IDs to colors
        fmt: Output format (svg, png, etc.)
        
    Returns:
        Graphviz Digraph object or None if view=True
        
    Raises:
        VisualizationError: If network drawing fails
    """
    # Add Graphviz to PATH if needed
    if GRAPHVIZ_PATH not in os.environ.get("PATH", ""):
        os.environ["PATH"] += os.pathsep + GRAPHVIZ_PATH
        
    if graphviz is None:
        warnings.warn("This display is not available due to a missing optional dependency (graphviz)")
        return None

    try:
        # If requested, use a copy of the genome which omits all components that won't affect the output.
        if prune_unused:
            genome = genome.get_pruned_copy(config.genome_config)

        if node_names is None:
            node_names = {}

        if node_colors is None:
            node_colors = {}

        node_attrs = {
            'shape': 'circle',
            'fontsize': '9',
            'height': '0.2',
            'width': '0.2'
        }

        dot = graphviz.Digraph(format=fmt, node_attr=node_attrs)

        # Add input nodes
        inputs = set()
        for k in config.genome_config.input_keys:
            inputs.add(k)
            name = node_names.get(k, str(k))
            input_attrs = {'style': 'filled', 'shape': 'box', 'fillcolor': node_colors.get(k, 'lightgray')}
            dot.node(name, _attributes=input_attrs)

        # Add output nodes
        outputs = set()
        for k in config.genome_config.output_keys:
            outputs.add(k)
            name = node_names.get(k, str(k))
            node_attrs = {'style': 'filled', 'fillcolor': node_colors.get(k, 'lightblue')}
            dot.node(name, _attributes=node_attrs)

        # Add hidden nodes
        used_nodes = set(genome.nodes.keys())
        for n in used_nodes:
            if n in inputs or n in outputs:
                continue

            attrs = {'style': 'filled', 'fillcolor': node_colors.get(n, 'white')}
            dot.node(str(n), _attributes=attrs)

        # Add connections
        for cg in genome.connections.values():
            if cg.enabled or show_disabled:
                input_node, output_node = cg.key
                a = node_names.get(input_node, str(input_node))
                b = node_names.get(output_node, str(output_node))
                style = 'solid' if cg.enabled else 'dotted'
                color = 'green' if cg.weight > 0 else 'red'
                width = str(0.1 + abs(cg.weight / 5.0))
                dot.edge(a, b, _attributes={'style': style, 'color': color, 'penwidth': width})

        if filename:
            dot.render(filename, view=view)

        return dot
        
    except Exception as e:
        raise VisualizationError(f"Failed to draw network: {str(e)}")