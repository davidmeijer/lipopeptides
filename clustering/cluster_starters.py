# -*- coding: utf-8 -*-

"""Cluster lipopeptide starter structures."""

import argparse 
import logging
import os
from typing import Dict, Generator
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from sklearn.decomposition import PCA
from scipy.stats import gaussian_kde

# Use Arial font for plots.
plt.rcParams["font.family"] = "Arial"


ALPHA_GREEK = "\u03B1"


def cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", type=str, required=True)
    parser.add_argument("-o", type=str, required=True)
    return parser.parse_args()


def get_records(input_file: str) -> Generator[Dict[str, str], None, None]:
    with open(input_file, "r") as fo:
        header = fo.readline().strip().split(",")

        parsed_record = {}
        
        for line in fo:
            line = line.strip().split(",")  
            record = {k: v for k, v in zip(header, line)}

            # Skip if record is not of bacterial origin.
            if record["origin"] != "bacterial":
                continue
            else:
                parsed_record["origin"] = record["origin"]

            # Skip if record has no associated MIBiG accession.
            if not record["mibig"].startswith("BGC"):
                continue
            else:
                parsed_record["mibig"] = record["mibig"]

            # Generate valid canonical SMILES if it is not present. Skip if this
            # is not possible.
            canonical_smiles = record["canonical_smiles"]
            isomeric_smiles = record["isomeric_smiles"]

            if canonical_smiles != "unknown":
                mol = Chem.MolFromSmiles(canonical_smiles)
                smiles = Chem.MolToSmiles(mol, isomericSmiles=False)
            elif isomeric_smiles != "unknown":
                mol = Chem.MolFromSmiles(isomeric_smiles)
                smiles = Chem.MolToSmiles(mol, isomericSmiles=False)
            else:
                continue

            parsed_record["smiles"] = smiles

            # Validate starter structure if present. Skip if it is not valid.
            if record["starter"] in ["no_tail", "unknown"]:
                continue
            else:
                starter_smiles = record["starter"]
                mol = Chem.MolFromSmiles(starter_smiles)
                parsed_record["starter"] = Chem.MolToSmiles(mol, isomericSmiles=False)

            # Skip if category is AR, FA, LCFA, MCFA, or SCFA.
            if record["category"] not in ["AR", "FA", "LCFA", "MCFA", "SCFA"]:
                continue

            # Parse out other properties from the record.
            parsed_record["name"] = record["trivial_name"]
            parsed_record["genus"] = record["genus"]
            parsed_record["species"] = record["species"]
            parsed_record["category"] = record["category"]

            yield parsed_record



def plot_starter_pca(ax, mols, labels_category, category_to_color):
    """Function to generate the PCA plot for starter structures on a specific axis."""
    logger = logging.getLogger(__name__)

    fingerprint_generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fingerprints = [fingerprint_generator.GetFingerprint(m) for m in mols]

    # Convert fingerprints to numpy array.
    X = np.zeros((len(fingerprints), 2048))
    for i, fp in enumerate(fingerprints):
        DataStructs.ConvertToNumpyArray(fp, X[i])
    logger.info(f"fingerprints shape: {X.shape}")

    # Perform PCA on the fingerprints.
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(X)
    logger.info(f"principal components shape: {pcs.shape}")
    explained_variance = pca.explained_variance_ratio_
    logger.info(f"explained variance: {explained_variance}")

    unique_labels = np.unique(labels_category)
    unique_labels = sorted(unique_labels)
    
    for label in unique_labels:
        # First plot a density plot.
        data = pcs[labels_category == label].T
        if label in ["AR", "LCFA", "MCFA", "SCFA"]:
            kde = gaussian_kde(data)
            xgrid = np.linspace(pcs[:, 0].min() - 0.5, pcs[:, 0].max() + 0.5, 100)
            ygrid = np.linspace(pcs[:, 1].min() - 0.5, pcs[:, 1].max() + 0.5, 100)
            Xgrid, Ygrid = np.meshgrid(xgrid, ygrid)
            Z = kde(np.vstack([Xgrid.ravel(), Ygrid.ravel()])).reshape(Xgrid.shape)
            ax.contour(Xgrid, Ygrid, Z, levels=10, colors=category_to_color[label], alpha=0.5, zorder=1)

    # for label in unique_labels:
    for label in ["AR", "SCFA", "MCFA", "LCFA"]:
        if label not in ["AR", "LCFA", "MCFA", "SCFA"]:
            continue
        jitter = 0.05
        x = pcs[labels_category == label, 0]
        y = pcs[labels_category == label, 1]
        color = category_to_color[label]

        # Add a bit of jitter to x and y values so that they are not overlapping.
        x += np.random.normal(0, jitter, x.shape)
        y += np.random.normal(0, jitter, y.shape)

        ax.scatter(
            x, y, 
            c=color, 
            label=f"{label} ({len(x)})",
            edgecolor="black", 
            linewidth=0.5,
            zorder=2
        )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(f"PC 1 ({explained_variance[0] * 100:.2f}%)", fontsize=16)
    ax.set_ylabel(f"PC 2 ({explained_variance[1] * 100:.2f}%)", fontsize=16)
    ax.legend(title="Category", title_fontsize="16", fontsize="14")


def bfs(graph, graph_identity, start_node):
    # graph identity stores key as atom idx and value as atom symbol

    queue = deque([start_node])
    distances = {node: float('inf') for node in graph}
    distances[start_node] = 0
    shortest_paths = {node: [] for node in graph}
    shortest_paths[start_node] = [start_node]
    
    while queue:
        current_node = queue.popleft()
        
        for neighbor in graph[current_node]:
            if distances[neighbor] == float('inf'):
                distances[neighbor] = distances[current_node] + 1
                shortest_paths[neighbor] = shortest_paths[current_node] + [neighbor]
                queue.append(neighbor)
                
    return distances, shortest_paths

def bfs_carbon_paths(graph, graph_identity, start_node):
    # Check if the start node is carbon
    if graph_identity[start_node] != 'C':
        return {}, {}  # No paths if the start node is not carbon

    queue = deque([start_node])
    distances = {node: float('inf') for node in graph}
    distances[start_node] = 0
    shortest_paths = {node: [] for node in graph}
    shortest_paths[start_node] = [start_node]
    
    while queue:
        current_node = queue.popleft()
        
        for neighbor in graph[current_node]:
            # Only consider carbon atoms for the path
            if graph_identity[neighbor] == 'C' and distances[neighbor] == float('inf'):
                distances[neighbor] = distances[current_node] + 1
                shortest_paths[neighbor] = shortest_paths[current_node] + [neighbor]
                queue.append(neighbor)
    
    # Filter out non-carbon atoms from the results
    carbon_distances = {node: dist for node, dist in distances.items() if graph_identity[node] == 'C'}
    carbon_paths = {node: path for node, path in shortest_paths.items() if graph_identity[node] == 'C'}
    
    return carbon_distances, carbon_paths


def find_longest_shortest_path_unweighted(graph, graph_identity, start_node):
    distances, shortest_paths = bfs_carbon_paths(graph, graph_identity, start_node)
    
    # Find the longest shortest path
    longest_shortest_path = max(shortest_paths.values(), key=len)
    return longest_shortest_path


def retrieve_backbone(mol):
    graph_identity = {}  # atom idx -> atom symbol
    graph = {}
    for atom in mol.GetAtoms():
        atom_id = atom.GetIdx()
        atom_symbol = atom.GetSymbol()
        graph_identity[atom_id] = atom_symbol
        atom_neighbors = [neighbor.GetIdx() for neighbor in atom.GetNeighbors()]
        graph[atom_id] = atom_neighbors

    # Find atom indices of COOH group.
    pattern = Chem.MolFromSmarts("C(=O)[OH]")
    matches = mol.GetSubstructMatches(pattern)
    if len(matches) == 0:
        raise ValueError("COOH group not found.")
    alpha_carbon_index = matches[0][0]
    
    # Find the longest shortest path from the alpha carbon.
    longest_shortest_path = find_longest_shortest_path_unweighted(graph, graph_identity, alpha_carbon_index)
    return longest_shortest_path


def index_lipid_tail_starters(ax, mols, backbones, category, category_color, max_height, max_backbone_len, set_xaxis_label=False):
    labels = {
        "Carbonic acid": "#e69f00",
        "Hydroxyl": "#56b4e9",
        "Methyl": "#0072b2",
        "Epoxide": "#d55f00",
        "Ketone": "#039e73",
        "Amine": "#cc79a7",
        "Imine": "#f0e442",
    }
    counts = [[0 for _ in range(len(labels) + 1)] for _ in range(max_backbone_len)]
    for mol, backbone in zip(mols, backbones):
        for i, atom in enumerate(backbone):

            pattern = "C(=O)[OH]"
            matches = mol.GetSubstructMatches(Chem.MolFromSmarts(pattern))
            if atom == matches[0][0]:
                counts[i][0] += 1

            pattern = "[C][CH1]([OH])[C]"
            matches = mol.GetSubstructMatches(Chem.MolFromSmarts(pattern))
            for match in matches:
                if atom == match[1]:
                    counts[i][1] += 1

            pattern = "[C][CH1]([CH3])[C]"
            matches = mol.GetSubstructMatches(Chem.MolFromSmarts(pattern))
            for match in matches:
                if atom == match[1]:
                    counts[i][2] += 1

            pattern = "C1CO1"
            matches = mol.GetSubstructMatches(Chem.MolFromSmarts(pattern))
            for match in matches:
                if atom == match[0]:
                    counts[i][3] += 1

            pattern = "[C][C](=O)[C]"
            matches = mol.GetSubstructMatches(Chem.MolFromSmarts(pattern))
            for match in matches:
                if atom == match[1]:
                    counts[i][4] += 1

            pattern = "[C][NH2]"
            matches = mol.GetSubstructMatches(Chem.MolFromSmarts(pattern))
            for match in matches:
                if atom == match[0]:
                    counts[i][5] += 1

            pattern = "[C][NH][C]=[N]"
            matches = mol.GetSubstructMatches(Chem.MolFromSmarts(pattern))
            for match in matches:
                if atom == match[0]:
                    counts[i][6] += 1
    
            counts[i][7] += 1

    # plot line plot for lengths
    ax.plot(range(1, max_backbone_len+1), [x[7] for x in counts], color="black", marker="o", label="Length", markersize=3)
    ax.grid(axis="y", linestyle="--", alpha=0.3, zorder=0)
    ax.grid(axis="x", linestyle="--", alpha=0.3, zorder=0)

    if set_xaxis_label:
        ax.set_xticks(
            [i for i in range(1, max_backbone_len+1)], 
            ["1"] + [str(i) if (i) % 5 == 0 else "" for i in range(1, max_backbone_len)],
            fontsize=11
        )
    else:
        ax.set_xticks(
            [i for i in range(1, max_backbone_len+1)], 
        )
        ax.set_xticklabels(
            ["", ""] + ["" if (i) % 5 == 0 else "" for i in range(2, max_backbone_len)],
            fontsize=11
        )
    
    y_tick_distance = 1
    if max_height > 10 and max_height <= 100:
        y_tick_distance = 10
    if max_height > 100 and max_height <= 1000:
        y_tick_distance = 20
    ax.set_yticks(
        [i for i in range(0, int(1.2 * max_height), y_tick_distance)],
        [str(i) for i in range(0, int(1.2 * max_height), y_tick_distance)],
        fontsize=11
    )
    # stacked bar plot for other labels
    bottom = [0 for _ in range(max_backbone_len)]
    for i, label in enumerate(labels.keys()):
        if any([x[i] > 0 for x in counts]):
            total = sum([x[i] for x in counts])
            ax.bar(
                range(1, max_backbone_len+1), 
                [x[i] for x in counts], 
                label=f"{label} ({total})", 
                bottom=bottom, 
                color=labels[label], 
                edgecolor="black", 
                zorder=100
            )
            bottom = [sum(x) for x in zip(bottom, [x[i] for x in counts])]
    ax.set_ylim(0, 1.1 * max_height)
    if set_xaxis_label:
        ax.set_xlabel(f"Backbone position (starting at {ALPHA_GREEK}-carbon)", fontsize=16)

    # ax.text(
    #     0.05, 0.92, 
    #     f"{category}",
    #     transform=ax.transAxes,
    #     fontsize=12, verticalalignment='top',
    #     bbox=dict(
    #         facecolor=category_color,
    #         alpha=0.8,
    #         edgecolor='black',
    #         boxstyle='square,pad=0.5')
    #         # boxstyle='round,pad=0.5')
    #     )

    ax.legend(
        markerfirst=False,
        title_fontsize="16", 
        fontsize="14", 
        title=f"{category} properties"
    )


def main() -> None:
    args = cli()
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info(f"command line arguments: {args}")

    starters = []
    labels_category = []
    labels_genus = []
    labels_species = []

    for record in get_records(args.i):
        starters.append(record["starter"])
        labels_category.append(record["category"])
        labels_genus.append(record["genus"])
        labels_species.append(record["species"])
    logger.info(f"{len(starters)} starter structures found")

    starter_mols = [Chem.MolFromSmiles(s) for s in starters]
    labels_category = np.array(labels_category)
    labels_genus = np.array(labels_genus)
    labels_species = np.array(labels_species)

    palette = [
        "#e69f00",
        "#56b4e9",
        "#039e73",
        "#f0e442",
        "#0072b2",
        "#d55f00",
        "#cc79a7",
        "#000000",
        "#808285",
    ]
    category_to_color = {
        "AR": palette[0],
        "FA": palette[1],
        "LCFA": palette[2],
        "MCFA": palette[3],
        "SCFA": palette[4],
    }

    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(4, 3)

    # # Dummy plot with just label for AR.
    # ax6 = fig.add_subplot(gs[2, 0])
    # # remove axis
    # ax6.axis("off")

    # Plot PCA for all starter structures.
    ax1 = fig.add_subplot(gs[1:3, 0:2])
    plot_starter_pca(ax1, starter_mols, labels_category, category_to_color)

    # Calculate max backbone lengths.
    backbones = []
    backbone_mols = []
    for mol, label in zip(starter_mols, labels_category):
        if label in ["FA", "LCFA", "MCFA", "SCFA"]:
            backbone_inds = retrieve_backbone(mol)
            backbones.append(backbone_inds)
            backbone_mols.append(mol)
        else:
            backbones.append([])
    backbone_lens = [len(backbone) for backbone in backbones]
    max_backbone_len = max(backbone_lens)

    # Get max category size.
    max_category_size = max([sum(labels_category == category) for category in category_to_color])

    # Index FA starters.
    # ax2 = fig.add_subplot(gs[2, 0])
    # index_lipid_tail_starters(
    #     ax2,
    #     [m for m, l in zip(starter_mols, labels_category) if l == "FA"],
    #     [b for b, l in zip(backbones, labels_category) if l == "FA"],
    #     "FA",
    #     category_to_color["FA"],
    #     # max_category_size,
    #     len([c for c in labels_category if c == "FA"]),
    #     max_backbone_len,
    #     set_xaxis_label=True
    # )

    # make row space between ax3, ax4, ax5 lower, so only column 1
    sub_gs1 = gs[:, 2].subgridspec(4, 1, hspace=0.01)

    # Index LCFA starters.
    # ax3 = fig.add_subplot(gs[2, 1])
    ax3 = fig.add_subplot(sub_gs1[3, 0])
    index_lipid_tail_starters(
        ax3,
        [m for m, l in zip(starter_mols, labels_category) if l == "LCFA"],
        [b for b, l in zip(backbones, labels_category) if l == "LCFA"],
        "LCFA",
        category_to_color["LCFA"],
        # max_category_size,
        len([c for c in labels_category if c == "LCFA"]),
        max_backbone_len,
        set_xaxis_label=True
    )

    # Index MCFA starters.
    # ax4 = fig.add_subplot(gs[1, 1])
    ax4 = fig.add_subplot(sub_gs1[2, 0])
    index_lipid_tail_starters(
        ax4,
        [m for m, l in zip(starter_mols, labels_category) if l == "MCFA"],
        [b for b, l in zip(backbones, labels_category) if l == "MCFA"],
        "MCFA",
        category_to_color["MCFA"],
        # max_category_size,
        len([c for c in labels_category if c == "MCFA"]),
        max_backbone_len,
        set_xaxis_label=False
    )

    # Index SCFA starters.
    # ax5 = fig.add_subplot(gs[0, 1])
    ax5 = fig.add_subplot(sub_gs1[1, 0])
    index_lipid_tail_starters(
        ax5,
        [m for m, l in zip(starter_mols, labels_category) if l == "SCFA"],
        [b for b, l in zip(backbones, labels_category) if l == "SCFA"],
        "SCFA",
        category_to_color["SCFA"],
        # max_category_size,
        len([c for c in labels_category if c == "SCFA"]),
        max_backbone_len,
        set_xaxis_label=False
    )

    ########################################################################################################################
    ########################################################################################################################
    ########################################################################################################################
    ########################################################################################################################

    # Make bar plot of backbone lengths.
    ax7 = fig.add_subplot(gs[0, 1])
    ax7.hist(backbone_lens, bins=range(1, max_backbone_len+2), edgecolor="black", facecolor="#ceccca", zorder=100)
    max_bin_size = max([backbone_lens.count(i) for i in range(1, max_backbone_len+1)])
    ax7.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
    ax7.set_xticks(
        [i + 0.5 for i in range(1, max_backbone_len+1)], 
        ["1"] + [str(i) if i % 5 == 0 else "" for i in range(2, max_backbone_len+1)],
        fontsize=11
    )
    ax7.set_yticks(
        [i for i in range(0, int(1.1 * max_bin_size), 5)],
        [str(i) for i in range(0, int(1.1 * max_bin_size), 5)],
        fontsize=11
    )
    ax7.set_xlabel(f"Backbone length (including {ALPHA_GREEK}-carbon)", fontsize=16)


    ########################################################################################################################
    ########################################################################################################################
    ########################################################################################################################
    ########################################################################################################################

    # Get bond order plot of backbone.
    counted = 0
    bond_orders = [[] for _ in range(max_backbone_len)]
    for mol, categry in zip(backbone_mols, labels_category):
        counted += 1
        backbone = retrieve_backbone(mol)
        for i, atom in enumerate(backbone[:-1]):
            next_atom = backbone[i+1]
            bond = mol.GetBondBetweenAtoms(atom, next_atom)
            bond_order = bond.GetBondTypeAsDouble()
            # if bond_order > 1.0:
            bond_orders[i].append(bond_order)

    bond_orders_1_heights = []
    bond_orders_2_heights = []
    bond_orders_3_heights = []
    for bin in bond_orders:
        count_1 = bin.count(1.0)
        count_2 = bin.count(2.0)
        count_3 = bin.count(3.0)
        bond_orders_1_heights.append(count_1)
        bond_orders_2_heights.append(count_2)
        bond_orders_3_heights.append(count_3)

    ax8 = fig.add_subplot(gs[0, 0])
    ax8.bar(range(1, max_backbone_len + 1), bond_orders_2_heights, color="#23aae1", edgecolor="black", label=f"Double bonds ({sum(bond_orders_2_heights)})", bottom=[0 for _ in range(max_backbone_len)], facecolor=palette[-2])
    # ax8.bar(range(1, max_backbone_len + 1), bond_orders_3_heights, color="#f9a11b", edgecolor="black", label=f"Triple bonds ({sum(bond_orders_3_heights)})", bottom=bond_orders_2_heights, facecolor=palette[-1])
    ax8.bar(range(1, max_backbone_len + 1), bond_orders_1_heights, color="#f9e11b", edgecolor="black", label=f"Single bonds ({sum(bond_orders_1_heights)})", bottom=[sum(x) for x in zip(bond_orders_2_heights, bond_orders_3_heights)], facecolor="#ceccca")
    ax8.set_ylim(0, 1.1 * max([sum(x) for x in zip(bond_orders_1_heights, bond_orders_2_heights, bond_orders_3_heights)]))
    ax8.set_xticks(
        [0.5] + [i + 0.5 for i in range(1, max_backbone_len+1)], 
        ["1"] + [str(i + 1) if (i + 1) % 5 == 0 else "" for i in range(0, max_backbone_len)],
        fontsize=11
    )
    ax8.set_yticks(
        [i for i in range(0, int(1.1 * max([sum(x) for x in zip(bond_orders_1_heights, bond_orders_2_heights, bond_orders_3_heights)])), 20)],
        [str(i) for i in range(0, int(1.1 * max([sum(x) for x in zip(bond_orders_1_heights, bond_orders_2_heights, bond_orders_3_heights)])), 20)],
        fontsize=11
    )
    # ax8.legend(fontsize=11, title_fontsize=12, title="Saturation")

    # Change order of labels in legend: single > double
    handles, labels = ax8.get_legend_handles_labels()
    order = [1, 0]
    ax8.legend([handles[idx] for idx in order], [labels[idx] for idx in order], fontsize=14, title_fontsize=16, title="Saturation")

    ax8.grid(axis="y", linestyle="--", alpha=0.5)
    
    ax8.set_xlabel(f"Backbone position (starting at {ALPHA_GREEK}-carbon)", fontsize=16)

    # Save the entire figure grid to a file.
    path_out = os.path.join(args.o, "cheminformatics_analysis.png")
    plt.subplots_adjust(hspace=0.8, wspace=0.15)
    # plt.tight_layout()
    # Remove space around plot.
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    plt.savefig(path_out, dpi=600, transparent=True)

    ######

    # Get all structures for AR, calc inchikey and count which structures more prevalent
    ar_mols = [m for m, l in zip(starter_mols, labels_category) if l == "AR"]
    ar_inchikeys = [Chem.MolToInchiKey(m).split("-")[0] for m in ar_mols]
    ar_inchikey_counts = {k: ar_inchikeys.count(k) for k in set(ar_inchikeys)}
    ar_inchikey_counts = dict(sorted(ar_inchikey_counts.items(), key=lambda x: x[1], reverse=True))
    logger.info(f"AR inchikey counts: {ar_inchikey_counts}")
    # get top 5 inchikeys and their counts, and one asosciated mol and turn into smiles
    top_5_inchikeys = list(ar_inchikey_counts.keys())[:5]
    top_5_counts = [ar_inchikey_counts[k] for k in top_5_inchikeys]
    top_5_mols = [ar_mols[ar_inchikeys.index(k)] for k in top_5_inchikeys]
    top_5_smiles = [Chem.MolToSmiles(m) for m in top_5_mols]
    for k, c, s in zip(top_5_inchikeys, top_5_counts, top_5_smiles):
        logger.info(f"AR inchikey: {k}, count: {c}, smiles: {s}")

    logger.info("goodbye!")
    

if __name__ == "__main__":
    main()
