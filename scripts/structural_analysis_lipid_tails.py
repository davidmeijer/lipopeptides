import argparse 
from collections import deque

from rdkit import Chem
import matplotlib.pyplot as plt


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="Path to CSV file containing curated lipopeptides with their lipopeptide tails.")
    parser.add_argument("--out", type=str, required=True, help="Path to output directory.")
    return parser.parse_args()


def parse_mols(path):
    fo = open(path, "r")
    fo.readline()
    mols = []
    for line in fo:
        line = line.strip().split(",")
        _, _, _, _, _, tail_smiles = line
        mol = Chem.MolFromSmiles(tail_smiles)
        mols.append(mol)
    fo.close()
    return mols


def bfs(graph, start_node):
    # TODO: only consider paths that consist solely of carbon atoms
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

def find_longest_shortest_path_unweighted(graph, start_node):
    distances, shortest_paths = bfs(graph, start_node)
    
    # Find the longest shortest path
    longest_shortest_path = max(shortest_paths.values(), key=len)
    return longest_shortest_path


def retrieve_backbone(mol):
    graph = {}
    for atom in mol.GetAtoms():
        atom_id = atom.GetIdx()
        atom_neighbors = [neighbor.GetIdx() for neighbor in atom.GetNeighbors()]
        graph[atom_id] = atom_neighbors

    # Find atom indices of COOH group.
    pattern = Chem.MolFromSmarts("C(=O)[OH]")
    matches = mol.GetSubstructMatches(pattern)
    if len(matches) == 0:
        raise ValueError("COOH group not found.")
    alpha_carbon_index = matches[0][0]
    
    # Find the longest shortest path from the alpha carbon.
    longest_shortest_path = find_longest_shortest_path_unweighted(graph, alpha_carbon_index)
    return longest_shortest_path


def main():
    args = cli()
    mols = parse_mols(args.csv)

    backbones = []
    for mol in mols:
        backbone_inds = retrieve_backbone(mol)
        backbones.append(backbone_inds)

    backbone_lens = [len(backbone) for backbone in backbones]
    max_backbone_len = max(backbone_lens)

    # Make bar plot of backbone lengths.
    plt.hist(backbone_lens, bins=range(1, max_backbone_len+2), edgecolor="black", facecolor="#23aae1")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.xticks(
        [i + 0.5 for i in range(1, max_backbone_len+1)], 
        ["1"] + [str(i) if i % 5 == 0 else "" for i in range(2, max_backbone_len+1)]
    )
    plt.savefig(args.out + "/tail_lengths.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Get bond order plot of backbone.
    bond_orders = [[] for _ in range(max_backbone_len)]
    for mol, backbone in zip(mols, backbones):
        for i, atom in enumerate(backbone[:-1]):
            next_atom = backbone[i+1]
            bond = mol.GetBondBetweenAtoms(atom, next_atom)
            bond_order = bond.GetBondTypeAsDouble()
            if bond_order > 1.0:
                bond_orders[i].append(bond_order)

    bond_orders_2_heights = []
    bond_orders_3_heights = []
    for bin in bond_orders:
        count_2 = bin.count(2.0)
        count_3 = bin.count(3.0)
        bond_orders_2_heights.append(count_2)
        bond_orders_3_heights.append(count_3)

    plt.bar(range(1, max_backbone_len + 1), bond_orders_2_heights, color="#23aae1", edgecolor="black", label="Double bonds")
    plt.bar(range(1, max_backbone_len + 1), bond_orders_3_heights, color="#f9a11b", edgecolor="black", label="Triple bonds", bottom=bond_orders_2_heights)
    plt.ylim(0, 1.1 * max([sum(x) for x in zip(bond_orders_2_heights, bond_orders_3_heights)]))
    plt.xticks(
        [0.5] + [i + 0.5 for i in range(1, max_backbone_len+1)], 
        ["0"] + [str(i) if i % 5 == 0 else "" for i in range(1, max_backbone_len + 1)]
    )
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.savefig(args.out + "/bond_orders.png", dpi=300, bbox_inches="tight")
    plt.close()

    # labels
    labels = [
        "Carbonic acid",
        "Hydroxyl",
        "Amine",
        "Amine and imine",
        "Methyl",
        "Epoxide",
        "Ketone",
        "Length"
    ]

    label_colors = [
        "#23aae1",
        "#ec008c",
        "#00a651",
        "#f7941d",
        "#f5cb0e",
        "#ed2024",
        "#6d6e71",
        "#3954a5"
    ]


    counts = [[0 for _ in labels] for _ in range(max_backbone_len)]
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

            pattern = "[C][NH2]"
            matches = mol.GetSubstructMatches(Chem.MolFromSmarts(pattern))
            for match in matches:
                if atom == match[0]:
                    counts[i][2] += 1

            pattern = "[C][NH][C]=[N]"
            matches = mol.GetSubstructMatches(Chem.MolFromSmarts(pattern))
            for match in matches:
                if atom == match[0]:
                    counts[i][3] += 1

            pattern = "[C][CH1]([CH3])[C]"
            matches = mol.GetSubstructMatches(Chem.MolFromSmarts(pattern))
            for match in matches:
                if atom == match[1]:
                    counts[i][4] += 1

            pattern = "C1CO1"
            matches = mol.GetSubstructMatches(Chem.MolFromSmarts(pattern))
            for match in matches:
                if atom == match[0]:
                    counts[i][5] += 1

            pattern = "[C][C](=O)[C]"
            matches = mol.GetSubstructMatches(Chem.MolFromSmarts(pattern))
            for match in matches:
                if atom == match[1]:
                    counts[i][6] += 1

            counts[i][7] += 1

    # plot line plot for lengths
    plt.plot(range(1, max_backbone_len+1), [x[7] for x in counts], color="black", marker="o", label="Length")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.xticks(
        [i for i in range(1, max_backbone_len+1)], 
        ["1"] + [str(i) if i % 5 == 0 else "" for i in range(2, max_backbone_len+1)]
    )
    # stacked bar plot for other labels
    bottom = [0 for _ in range(max_backbone_len)]
    for i, label in enumerate(labels[:-1]):
        plt.bar(range(1, max_backbone_len+1), [x[i] for x in counts], label=label, bottom=bottom, color=label_colors[i], edgecolor="black")
        bottom = [sum(x) for x in zip(bottom, [x[i] for x in counts])]

    plt.ylim(0, 530)
    plt.legend()
    plt.savefig(args.out + "/backbone_lengths.png", dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
