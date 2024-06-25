import argparse 
import os
import typing as ty

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from scipy.cluster.hierarchy import linkage, dendrogram, to_tree
from scipy.spatial.distance import squareform



def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="Path to CSV file containing curated lipopeptides with their lipopeptide tails.")
    parser.add_argument("--out", type=str, required=True, help="Path to output directory.")
    return parser.parse_args()


def parse_mols(path):
    fo = open(path, "r")
    fo.readline()
    names = []
    mols = []
    for line in fo:
        line = line.strip().split(",")
        compound_id,compound_name,producing_organism,mibig_cluster,canonical_smiles,tail_smiles = line
        mol = Chem.MolFromSmiles(tail_smiles)
        mols.append(mol)
        names.append(compound_name)
    fo.close()
    return mols,names


def mol_to_fingerprint(mol: Chem.Mol, radius: int, num_bits: int) -> np.array:
    """Convert a molecule to a fingerprint.
    
    :param mol: The molecule.
    :type mol: Chem.Mol
    :param radius: The radius of the fingerprint.
    :type radius: int
    :param num_bits: The number of bits.
    :type num_bits: int
    :return: The fingerprint.
    :rtype: np.array
    """
    fp_arr = np.zeros((0,), dtype=np.int8)
    fp_vec = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=num_bits)
    DataStructs.ConvertToNumpyArray(fp_vec, fp_arr)
    return fp_arr

def tanimoto_similarity(fp1: np.array, fp2: np.array) -> float:
    """Calculate the Tanimoto similarity between two fingerprints.
    
    :param fp1: The first fingerprint.
    :type fp1: np.array
    :param fp2: The second fingerprint.
    :type fp2: np.array
    :return: The Tanimoto similarity.
    :rtype: float
    """
    return np.logical_and(fp1, fp2).sum() / np.logical_or(fp1, fp2).sum()


def main():
    args = cli()
    mols, labels = parse_mols(args.csv)
    fps = [mol_to_fingerprint(mol, 2, 2048) for mol in mols]

    # Calculate the structural similarity matrix.
    sim_matrix = np.zeros((len(mols), len(mols)))
    for i in tqdm(range(len(mols))):
        for j in range(len(mols)):
            if i == j:
                sim_matrix[i, j] = 1.0
            elif i < j:
                sim_matrix[i, j] = tanimoto_similarity(fps[i], fps[j])
                sim_matrix[j, i] = sim_matrix[i, j]

    # Perform hierarchical clustering and plot as circular dendrogram.
    dist_matrix = 1 - sim_matrix
    dist_array = squareform(dist_matrix)
    linkage_matrix = linkage(dist_array, method="average")

    def to_newick(linkage_matrix: np.ndarray, labels: ty.List[str]) -> str:
        """
        Outputs linkage matrix as Newick file.

        Parameters
        ----------
        linkage_matrix : np.ndarray
            condensed distance matrix
        labels : list of str, optional
            leaf labels

        Returns
        -------
        newick : str
            linkage matrix in newick format tree

        Source:
        https://stackoverflow.com/questions/28222179/save-dendrogram-to-newick-format
        """
        tree = to_tree(linkage_matrix, rd=False)

        def get_newick(node, newick, parentdist, leaf_names):
            if node.is_leaf():
                return "%s:%.2f%s" % (
                    leaf_names[node.id],
                    parentdist - node.dist,
                    newick
                )
            else:
                if len(newick) > 0:
                    newick = "):%.2f%s" % (parentdist - node.dist, newick)
                else:
                    newick = ");"
                newick = get_newick(
                    node.get_left(),
                    newick,
                    node.dist,
                    leaf_names
                )
                newick = get_newick(
                    node.get_right(),
                    ",%s" % (newick),
                    node.dist,
                    leaf_names
                )
                newick = "(%s" % (newick)
                return newick
        newick = get_newick(tree, "", tree.dist, labels)
        return newick

    newick_str = to_newick(linkage_matrix, labels)

    # Save Newick string to a file
    with open(os.path.join(args.out, "structural_similarity_tree.nwk"), "w") as f:
        f.write(newick_str)




if __name__ == "__main__":
    main()
    