import argparse 

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
        compound_id,compound_name,producing_organism,mibig_cluster,canonical_smiles,tail_smiles = line
        mol = Chem.MolFromSmiles(tail_smiles)
        mols.append(mol)
    fo.close()
    return mols


def main():
    args = cli()
    mols = parse_mols(args.csv)


if __name__ == "__main__":
    main()
    