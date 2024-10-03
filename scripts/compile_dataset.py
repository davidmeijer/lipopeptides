# we only take into account bacterial lipopeptides with name that have either structure or bgc + structure

import argparse 
import rdkit 
from rdkit import Chem

parser = argparse.ArgumentParser()
parser.add_argument("-i", type=str, required=True)
parser.add_argument("-o", type=str, required=True)
args = parser.parse_args()

genii = set()

with open(args.i, "r") as fo:
    fo.readline() # Skip header
    for line in fo:
        line = line.strip()
        _, compound_name, producing_organism, mibig_cluster, canonical_smiles, isomeric_smiles = line.split(",")
        print(producing_organism)
        producing_organism = producing_organism.strip()
        genus = producing_organism.split(" ")[0]
        genii.add(genus)

for i, genus in enumerate(genii):
    print(i, genus)