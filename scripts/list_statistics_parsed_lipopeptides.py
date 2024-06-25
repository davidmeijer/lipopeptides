import argparse 
from collections import Counter

parser = argparse.ArgumentParser()
parser.add_argument("--csv", type=str, required=True, help="Path to CSV file containing curated lipopeptides with their lipopeptide tails.")
args = parser.parse_args()

fo = open(args.csv, "r")
fo.readline()  # Skip header.

compounds_with_mibig_accessions_count = 0
mibig_accessions = set()

compounds_with_producing_organism = 0
producing_organisms = set()
producing_genus = Counter()

for line in fo:
    line = line.strip().split(",")
    compound_id, compound_name, producing_organism, mibig_accession, compound_smiles, tail_smiles = line

    if mibig_accession != "":
        compounds_with_mibig_accessions_count += 1
        mibig_accessions.add(mibig_accession)

    if producing_organism != "":
        compounds_with_producing_organism += 1
        producing_organisms.add(producing_organism)
        producing_genus[producing_organism.split(" ")[0]] += 1

fo.close()

print(f"Number of compounds with MIBiG accessions: {compounds_with_mibig_accessions_count}")
print(f"Number of unique MIBiG accessions: {len(mibig_accessions)}")

print(f"Number of compounds with producing organisms: {compounds_with_producing_organism}")
print(f"Number of unique producing organisms: {len(producing_organisms)}")

print(f"Producing genus distribution ({len(producing_genus)} genera):")
# Sort by count.
producing_genus = dict(sorted(producing_genus.items(), key=lambda item: item[1], reverse=True))
for genus, count in producing_genus.items():
    print(f" > {genus}: {count}")
