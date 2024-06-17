# -*- coding: utf-8 -*-

"""Parse XLSX lipopeptides database file into TSV format."""

import argparse 

import pandas as pd


def cli() -> argparse.Namespace:
    """Parse command line arguments.
    
    :return: Command line arguments.
    :rtype: argparse.Namespace
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-xlsx", required=True, type=str)
    parser.add_argument("--out-tsv", required=True, type=str)
    return parser.parse_args()


def main() -> None:
    """Main function."""
    args = cli()

    in_xlsx = args.in_xlsx
    out_tsv = args.out_tsv

    df = pd.read_excel(in_xlsx)

    # Select columns of interest.
    df = df[['ID', 'Organism', 'MIBiG_cluster', 'Canonical smiles', 'Isomeric smiles']]

    # Convert Unknown values to None.
    df = df.replace('Unknown', None)

    # Rename columns.
    df.columns = ['compound_name', 'producing_organism', 'mibig_cluster', 'canonical_smiles', 'isomeric_smiles']

    # Add compound_id column.
    df.insert(0, 'compound_id', range(1, 1 + len(df)))

    # Save to TSV.
    df.to_csv(out_tsv, sep='\t', index=False)


if __name__ == '__main__':
    main()
