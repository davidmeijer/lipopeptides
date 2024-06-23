# -*- coding: utf-8 -*-

"""Parse lipid tails from lipopeptides with RetroMol."""

import argparse
import logging
import typing as ty

import pydantic
from rdkit import Chem
from tqdm import tqdm

from retromol.retrosynthesis.chem import MolecularPattern, Molecule, ReactionRule
from retromol.retrosynthesis.parsing import Result, parse_mol, parse_molecular_patterns, parse_reaction_rules


def cli() -> argparse.Namespace:
    """Command line interface.
    
    :return: Command line arguments.
    :rtype: argparse.Namespace
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-csv", required=True, type=str)
    parser.add_argument("--out-csv", required=True, type=str)
    parser.add_argument("--log-file", default=None, type=str)
    parser.add_argument("--log-level", default="INFO", type=str)
    return parser.parse_args()


class CompoundRecord(pydantic.BaseModel):
    """Compound record.
    
    :param compound_id: Compound ID.
    :type compound_id: str
    :param compound_name: Compound name.
    :type compound_name: str
    :param compound_smiles: Compound SMILES.
    :type compound_smiles: str
    :param producing_organism: Producing organism.
    :type producing_organism: str
    :param mibig_cluster: MIBiG cluster.
    :type mibig_cluster: str
    """
    compound_id: str
    compound_name: str
    compound_smiles: str
    producing_organism: str
    mibig_cluster: str


def parse_compounds(in_csv: str) -> ty.Generator[CompoundRecord, None, None]:
    """Parse compounds from a CSV file.
    
    :param in_csv: Path to the input CSV file.
    :type in_csv: str
    :return: Generator of compound records.
    :rtype: ty.Generator[CompoundRecord, None, None]
    :raises ValueError: If a molecule cannot be parsed.
    """
    logger = logging.getLogger(__name__)

    with open(in_csv, "r") as f:
        f.readline()  # Skip header

        for line in f:    

            # Parse the line.
            (
                compound_id, 
                compound_name, 
                producing_organism, 
                mibig_cluster, 
                canonical_smiles, 
                isomeric_smiles  # Unused.
            ) = line.strip().split(",")

            # Skip if the canonical SMILES is empty.
            if not canonical_smiles:
                continue

            # Parse the molecule.
            if _ := Chem.MolFromSmiles(canonical_smiles):

                # Yield the record.
                yield CompoundRecord(
                    compound_id=compound_id,
                    compound_name=compound_name,
                    compound_smiles=canonical_smiles,
                    producing_organism=producing_organism,
                    mibig_cluster=mibig_cluster
                )

            else:  # Raise an error if the molecule cannot be parsed.
                msg = f"Unable to parse molecule with compound_id {compound_id}"
                logger.error(msg)
                raise ValueError(msg)


def main() -> None:
    """Driver code."""
    args = cli()
    logging.basicConfig(
        level=args.log_level,
        format="[%(levelname)s] %(message)s",
        filename=args.log_file
    )
    logger = logging.getLogger(__name__)

    # Parse reactions.
    reactions = [
        ReactionRule(
            name="open_lactone",
            pattern="[CR:1][ORH0:2][CR:3](=[O:4])[CR:5]>>([C:1][OH1:2].[OH1][C:3](=[O:4])[C:5])"
        ),
        ReactionRule(
            name="open_lactam",
            pattern="[C:1][CRH0:2](=[O:3])[NRH1:4][CR:5]>>([C:1][CH0:2](=[O:3])[OH].[NH2:4][C:5])"
        ),
        ReactionRule(
            name="remove_alpha_amino_acid",
            pattern="[C](=[O])([OH,NH2])[CH0,CH1,CH2][NH1,NH0][C:6](=[O:7])>>[C:6](=[O:7])([OH])"
        ),
        ReactionRule(
            name="demethylate_nitrogen_peptide_bond",
            pattern="[NH0:1][CH3]>>[NH1:1]"
        )
    ]

    # Parse monomers.
    monomers = []

    # Open out file.
    out_file = open(args.out_csv, "w")
    out_file.write("compound_id,compound_name,producing_organism,mibig_cluster,canonical_smiles,lipid_tail_smiles\n")

    # Parse compounds.
    for compound_record in tqdm(parse_compounds(args.in_csv)):

        # Create a molecule for parsing with RetroMol.
        mol = Molecule(
            name=compound_record.compound_id, 
            smiles=compound_record.compound_smiles
        )

        # Parse the molecule.
        result = parse_mol(mol, reactions, monomers)

        # Get leaves from reaction tree.
        tree = result.reaction_tree
        leafs = [node for node, props in tree.items() if len(props["children"]) == 0]
        leaf_smis = []
        for leaf in leafs:
            leaf_mol = Chem.MolFromSmiles(tree[leaf]["smiles"])
            for atom in leaf_mol.GetAtoms():
                atom.SetAtomMapNum(0)
            leaf_smi = Chem.MolToSmiles(leaf_mol)
            leaf_smis.append(leaf_smi)
        
        logger.info(f"Compound ID: {compound_record.compound_id}")
        logger.info(f"Compoud SMILES: {compound_record.compound_smiles}")
        logger.info(f"Applied reactions: {result.applied_reactions}")
        logger.info(f"Number of leaf nodes: {len(leaf_smis)}")
        for leaf_index, leaf_smi in enumerate(leaf_smis):
            leaf_index += 1
            logger.info(f"Leaf {leaf_index}: {leaf_smi}")        

        if len(leaf_smis) != 1:
            msg = f"Compound {compound_record.compound_id} has more than one lipid tail."
            logger.error(msg)
            # raise ValueError(msg)
        
        lipid_tail_smi = leaf_smis[0]

        # Write to out file.
        out_file.write(f"{compound_record.compound_id},{compound_record.compound_name},{compound_record.producing_organism},{compound_record.mibig_cluster},{compound_record.compound_smiles},{lipid_tail_smi}\n")

    out_file.close()


if __name__ == "__main__":
    main()
