# -*- coding: utf-8 -*-

"""Cluster lipopeptide starter structures."""

import argparse 
import logging
import os
from typing import Dict, Generator

import matplotlib.pyplot as plt
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from sklearn.decomposition import PCA
from scipy.stats import gaussian_kde


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

            # Parse out other properties from the record.
            parsed_record["name"] = record["trivial_name"]
            parsed_record["genus"] = record["genus"]
            parsed_record["species"] = record["species"]
            parsed_record["category"] = record["category"]

            yield parsed_record


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

    labels_category = np.array(labels_category)
    labels_genus = np.array(labels_genus)
    labels_species = np.array(labels_species)

    # Get fingerprints for all starters.
    mols = [Chem.MolFromSmiles(s) for s in starters]
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

    # Plot the PCA for category labels.
    unique_labels = set(labels_category)
    
    # Sort the labels so that the colors are consistent across different runs.
    unique_labels = sorted(unique_labels)

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
    label_to_color = {label: color for label, color in zip(unique_labels, palette)}

    plt.figure(figsize=(12, 8))
    for label in unique_labels:

        # First plot a density plot.
        data = pcs[labels_category == label].T
        if not data.shape[1] < 10 and label not in ["unknown", "unassigned"]:
            kde = gaussian_kde(data)
            xgrid = np.linspace(pcs[:, 0].min() - 0.5, pcs[:, 0].max() + 0.5, 100)
            ygrid = np.linspace(pcs[:, 1].min() - 0.5, pcs[:, 1].max() + 0.5, 100)
            Xgrid, Ygrid = np.meshgrid(xgrid, ygrid)
            Z = kde(np.vstack([Xgrid.ravel(), Ygrid.ravel()])).reshape(Xgrid.shape)
            plt.contour(Xgrid, Ygrid, Z, levels=10, colors=label_to_color[label], alpha=0.5, zorder=1)

    for label in unique_labels:
        jitter = 0.05

        # Create a grid of points.
        x = pcs[labels_category == label, 0]
        y = pcs[labels_category == label, 1]
        color = label_to_color[label]

        # Add a bit of jitter to x and y values so that they are not overlapping.
        x += np.random.normal(0, jitter, x.shape)
        y += np.random.normal(0, jitter, y.shape)

        # Add line around the points.
        plt.scatter(
            x, y, 
            c=color, 
            label=f"{label} ({len(x)})",
            edgecolor="black", 
            linewidth=0.5,
            zorder=2
        )

    # Remove values across the axes.
    plt.xticks([])
    plt.yticks([])

    # Increae font size for the labels.
    plt.xlabel(f"PC 1 ({explained_variance[0] * 100:.2f}%)", fontsize=14)
    plt.ylabel(f"PC 2 ({explained_variance[1] * 100:.2f}%)", fontsize=14)

    # Give the legend a title. Title needs to be in bold.
    plt.legend(title="Category", title_fontsize="14", fontsize="12")
    
    path_out = os.path.join(args.o, "starter_clusters.png")
    plt.savefig(path_out, dpi=300)

    # List 10 most important features for PC1 and PC2 respectively.
    feature_importance_pc1 = pca.components_[0]
    feature_importance_pc2 = pca.components_[1]

    top_n = 10

    # TODO: check which features are most important for the tails across the PCs.
    # TODO: keep track of how bits are set so you can do a readout to get insight into these features.

    logger.info("\n10 most important features for PC1:")
    for i in np.argsort(feature_importance_pc1)[-top_n:]:
        logger.info(f"feature {i}: {feature_importance_pc1[i]}")

    logger.info("\n10 most important features for PC2:")
    for i in np.argsort(feature_importance_pc2)[-top_n:]:
        logger.info(f"feature {i}: {feature_importance_pc2[i]}")
    
    # TODO: get lengths/sizes of tails and plot them in some way as well.

    logger.info("goodbye!")
    

if __name__ == "__main__":
    main()
