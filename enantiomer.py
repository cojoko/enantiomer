
import pandas as pd
import numpy as np

from rdkit import Chem
from rdkit.Chem import PandasTools
from rdkit import DataStructs
from rdkit.Chem import AllChem

from sklearn.neighbors import NearestNeighbors

import boto3
import s3fs

import logging
logging.getLogger().setLevel(logging.INFO)



"""Loads an sdf file from the local directory
"""


def load_sdf(filename):


    try:
        logging.info("Loading dataframe from .SDF...")
        ref_mols = PandasTools.LoadSDF(filename, smilesName='smiles')
        logging.info("Done.")
        return ref_mols
    except Exception as e:
        logging.warn("There was a problem loading .SDF file.")
        logging.warn(e)


def smile_to_fp(smile):
    '''
    Converts a smile to its Morgan fingerprint.
    :param smile: A string representing a smile to be converted
    :return: A string representing the Morgan fingerprint of the passed smile
    '''

    mol = Chem.MolFromSmiles(smile)
    fingerprint = Chem.RDKFingerprint(mol)
    return fingerprint


def get_closest_n_mols(test_smiles, ref_mols, n=1):
    """
    Returns a list of the `n` molecules from `ref_mols` with the highest
    Tanimoto similarity to each molecule in `test_smiles`.
    :param test_smiles: A list of SMILES strings representing the test 
    molecules
    :param ref_mols: A set of reference molecules for similarity calculation
    :param n: An int representing the number of most similar molecules to find
    from `ref_mols` for each molecule in `test_smiles`.
    :return: A list of `len(test_smiles)` lists, each containing `n` tuples.
    Each tuple should contain both the SMILES string representation AND 
    Tanimoto similarity of the similar molecule in `ref_mols`. Order by 
    Tanimoto similarity in descending order.
    """

    # Drop rows where mol fails to load
    ref_mols = ref_mols.dropna()

    # Add a column of fingerprints to dataframe
    logging.info("Generating Fingerprints...")
    ref_mols["fp"] = ref_mols['smiles'].map(smile_to_fp)
    logging.info("Done.")

    # Create feature list from fingerprint column
    X = ref_mols["fp"].tolist()

    # Unsupervised learner for searching nearest neighbor in chemical space
    neigh = NearestNeighbors(
        n_neighbors=n, metric="hamming", radius=2)

    # Fit features to learner
    logging.info("Fitting Features...")
    neigh.fit(X)
    logging.info("Done.")

    # Create a list of fingerprints from list of sample smiles
    sample_test_fps = []
    for smile in test_smiles:
        sample_test_fps.append(smile_to_fp(smile))

    # Generate a list of n closest neighbors for each sample. A list of lists of indicies.
    kneighbor_list = neigh.kneighbors(
        sample_test_fps, n_neighbors=n, return_distance=False)
    
    logging.info("Neighbor List:")
    logging.info(kneighbor_list)

    # The final output list. Will contain one list of neighbor tuples for each sample
    result_list = []
    for ctr, idxs in enumerate(kneighbor_list):

        # Iterate through the fingerprint of each test sample
        test_fp = sample_test_fps[ctr]

        neighbor_tuples = []

        # Iterate through the index list returned by the kneighbors class
        for idx in idxs:

            # Get smile and ts of neighbor molecule by dataframe index
            smile = ref_mols["smiles"].iloc[idx]
            ref_fp = ref_mols["fp"].iloc[idx]
            ts = DataStructs.FingerprintSimilarity(test_fp, ref_fp)

            assert Chem.RDKFingerprint(Chem.MolFromSmiles(smile)) == ref_fp
            
            # Create tuple and add to sample list
            neighbor_tuples.append((smile, ts))

        # Add sample list to output list
        result_list.append(neighbor_tuples)

    # Return output list
    return result_list


if __name__ == '__main__':

    lib_path = "./demo_library.sdf"
    virtual_library = load_sdf(lib_path)
    N = 2
    
    sample_test_smiles = [
        'C[C@]12CCCN1CC1=NNC(=O)c3cc(F)cc4[nH]c2c1c34',
        'Cn1ncnc1[C@H]1c2n[nH]c(=O)c3cc(F)cc(c23)N[C@@H]1c1ccc(F)cc1',
        'C[C@]1(c2nc3c(C(N)=O)cccc3[nH]2)CCCN1'
    ]
    
    output = get_closest_n_mols(sample_test_smiles, virtual_library, n=N)

    expected_output = [
        [
            ('CCC1=C[C@@H]2CN(C1)Cc1c([nH]c3ccccc13)[C@@](C(=O)OC)(c1cc3c(cc1OC)N(C)[C@H]1[C@](O)(C(=O)OC)[C@H](OC(C)=O)[C@]4(CC)C=CCN5CC[C@]31[C@@H]54)C2', 0.764),
            ('CCC1=CC2CN(C1)Cc1c([nH]c3ccccc13)[C@@](C(=O)OC)(c1cc3c(cc1OC)N(C)[C@H]1[C@@](O)(C(=O)OC)[C@H](OC(C)=O)[C@]4(CC)C=CCN5CC[C@]31[C@@H]54)C2', 0.764)
        ],
        [
            ('Cn1ncnc1C1c2n[nH]c(=O)c3cc(F)cc(c23)NC1c1ccc(F)cc1', 1.0),
            ('CCC1=C[C@@H]2CN(C1)Cc1c([nH]c3ccccc13)[C@@](C(=O)OC)(c1cc3c(cc1OC)N(C)[C@H]1[C@](O)(C(=O)OC)[C@H](OC(C)=O)[C@]4(CC)C=CCN5CC[C@]31[C@@H]54)C2', 0.697)
        ],
        [

            # Origional, non-enantiomer output: ('O=C(Nc1cccc(-c2nnn[nH]2)c1)c1cccc2[nH]cnc12', 0.506),
            ('CSCC[C@H](NC(=O)CC(C)(C)C)c1nc2ccccc2[nH]1', 0.468),
            # Origional output: ('O=C(Nc1cccc(-c2nnn[nH]2)c1)c1cc(F)cc2[nH]cnc12', 0.493)
            ('O=C(Nc1cccc(-c2nnn[nH]2)c1)c1cccc2[nH]cnc12', 0.506)

        ]
    ]

    # Verify that your output matches the expected output
    assert len(output) == len(sample_test_smiles)
    for i in range(len(sample_test_smiles)):

        assert len(output[i]) == N
        for j in range(N):

            output_smiles = output[i][j][0]
            expected_smiles = expected_output[i][j][0]
            output_ts = output[i][j][1]
            expected_ts = expected_output[i][j][1]

            print("output:")
            print(output[i][j])
            print("expected:")
            print(expected_output[i][j])

            output_ts = output[i][j][1]
            expected_ts = expected_output[i][j][1]
            assert output_smiles == expected_smiles
            np.testing.assert_almost_equal(output_ts, expected_ts, decimal=3)

    print('Success! Your output matches the expected output.')
