#!/usr/bin/env python

# Thanks to Peter Schmidtke for the guts of this script
# https://pschmidtke.github.io/blog/rdkit/crystallography/small%20molecule%20xray/xray/database/2021/01/25/cod-and-torsion-angles.html

import os
import pandas as pd
import numpy as np
import glob
from random import shuffle

# add a timeout to avoid hanging
import signal

def handler(signum, frame):
    raise Exception("Timeout")

from rdkit import Chem
from rdkit.Chem import rdMolTransforms
from rdkit.Chem import rdDetermineBonds

# ignore warnings
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

# If you use other files:
# from ETKDG paper: https://pubs.acs.org/doi/abs/10.1021/acs.jcim.5b00654
# torsions=pd.read_table("list_torsion_patterns.txt",header=None,usecols=[1])
# from Ring ETKDG paper: https://pubs.acs.org/doi/10.1021/acs.jcim.0c00025
# torsions=pd.read_table("ring_smarts_patterns.txt",header=None,usecols=[1])
# Based on the TorLib 2020 paper: https://doi.org/10.1021/acs.jcim.2c00043
torsions=pd.read_table("bouquet/data/torlib.txt",header=None,usecols=[1])

# filename template for output, e.g. t1.txt, t2.txt, etc.
out_template = 'torsions/tl{}.txt'

patterns=torsions[1]
# for debugging
# patterns=torsions[1][:3]

# bin size (in degrees)
bin_size = 1
bins = round(360 / bin_size)

# We compile the pattern, then loop through the SDF / XYZ files
angles_list = []
query_list = []
for index, torsionSmarts in enumerate(patterns):
    angles = np.zeros(bins) # create a histogram of angles with X degree bins
    angles_list.append(angles)

    torsionQuery = Chem.MolFromSmarts(torsionSmarts)
    query_list.append(torsionQuery)

# add the timeout
signal.signal(signal.SIGALRM, handler)

# loop through the files .. read SDF for connectivity, XYZ for conformer geometry
for sdf_file in glob.iglob("*/*.sdf"):
    print("Processing", sdf_file)
    signal.alarm(60) # more than enough

    try:
        mol = Chem.MolFromMolFile(sdf_file, removeHs=False)
        if mol is None:
            continue

        # read the corresponding XYZ file for the actual conformer coordinates
        xyz_file = sdf_file.replace('.sdf', '.xyz')
        if not os.path.exists(xyz_file):
            print(f"  XYZ file not found: {xyz_file}")
            continue
        xyz_mol = Chem.MolFromXYZFile(xyz_file)
        if xyz_mol is None:
            print(f"  Could not read XYZ file: {xyz_file}")
            continue

        # get the geometry from the XYZ file
        conf = xyz_mol.GetConformer(0)

        # loop through the possible SMARTS patterns from more general to more specific
        # .. we make a map of the atom indices from the SMARTS pattern
        # .. so if a more specific pattern matches later, we use that instead
        matched_list = {}
        for query in range(len(query_list)):
            torsionQuery = query_list[query]
            if torsionQuery is None:
                continue
            matches = mol.GetSubstructMatches(torsionQuery)

            try:
                # these SMARTS have atom maps, so convert them
                # http://www.rdkit.org/docs/GettingStartedInPython.html#atom-map-indices-in-smarts
                index_map = {}
                for atom in torsionQuery.GetAtoms() :
                    map_num = atom.GetAtomMapNum()
                    if map_num:
                        index_map[map_num-1] = atom.GetIdx()
                map_list = [index_map[x] for x in sorted(index_map)]

                for match in matches:
                    # get the atom maps from the SMARTS match
                    mapped = [match[x] for x in map_list]
                    key = f"{mapped[0]}-{mapped[1]}-{mapped[2]}-{mapped[3]}"

                    angle = rdMolTransforms.GetDihedralDeg(conf, mapped[0],mapped[1],mapped[2],mapped[3])
                    if (angle < 0.0):
                        angle += 360.0

                    # okay, we want to hash - e.g., 5° bins
                    angle = round(angle / bin_size) % bins

                    matched_list[key] = (query, angle)
            except Exception as e:
                print(f"  Error processing query {query}: {e}")
                continue
    finally:
        # reset the timeout, even on early continue / exception
        signal.alarm(0)

    # now we go through the matched_list
    # .. and increment the histogram for each match
    for key in matched_list:
        query, angle = matched_list[key]
        angles_list[query][angle] += 1

    # now we write out the angles
    for index in range(len(angles_list)):
        angles = angles_list[index]

        np.savetxt(out_template.format(index), angles.astype(int), fmt='%i',delimiter=',')
        # print(angles)
