# -*- coding: utf-8 -*-
"""
Create a stable sorting for all CT scan directories of LIDC-IDRI contained in
``list_of_all_axial_512x512px_3d_ct_dirs_in_lidc_idri.json``.
The result (`z_lists`) is stored in ``lidc_idri_z_positions.json``, which for
each `ct_dir` contains

    1. ``z_lists[ct_dir]``: the sorted list of z positions. Values are repeated
       if multiple slices with the same z position occur.
    2. ``z_lists['sort_sorted_dcm_filenames_by_z_indices'][ct_dir]``: the
       indices that sort the alphabetically sorted list of dcm files in
       `ct_dir` by z position. The sorting indices are stable, so if multiple
       slices with the same z position occur, they are still ordered by
       filename alphabetically.
"""
import os
import json
import numpy as np
from tqdm import tqdm
from pydicom.filereader import dcmread

OUTPUT_PATH = '.'
os.makedirs(OUTPUT_PATH, exist_ok=True)

DATA_PATH = '/localdata/LIDC-IDRI'

ALL_CT_DIRS_FILE = os.path.join(
    os.path.dirname(__file__),
    'list_of_all_axial_512x512px_3d_ct_dirs_in_lidc_idri.json')

with open(ALL_CT_DIRS_FILE, 'r') as f:
    all_ct_dirs = json.load(f)

z_lists = {'sort_sorted_dcm_filenames_by_z_indices': {}}
for ct_dir in tqdm(all_ct_dirs):
    dcm_files = sorted(
        [f for f in os.listdir(os.path.join(DATA_PATH, ct_dir))
         if f.endswith('.dcm')])
    z_list = []
    for dcm_file in dcm_files:
        dataset = dcmread(os.path.join(DATA_PATH, ct_dir, dcm_file))
        z = float(dataset.ImagePositionPatient[2])
        z_list.append(z)
    z_list = np.array(z_list)
    sort_by_z_indices = np.argsort(z_list, kind='stable')
    z_lists['sort_sorted_dcm_filenames_by_z_indices'][ct_dir] = (
        sort_by_z_indices.tolist())
    z_lists[ct_dir] = z_list[sort_by_z_indices].tolist()

with open(os.path.join(OUTPUT_PATH, 'lidc_idri_z_positions.json'),
          'w') as f:
    json.dump(z_lists, f, indent=1)
