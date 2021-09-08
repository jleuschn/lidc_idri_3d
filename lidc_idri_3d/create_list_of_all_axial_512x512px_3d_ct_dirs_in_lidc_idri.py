# -*- coding: utf-8 -*-
import os
import json
from pydicom.filereader import dcmread

DATA_PATH = '/localdata/LIDC-IDRI'

OUTPUT_FILE = os.path.join(
    os.path.dirname(__file__),
    'list_of_all_axial_512x512px_3d_ct_dirs_in_lidc_idri.json')

all_ct_dirs = []

for root, _, files in os.walk(DATA_PATH):
    dcm_files = [f for f in files if f.endswith('.dcm')]
    if len(dcm_files) > 1:
        dcm_file = dcm_files[0]
        dataset = dcmread(os.path.join(root, dcm_file))
        if (dataset.Rows == 512 and dataset.Columns == 512 and
                [float(a) for a in dataset.ImageOrientationPatient] == [
                    1., 0., 0., 0., 1., 0.]):
            all_ct_dirs.append(os.path.relpath(root, DATA_PATH))

all_ct_dirs = sorted(all_ct_dirs)

# with open(OUTPUT_FILE, 'w') as json_file:
#     json.dump(all_ct_dirs, json_file, indent=True)
