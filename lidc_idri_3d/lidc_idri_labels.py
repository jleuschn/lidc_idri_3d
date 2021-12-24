# -*- coding: utf-8 -*-
import os
import json
from tqdm import tqdm
import numpy as np
from pydicom.filereader import dcmread
import pylidc as pl
from pylidc.utils import consensus
from pylidc.Annotation import feature_names

OUTPUT_PATH = '/localdata/LIDC-IDRI_labels'
os.makedirs(OUTPUT_PATH, exist_ok=True)

DATA_PATH = '/localdata/LIDC-IDRI'

ALL_CT_DIRS_FILE = os.path.join(
    os.path.dirname(__file__),
    'list_of_all_axial_512x512px_3d_ct_dirs_in_lidc_idri.json')

with open(ALL_CT_DIRS_FILE, 'r') as f:
    all_ct_dirs = json.load(f)

for ct_dir in tqdm(all_ct_dirs):
    os.makedirs(os.path.join(OUTPUT_PATH, ct_dir), exist_ok=True)
    dcm_files = [f for f in os.listdir(os.path.join(DATA_PATH, ct_dir))
                 if f.endswith('.dcm')]
    for dcm_file in dcm_files:
        dataset = dcmread(os.path.join(DATA_PATH, ct_dir, dcm_file))
        z = float(dataset.ImagePositionPatient[2])
        scans = pl.query(pl.Scan).filter(
            # pl.Scan.patient_id == str(dataset.PatientID),
            pl.Scan.series_instance_uid == str(dataset.SeriesInstanceUID),
            )
        assert scans.count() == 1
        scan = scans[0]
        labels = []
        bboxes = []
        features = []
        nodules = scan.cluster_annotations()
        for j, nodule in enumerate(nodules):
            cmask, cbbox = consensus(nodule, ret_masks=False)
            z_ind_ = np.argwhere(np.abs(scan.slice_zvals[cbbox[2]] - z) < 5e-5)
            if z_ind_.size > 0:
                assert len(z_ind_[0]) == 1
                z_ind = z_ind_[0][0]
                assert cbbox[0].step is None and cbbox[1].step is None
                if np.any(cmask[:, :, z_ind]):
                    labels.append(cmask[:, :, z_ind])
                    bboxes.append((cbbox[0].start, cbbox[1].start,
                                   cbbox[0].stop, cbbox[1].stop))
                    feature_dict_list = []
                    for a in nodule:
                        feature_dict = {
                                k: v.item() for k, v in zip(
                                        feature_names, a.feature_vals())}
                        feature_dict['num_annotations'] = len(nodule)
                        feature_dict_list.append(feature_dict)
                    features.append(feature_dict_list)
                    print('z = {:f}'.format(z))
        if labels:
            for j, label in enumerate(labels):
                np.save(
                    os.path.join(
                        OUTPUT_PATH, ct_dir,
                        os.path.splitext(dcm_file)[0] + f'_nodule{j:d}.npy'),
                    label)
            np.save(
                os.path.join(
                    OUTPUT_PATH, ct_dir,
                    os.path.splitext(dcm_file)[0] + '_bboxes.npy'),
                np.stack(bboxes))
            with open(os.path.join(
                    OUTPUT_PATH, ct_dir,
                    os.path.splitext(dcm_file)[0] + '_features.json'),
                    'w') as f:
                json.dump(features, f, indent=1)
