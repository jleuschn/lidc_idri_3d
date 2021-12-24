# -*- coding: utf-8 -*-
"""
Load contiguous parts of CT scans in the LIDC-IDRI database that have been
detected by ``lidc_idri_z_contiguous_parts.py``.
"""
import os
import json
import numpy as np
from tqdm import tqdm
from pydicom.filereader import dcmread

CONTIGUOUS_PARTS_FILE_LISTS_PATH = (
    '/localdata/LIDC-IDRI_z_contiguous_parts/file_lists')

DATA_PATH = '/localdata/LIDC-IDRI'

def get_num_parts(
        contiguous_parts_file_lists_path=CONTIGUOUS_PARTS_FILE_LISTS_PATH):
    """
    Return the total number of contiguous parts determined by
    ``lidc_idri_determine_z_contiguous_parts.py``.

    Parameters
    ----------
    contiguous_parts_file_lists_path : str, optional
        Directory containing the ``'info.json'`` file (and the part file list
        files) created by ``lidc_idri_determine_z_contiguous_parts.py``.
        The default is `CONTIGUOUS_PARTS_FILE_LISTS_PATH`.

    Returns
    -------
    num_parts : int
        Total number of contiguous parts.
    """
    with open(os.path.join(contiguous_parts_file_lists_path, 'info.json'),
              'r') as f:
        num_parts = len(json.load(f)['info_parts'])
    return num_parts

def load_part(
        part_index, data_path=DATA_PATH,
        contiguous_parts_file_lists_path=CONTIGUOUS_PARTS_FILE_LISTS_PATH):
    """
    Load and stack slices of a contiguous part based on the part file created
    by ``lidc_idri_determine_z_contiguous_parts.py``.

    Parameters
    ----------
    part_index : int
        Part index in ``range(get_num_parts())``.
    contiguous_parts_file_lists_path : str, optional
        Directory containing the part file list files created by
        ``lidc_idri_determine_z_contiguous_parts.py``.
        The default is `CONTIGUOUS_PARTS_FILE_LISTS_PATH`.

    Returns
    -------
    volume : array
        The stacked pixel arrays of the listed dicom files.
        Shape: ``(n, 512, 512)``, where ``n`` is the number of dicom files.
    """
    with open(os.path.join(contiguous_parts_file_lists_path,
                           'part_{:04d}.csv'.format(part_index)), 'r') as f:
        dcm_file_list = f.read().splitlines()
    slices = []
    for dcm_file in dcm_file_list:
        dcm = dcmread(os.path.join(data_path, dcm_file))
        array = dcm.pixel_array
        rescale_intercept = int(float(dcm.RescaleIntercept))
        assert float(dcm.RescaleSlope) == 1.
        assert float(dcm.RescaleIntercept) == rescale_intercept
        assert array.dtype == np.int16 or (
            array.dtype == np.uint16 and
            min(int(array.min()), int(array.min()) + rescale_intercept) >=
                    np.iinfo(np.int16).min and
            max(int(array.max()), int(array.max()) + rescale_intercept) <=
                    np.iinfo(np.int16).max)
        array = array.astype(np.int16)
        array += rescale_intercept
        slices.append(array)
    volume = np.stack(slices)
    # # in the initial version of this dataset no rescale was performed:
    # volume = np.stack([dcmread(os.path.join(data_path, dcm_file)).pixel_array
    #                    for dcm_file in dcm_file_list])
    return volume

if __name__ == '__main__':
    num_parts = get_num_parts()

    OUTPUT_PATH = '/localdata/LIDC-IDRI_z_contiguous_parts'
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    print('Total number of parts:', num_parts)
    for i in tqdm(range(get_num_parts())):
        volume = load_part(i)
        # volume = np.load(os.path.join('lidc_idri_z_contiguous_parts',
        #                               '{:04d}.npy'.format(i)))
        np.save(os.path.join(OUTPUT_PATH, 'part_{:04d}.npy'.format(i)), volume)
