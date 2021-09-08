# -*- coding: utf-8 -*-
"""
Determine contiguous parts of CT scans in the LIDC-IDRI database.

Steps:
    
    1. Determine whether scans are composed of multiple acquisitions.
       We would like to detect these cases in order to split them, in order to
       avoid jumps in a single volume between neighbouring z slices that have
       been acquired at different times (i.e. the patient probably has moved).
       We try to identify such cases from meta data only, which may be
       insufficient, so it can well be we miss some cases with multiple
       acquisitions.
       Ideally this could be identified by the 'AcquisitionNumber' field, but
       it is not present in all scans and its value is not always indicative.
       Therefore, fields from a list of potentially indicative fields are tried
       out one after the other. Thereby only the values of the first and the
       last z slice are compared, as they are assumed to belong to different
       acquisitions if there are multiple.
    2. Exclude some slices which are either identical duplicates of others or
       lie far outside of the rest of the scanned volume.
    3. Split the parts on non-standard steps, i.e. where the step size deviates
       from the most common step size in the scan directory.
    4. Omit parts that contain less than 64 slices.
    5. Write each part to a csv file that lists the dicom filenames.
       These lists of dicom files each have equidistant step sizes in z
       direction, however the step can differ between different scans.
"""
import os
import json
import numpy as np
from pydicom.filereader import dcmread
from pydicom.tag import Tag
from pydicom.datadict import keyword_for_tag

OUTPUT_PATH = '/localdata/LIDC-IDRI_z_contiguous_parts/file_lists/'
os.makedirs(OUTPUT_PATH, exist_ok=True)

DATA_PATH = '/localdata/LIDC-IDRI'

ABS_TOL = 7e-5  # absolute tolerance for z positions and steps

MIN_SLICES = 64

KEYWORDS_IDENTIFYING_ACQUISITION = [
    'AcquisitionNumber',
    'ExposureTime',
]
TAGS_TO_IGNORE_FOR_ACQUISITION_DETECTION = [
    Tag('UID'),
    Tag('SOPInstanceUID'),
    Tag('ReferencedSOPInstanceUID'),
    Tag('StorageMediaFileSetUID'),
    Tag('InstanceNumber'),
    Tag('InstanceCreationTime'),
    Tag('ImagePositionPatient'),
    Tag('SliceLocation'),
    Tag('Location'),
    Tag('PixelData'),
    Tag('SmallestImagePixelValue'),
    Tag('LargestImagePixelValue'),
    Tag('ContentTime'),
    Tag('Exposure'),  # only few unique values in some cases due to unit
    Tag('TableHeight'),  # two unique values occur in LIDC-IDRI-0711 and
                         # LIDC-IDRI-0995, but no pronounced jump is visible
                         # where the values changes
    Tag('WindowCenter'),  # missing in some slices of LIDC-IDRI-0995
    Tag('WindowWidth'),  # missing in some slices of LIDC-IDRI-0995
]
# fields that also seem ignorable for acquisition detection:
# 'AcquisitionComments', 'AcquisitionTime', 'XRayTubeCurrent'

with open(os.path.join(
        os.path.dirname(__file__),
        'list_of_all_axial_512x512px_3d_ct_dirs_in_lidc_idri.json'), 'r') as f:
    all_ct_dirs = json.load(f)
    
with open(os.path.join(
        os.path.dirname(__file__), 'lidc_idri_z_positions.json'), 'r') as f:
    z_lists = json.load(f)

def get_dcm_files_sorted_by_z(ct_dir, data_path=DATA_PATH):
    """
    Return the dcm filenames sorted by z position like specified by
    ``lidc_idri_z_positions.json``, i.e. by first sorting the dcm filenames
    alphabetically and then applying the
    ``'sort_sorted_dcm_filenames_by_z_indices'`` for the respective CT scan
    directory.

    Parameters
    ----------
    ct_dir : str
        CT scan directory relative to `data_path`.
    data_path : str, optional
        Path to the LIDC-IDRI dataset. The default is `DATA_PATH`.

    Returns
    -------
    dcm_files_sorted_by_z : list of str
        The dcm filenames sorted by z position like specified by
        ``lidc_idri_z_positions.json``
    """
    dcm_files = sorted([
        f for f in os.listdir(os.path.join(data_path, ct_dir))
        if f.endswith('.dcm')])
    dcm_files_sorted_by_z = [
        dcm_files[j]
        for j in z_lists['sort_sorted_dcm_filenames_by_z_indices'][ct_dir]]
    return dcm_files_sorted_by_z

def detect_multiple_acquisitions_from_meta_data(
        ct_dir, data_path=DATA_PATH, debug=True, debug_unique_values_thr=5,
        debug_verbose=True):
    """
    Try to detect whether a CT scan is composed of multiple acquisitions.
    The detection is performed by comparing the meta data of the first and the
    last z slice, according to the sorting indices provided by
    ``lidc_idri_z_positions.json``.

    Parameters
    ----------
    ct_dir : str
        CT scan directory relative to `data_path`.
    data_path : str, optional
        Path to the LIDC-IDRI dataset. The default is `DATA_PATH`.
    debug : bool, optional
        Whether to verify that each tag is either listed in
        `TAGS_TO_IGNORE_FOR_ACQUISITION_DETECTION` or has more than
        `debug_unique_values_thr` different values. The default is `True`.
    debug_unique_values_thr : int, optional
        If ``debug=True`` is passed, assume tags to be not suitable resp.
        irrelevant if there are at least this many different values for it in
        the CT scan directory.
    debug_verbose : bool, optional
        If ``debug=True`` is passed, whether to print when ignoring a tag due
        to the number of unique values being at least
        `debug_unique_values_thr`.

    Returns
    -------
    distinguish_by_keyword : str or None
        If there seem to be multiple acquisitions, the name of a DICOM keyword
        by which acquisitions can be distinguished; else `None` is returned.
    """
    dcm_files_sorted_by_z = get_dcm_files_sorted_by_z(
        ct_dir, data_path=data_path)
    dataset_first = dcmread(os.path.join(data_path, ct_dir,
                                         dcm_files_sorted_by_z[0]))
    dataset_last = dcmread(os.path.join(data_path, ct_dir,
                                        dcm_files_sorted_by_z[-1]))
    for keyword in KEYWORDS_IDENTIFYING_ACQUISITION:
        value_first = getattr(dataset_first, keyword, None)
        value_last = getattr(dataset_last, keyword, None)
        if value_last is not None and value_last != value_first:
            if debug:
                # verify that the keyword has less than 
                # `debug_unique_values_thr` different values
                values = []
                for dcm_file in dcm_files_sorted_by_z:
                    values.append(
                        dcmread(os.path.join(DATA_PATH, ct_dir, dcm_file))[
                            keyword].value)
                if len(np.unique(values)) >= debug_unique_values_thr:
                    raise RuntimeError(
                        'in CT scan directory \'{}\': keyword \'{}\' does not '
                        'seem suitable to detect acquisitions, as there are '
                        'less than {:d} different values'.format(
                            ct_dir, keyword, debug_unique_values_thr))
            return keyword
    if not debug:
        return None  # return early, i.e. do not check other tags
    # verify that each tag is either listed in
    # `TAGS_TO_IGNORE_FOR_ACQUISITION_DETECTION` or has at least
    # `debug_unique_values_thr` different values
    for tag in dataset_first.keys():
        if any(tag == t for t in TAGS_TO_IGNORE_FOR_ACQUISITION_DETECTION):
            continue
        value_first = dataset_first[tag].value
        if tag not in dataset_last:
            raise RuntimeError(
                'in CT scan directory \'{}\': tag \'{}\' is only present '
                'in a subset of slices, should be evaluated manually and '
                'then added either to `KEYWORDS_IDENTIFYING_ACQUISITION` or '
                'to `TAGS_TO_IGNORE_FOR_ACQUISITION_DETECTION`'.format(
                    ct_dir, keyword_for_tag(tag) or tag))
        value_last = dataset_last[tag].value
        if value_last != value_first:
            values = []
            for dcm_file in dcm_files_sorted_by_z:
                values.append(dcmread(os.path.join(DATA_PATH, ct_dir,
                                                   dcm_file))[tag].value)
            if len(np.unique(values)) < debug_unique_values_thr:
                raise RuntimeError(
                    'in CT scan directory \'{}\': tag \'{}\' could identify '
                    'acquisitions, should be evaluated manually and then '
                    'added either to `KEYWORDS_IDENTIFYING_ACQUISITION` or to '
                    '`TAGS_TO_IGNORE_FOR_ACQUISITION_DETECTION`. Values are '
                    '{}.'.format(ct_dir, keyword_for_tag(tag) or tag, values))
            else:
                if debug_verbose:
                    print(
                        'in CT scan directory \'{}\': tag \'{}\' does not '
                        'seem to identify acquisitions, as there are at least '
                        '{:d} different values'
                        .format(ct_dir, keyword_for_tag(tag) or tag,
                                debug_unique_values_thr))
    return None

def get_indices_to_exclude_from_slices_sorted_by_z(
        ct_dir, data_path=DATA_PATH):
    """
    Return indices of slices to exclude from the list of slices of a CT scan
    directory sorted like by :func:`get_dcm_files_sorted_by_z`.

    Parameters
    ----------
    ct_dir : str
        CT scan directory relative to `data_path`.
    data_path : str, optional
        Path to the LIDC-IDRI dataset. The default is `DATA_PATH`.

    Returns
    -------
    indices_to_exclude : list of int
        Indices of slices to exclude from the list of slices sorted like by
        :func:`get_dcm_files_sorted_by_z`.
    """
    dcm_files_sorted_by_z = get_dcm_files_sorted_by_z(
        ct_dir, data_path=data_path)
    diff = np.diff(z_lists[ct_dir])

    # exclude duplicate slices that are pixel-wise identical
    # (occurs in LIDC-IDRI-0572)
    zero_step_inds = np.argwhere(diff < ABS_TOL).flatten()
    indices_to_exclude = []
    for zero_step_ind in zero_step_inds:
        dataset_before = dcmread(
            os.path.join(data_path, ct_dir,
                         dcm_files_sorted_by_z[zero_step_ind]))
        dataset_after = dcmread(
            os.path.join(data_path, ct_dir,
                         dcm_files_sorted_by_z[zero_step_ind+1]))
        if np.all(dataset_before.pixel_array == dataset_after.pixel_array):
            indices_to_exclude.append(zero_step_ind+1)

    # exclude first or last slice if the step is unreasonably large
    diff_unique, diff_counts = np.unique(diff, return_counts=True)
    most_common_diff = diff_unique[np.argmax(diff_counts)]
    if diff[0] > most_common_diff * 10.:
        indices_to_exclude.append(0)
    if diff[-1] > most_common_diff * 10.:
        indices_to_exclude.append(len(z_lists[ct_dir])-1)

    indices_to_exclude.sort()
    return indices_to_exclude

def get_contiguous_parts(ct_dir, data_path=DATA_PATH, min_slices=2,
                         debug=True, debug_detect=True):
    """
    Return lists of slice indices that form contiguous parts of a CT scan
    directory, whereby the indices refer to the slice list returned by
    :func:`get_dcm_files_sorted_by_z`.

    Parameters
    ----------
    ct_dir : str
        CT scan directory relative to `data_path`.
    data_path : str, optional
        Path to the LIDC-IDRI dataset. The default is `DATA_PATH`.
    min_slices : int, optional
        Minimum number of slices in a part. Parts with less slices are omitted.
        The default is `2`.
    debug : bool, optional
        Whether to print debug messages in this function.
        The default is `True`.
    debug_detect : bool, optional
        The `debug` parameter to `detect_multiple_acquisitions_from_meta_data`.
        The default is `True`.

    Returns
    -------
    part_indices_list : list of list of int
        List of lists of slice indices with each list specifying a part with
        equidistant steps in the slice list returned by
        :func:`get_dcm_files_sorted_by_z`.
    """
    dcm_files_sorted_by_z = get_dcm_files_sorted_by_z(ct_dir)
    datasets = [dcmread(os.path.join(data_path, ct_dir, dcm_file))
                for dcm_file in dcm_files_sorted_by_z]
    distinguish_by_keyword = detect_multiple_acquisitions_from_meta_data(
        ct_dir, data_path=data_path, debug=debug_detect)
    if distinguish_by_keyword is None:
        # one part
        part_indices_list = [[j for j in range(len(datasets))]]
    else:
        # multiple parts to be distinguished by `distinguish_by_keyword`
        values = [d[distinguish_by_keyword].value for d in datasets]
        unique_values = np.unique(values)
        part_indices_list = [
            [j for j, v in enumerate(values) if v == unique_value]
            for unique_value in unique_values]
        if debug:
            print('detected {:d} acquisitions in CT scan directory \'{}\''
                  .format(len(part_indices_list), ct_dir))
    # exclude indices
    indices_to_exclude = get_indices_to_exclude_from_slices_sorted_by_z(
        ct_dir, data_path=data_path)
    if debug and indices_to_exclude:
        print('Excluding indices {} from CT scan directory \'{}\''.format(
            indices_to_exclude, ct_dir))
    part_indices_list = [
        [j for j in part_indices if j not in indices_to_exclude]
        for part_indices in part_indices_list]
    # split on non-standard steps
    part_indices_after_splits_list = []
    for part_indices in part_indices_list:
        part_z_list = [z_lists[ct_dir][j] for j in part_indices]
        diff = np.diff(part_z_list)
        diff_unique, diff_counts = np.unique(diff, return_counts=True)
        most_common_diff = diff_unique[np.argmax(diff_counts)]
        is_non_standard_step = np.abs(diff - most_common_diff) >= ABS_TOL
        if np.any(is_non_standard_step):
            assert not np.any(diff[is_non_standard_step] < ABS_TOL), (
                'zero-steps should be filtered out already')
            non_standard_step_indices = np.argwhere(
                np.abs(diff - most_common_diff) >= ABS_TOL).flatten()
            split_parts_list = []
            for lower, upper in zip(
                    np.concatenate([[0], non_standard_step_indices]),
                    np.concatenate([non_standard_step_indices, [len(diff)]])):
                if upper - lower >= min_slices:
                    split_parts_list.append(part_indices[lower+1:upper+1])
                else:
                    if debug:
                        print('Omitting part between non-standard steps with '
                              'only {:d} slices'.format(upper - lower))
            if debug:
                print('split a single acquisition into multiple parts due to '
                      'non-standard steps')
            part_indices_after_splits_list += split_parts_list
        else:
            part_indices_after_splits_list.append(part_indices)
    part_indices_list = part_indices_after_splits_list
    return part_indices_list

if __name__ == '__main__':
    dcm_file_lists = []
    info = {'info_parts': []}
    for ct_dir in all_ct_dirs:
        print('starting with CT scan directory \'{}\''.format(ct_dir))
        dcm_files_sorted_by_z = get_dcm_files_sorted_by_z(ct_dir)
        part_indices_list = get_contiguous_parts(ct_dir)
        assert min((len(a) for a in part_indices_list)) >= 3
        print('found {:d} contiguous parts in CT scan directory \'{}\''.format(
            len(part_indices_list), ct_dir))
        for part_indices in part_indices_list:
            print('\t* {:d} slices'.format(len(part_indices)))
            part_z_list = [z_lists[ct_dir][j] for j in part_indices]
            diff = np.diff(part_z_list)
            diff_unique, diff_counts = np.unique(diff, return_counts=True)
            most_common_diff = diff_unique[np.argmax(diff_counts)]
            assert np.max(np.abs(diff_unique - most_common_diff)) < ABS_TOL
            if len(part_indices) >= MIN_SLICES:
                dcm_file_list = [dcm_files_sorted_by_z[j]
                                 for j in part_indices]
                dcm_file_lists.append(dcm_file_list)
                with open(os.path.join(
                        OUTPUT_PATH,
                        'part_{:04d}.csv'.format(len(dcm_file_lists)-1)),
                        'w') as f:
                    for dcm_file in dcm_file_list:
                        f.write(os.path.join(ct_dir, dcm_file) + '\n')
                info['info_parts'].append({'z_start': part_z_list[0],
                                           'z_stop': part_z_list[-1],
                                           'z_step': most_common_diff})
            else:
                print('\t\tomitting')
        print('\n')
    with open(os.path.join(OUTPUT_PATH, 'info.json'), 'w') as f:
        json.dump(info, f, indent=True)

# some notes from previous semi-manual looking at the data, whereby the step
# size was used to identify exceptions only, i.e. no acquisition detection
# by meta info was done before
# 'LIDC-IDRI-0085/01-01-2000-96172/30084-46659': 'split',  # overlapping acquisitions
# 'LIDC-IDRI-0123/01-01-2000-95866/3000648-68219': 'split,crop',  # inconsistent steps
# 'LIDC-IDRI-0146/01-01-2000-37649/3000653-67579': 'split',  # overlapping acquisitions
# 'LIDC-IDRI-0267/01-01-2000-CT THORAX WO CONTRAST-75468/2-CHEST-23992': 'split',  # inconsistent steps
# 'LIDC-IDRI-0418/01-01-2000-71464/3000651-51680': 'split',  # overlapping acquisitions
# 'LIDC-IDRI-0514/01-01-2000-87995/3000281-52230': 'crop',  # at lower end
# 'LIDC-IDRI-0572/01-01-2000-CHEST-03884/31067-ACRIN LARGE-87949': 'crop',  # at upper end
# 'LIDC-IDRI-0672/01-01-2000-CHEST-44829/31265-Recon 2 ACRIN LARGE-35798': 'split',  # inconsistent steps, close to nodule
# 'LIDC-IDRI-0979/01-01-2000-22128/3000652-81401': 'split',  # overlapping acquisitions
