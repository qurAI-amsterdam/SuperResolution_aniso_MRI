import os
import glob
import numpy as np
import yaml
from datasets.data_config import get_config


dta_settings = get_config('ARVC')


def get_image_file_list(search_mask_img) -> list:
    files_to_load = glob.glob(search_mask_img)
    files_to_load.sort()
    if len(files_to_load) == 0:
        raise ValueError("ERROR - get_image_file_list - Can't find any files to load in {}".format(search_mask_img))
    return files_to_load


def create_train_test_split_segmentation(split=(0.70, 0.10, 0.20), rs=None):
    dta_settings = get_config('ARVC')

    def get_subset_ids(all_patids, subset_ids) -> list:
        return [all_patids[fid] for fid in subset_ids]

    def combine_ids(original_id, pseudo_id) -> list:
        return [c_ids for c_ids in zip(original_id, pseudo_id)]

    def numpy_array_to_native_python(np_arr) -> list:
        return [val.item() for val in np_arr]

    # create new split
    assert sum(split) == 1.
    # get a list with the short-axis image files that we have in total
    # (e.g. in ~/data/ARVC/annotations/contour_ref/*.nii.gz)
    search_suffix = "*" + dta_settings.img_file_ext
    search_mask_img = os.path.expanduser(os.path.join(dta_settings.ref_label_dir, search_suffix))
    # we make a list of relative file names (root data dir is omitted)
    patient_ids = [str(os.path.basename(abs_fname)).replace(dta_settings.img_file_ext, "") for abs_fname in get_image_file_list(search_mask_img)]
    num_of_patients = len(patient_ids)
    # permute the list of all files, we will separate the permuted list into train, validation and test sets
    if rs is None:
        rs = np.random.RandomState(78346)
    ids = rs.permutation(num_of_patients)
    # create three lists of files
    patids_train = numpy_array_to_native_python(ids[:int(split[0] * num_of_patients)])
    training_ids = get_subset_ids(patient_ids, patids_train)
    c_size = int(len(training_ids))
    patids_validation = numpy_array_to_native_python(ids[c_size:c_size + int(split[1] * num_of_patients)])
    validation_ids = get_subset_ids(patient_ids, patids_validation)

    c_size += len(validation_ids)
    patids_test = numpy_array_to_native_python(ids[c_size:])
    test_ids = get_subset_ids(patient_ids, patids_test)

    # write split configuration
    split_config = {'training': combine_ids(training_ids, patids_train),
                    'validation': combine_ids(validation_ids, patids_validation),
                    'test': combine_ids(test_ids, patids_test)}

    print("INFO - Write split file {}".format(dta_settings.split_file_segmentation))
    with open(dta_settings.split_file_segmentation, 'w') as fp:
        yaml.dump(split_config, fp)


def get_abs_filenames_segmentation():

    def create_absolute_file_names(patient_ids, src_path) -> list:
        # patient_ids is a list of tuples: [0]=NLUTR32_0_1; [1]=integer number
        return [os.path.join(src_path, val[0] + dta_settings.img_file_ext) for val in patient_ids]

    patient_ids_dict = get_patient_ids_segmentation()
    return {'training': create_absolute_file_names(patient_ids_dict['training'], dta_settings.short_axis_dir),
            'validation': create_absolute_file_names(patient_ids_dict['validation'], dta_settings.short_axis_dir),
            'test': create_absolute_file_names(patient_ids_dict['test'], dta_settings.short_axis_dir)}


def get_patient_ids_segmentation(force=False) -> dict:
    """

    """
    if not os.path.isfile(dta_settings.split_file_segmentation) or force:
        create_train_test_split_segmentation()
    print("INFO - get_patient_ids_segmentation - Get split file "
          "from {}".format(dta_settings.split_file_segmentation))
    # load existing splits
    with open(dta_settings.split_file_segmentation, 'r') as fp:
        split_config = yaml.load(fp, Loader=yaml.FullLoader)
        training_ids = split_config['training']
        validation_ids = split_config['validation']
        test_ids = split_config['test']

    return {'training': training_ids, 'validation': validation_ids, 'test': test_ids}


if __name__ == "__main__":
    create_train_test_split_segmentation(split=(0.5, 0.10, 0.40))
    print(len(get_patient_ids_segmentation()['training']))
    print(len(get_patient_ids_segmentation()['validation']))
    print(len(get_patient_ids_segmentation()['test']))