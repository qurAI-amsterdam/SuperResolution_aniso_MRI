from datasets.common import get_images_in_dir
import os


def get_results_conventional(dataset, exper_src_dir="~/expers/sr/", methods=['linear', 'bspline', 'lanczos'],
                             file_suffix=None, patid_list=None, dir_suffix="conventional"):
    assert dataset in ['dHCP', 'ADNI', 'OASIS', 'ACDC', 'sunnybrook', 'ARVC']
    exper_src_dir = os.path.expanduser(exper_src_dir)
    data_generators = {}
    for meth in methods:
        # e.g. ~/sr/ACDC/conventional/linear. For spie2021 results (128x128 and different test set) we
        # created additional conventional dir: conventional_spie2021.
        path_method = os.path.join(exper_src_dir, "{}/{}/{}".format(dataset, dir_suffix, meth))
        print("INFO - Loading volumes from {}".format(path_method))
        data_generators[meth] = get_images_in_dir(path_method, dataset_name=dataset, file_suffix=file_suffix,
                                                  rescale_int=False,
                                                  do_downsample=False, downsample_steps=None, patid_list=patid_list)

    return data_generators