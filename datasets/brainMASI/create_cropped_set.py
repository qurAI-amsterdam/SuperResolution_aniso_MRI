import SimpleITK as sitk
from glob import glob
import os
import numpy as np
from scipy.ndimage.interpolation import zoom
from scipy.ndimage import gaussian_filter1d
from skimage.morphology import convex_hull_image
from datasets.dHCP import pad_image


from datasets.brainMASI.common import find_bbox_object
from datasets.ACDC.data import sitk_save


def determine_mask_for_axis(img_ref, axis=1):
    mask = img_ref != 0
    nzero = np.nonzero(mask)
    return np.min(nzero[axis]), np.max(nzero[axis])


def apply_mask_per_slice(img, mask):
    new_image = np.zeros_like(img).astype(img.dtype)
    for s in np.arange(mask.shape[0]):
        # print("Find {} slice axis {}".format(s, axis), myslice.shape)
        bbox = find_bbox_object(mask[s], padding=0)
        new_image[s, bbox.slice_x, bbox.slice_y] = img[s, bbox.slice_x, bbox.slice_y]
    return new_image


def get_foreground_mask(ref_img):
    # nib.Nifti1Header.quaternion_threshold = -1e-06
    # img = nib.load(fname)
    # back_img = compute_background_mask(img)
    # return back_img.get_data().T
    mask = np.zeros_like(ref_img).astype(np.int)
    for s in np.arange(ref_img.shape[0]):
        mask[s] = convex_hull_image(ref_img[s])

    return mask


def create_train_test_split(file_list, rs=np.random.RandomState(1234)):
    patids = np.array([int(fname.split(os.sep)[-1].replace('.nii', "")) for fname in file_list])
    print(patids)
    rs.shuffle(patids)
    return patids[:15], patids[15:]


def create_cropped_dataset(src_path='~/data/brainMASI', out_path='~/data/brainMASI_cropped',
                           limited_load=False, patid=None):
    src_path = os.path.expanduser(src_path)
    out_path = os.path.expanduser(out_path)

    out_path_img_train = os.path.join(out_path, 'training/images')
    out_path_img_test = os.path.join(out_path, 'test/images')
    if not os.path.isdir(out_path_img_train):
        os.makedirs(out_path_img_train)
    if not os.path.isdir(out_path_img_test):
        os.makedirs(out_path_img_test)
    search_path = os.path.join(src_path, "images" + os.sep + "*.nii")
    file_list = glob(search_path)
    file_list.sort()
    if limited_load:
        file_list = file_list[:2]
    train_ids, test_ids = create_train_test_split(file_list)

    print("INFO - Saving {} files to {} and {}".format(len(file_list), out_path_img_train, out_path_img_test))
    # Brain MRIs axis: 3rd dimension is axial dim first two axial slice matrix
    for fname in file_list:
        p_id = int(fname.split(os.sep)[-1].replace('.nii', ""))
        img = sitk.ReadImage(fname)
        base_fname = os.path.basename(fname)
        orig_spacing = img.GetSpacing()
        np_img = sitk.GetArrayFromImage(img).astype(np.float32)
        ax0, ax1, ax2 = np_img.shape
        ref_fname = fname.replace("images", "manual_references/6classes").replace(".nii", ".mhd")
        np_ref = sitk.GetArrayFromImage(sitk.ReadImage(ref_fname)).astype(np.int32)
        mask = get_foreground_mask(np_ref)
        min_slice_ax1, max_slice_ax1 = determine_mask_for_axis(mask, axis=1)
        # if ax1 - max_slice_ax1 != 0:
        #     max_slice_ax1 += int((ax1 - max_slice_ax1) * 0.5)
        # if min_slice_ax1 != 0:
        #     min_slice_ax1 -= int(min_slice_ax1 * 0.5)
        min_slice_ax0, max_slice_ax0 = determine_mask_for_axis(mask, axis=0)
        if ax0 - max_slice_ax0 != 0:
            max_slice_ax0 += int((ax0 - max_slice_ax0) * 0.5)
        if min_slice_ax0 != 0:
            min_slice_ax0 -= int(min_slice_ax0 * 0.5)
        min_slice_ax2, max_slice_ax2 = determine_mask_for_axis(mask, axis=2)
        if ax2 - max_slice_ax2 != 0:
            max_slice_ax2 += int((ax2 - max_slice_ax2) * 0.5)
        if min_slice_ax2 != 0:
            min_slice_ax2 -= int(min_slice_ax2 * 0.5)

        np_img = np_img[min_slice_ax0:max_slice_ax0, min_slice_ax1:max_slice_ax1, min_slice_ax2:max_slice_ax2]
        # IMPORTANT, we swap axis: z in front (axial), coronal, sagittal
        # np_img = np.swapaxes(np_img, 1, 0)
        np_img = np.transpose(np_img, axes=(1, 0, 2))
        np_img = np.flip(np_img, axis=0)
        np_img = pad_image(np_img, patch_size=tuple((256, 256)))
        if p_id in train_ids:
            out_path_img = out_path_img_train
        else:
            out_path_img = out_path_img_test
        fname_out = os.path.join(out_path_img, base_fname)
        sitk_save(fname_out, np_img, spacing=orig_spacing, dtype=np.float32)
        print("Save to {}".format(fname_out), np_img.shape)


def apply_gaussian1d(np_img, sigma):
    # we assume we always apply 1d gaussian in the z direction of np_img with shape [z, y, x]
    _, w, h = np_img.shape
    new_img = np.zeros_like(np_img).astype(np_img.dtype)
    for y in np.arange(w):
        for x in np.arange(h):
            new_img[:, y, x] = gaussian_filter1d(np_img[:, y, x], sigma)

    return new_img


def create_low_resolution_dataset(src_path='~/data/BrainMASI_cropped', out_path='~/data/BrainMASI_LR_co',
                           new_spacing_z=5, limited_load=False):
    src_path = os.path.expanduser(src_path)
    out_path = os.path.expanduser(out_path)

    for dataset in ['Training', 'Test']:
        out_path_img = os.path.join(out_path, dataset + os.sep + 'images')
        if not os.path.isdir(out_path_img):
            os.makedirs(out_path_img)
        search_path = os.path.join(src_path, dataset + os.sep + "images" + os.sep + "*.nii")
        file_list = glob(search_path)
        file_list.sort()
        if limited_load:
            file_list = file_list[:2]
        print("INFO - create_low_resolution_dataset - {} - Saving {} files to {}".format(dataset, len(file_list), out_path_img))
        for fname in file_list:
            img = sitk.ReadImage(fname)
            base_fname = os.path.basename(fname)
            orig_spacing = img.GetSpacing()[::-1]
            np_img = sitk.GetArrayFromImage(img).astype(np.float32)
            z_factor = orig_spacing[0] / new_spacing_z
            # Taken from: http://imaging.mrc-cbu.cam.ac.uk/imaging/PrinciplesSmoothing: FWHM = sigma * sqrt(8*log(2))
            # Or FWHM = 2.355 * sigma
            # The Gaussian blur should have FWHM equal to the original spacing (in our case 1mm). Which is equal to
            # a sigma =
            # LR images were generated by using a Gaussian blur with the full-width-at-half maximum (FWHM) set to
            # slice thickness before a downsampling step
            # sigma = .25 / z_factor
            sigma = orig_spacing[0] / 2.355
            np_img = apply_gaussian1d(np_img, sigma)
            new_img = zoom(np_img, tuple((z_factor,)) + tuple(orig_spacing[1:]), order=1)
            # print(z_factor, np_img.shape, new_img.shape)
            fname_out = os.path.join(out_path_img, base_fname)
            sitk_save(fname_out, new_img, spacing=np.array([new_spacing_z, orig_spacing[1], orig_spacing[2]]),
                      dtype=np.float32)
        print("INFO - {} -  Ready".format(dataset))

# create_cropped_dataset()
# create_low_resolution_dataset(limited_load=False)
