import math
import cv2
import os
import SimpleITK as sitk
import numpy as np
from scipy.ndimage.measurements import center_of_mass
from datasets.ACDC.data import sitk_save


def determine_center_of_mass_segmentation(labels, lv_lbl=3, rv_lbl=1):
    # expecting labels to be numpy 3D tensor with integer labels for RV, LV at least
    cms_lv = np.round(center_of_mass(labels == lv_lbl), 0).astype(np.int)
    cms_rv = np.round(center_of_mass(labels == rv_lbl), 0).astype(np.int)
    return cms_lv, cms_rv


def determine_rotation_angle(cms_lv, cms_rv):
    delta_x = abs(cms_lv[1] - cms_rv[1])
    delta_y = abs(cms_lv[2] - cms_rv[2])
    theta_radians = math.atan2(delta_y, delta_x)
    degree = 90 - np.degrees(theta_radians)
    # cms_lv and cms_rv have three coordinates [z, y, x]. z=slice. We check whether the center of mass for RV
    # is a "lower point" in the image than the LV cmass. In that case we
    # We want LV and RV centered in the image and the line between
    # LV-RV cms should be horizontally aligned (RV left of LV).
    if cms_rv[1] > cms_lv[1]:
        degree = -1 * degree
    return degree


def det_transformed_cmass(img_2d_shape, cms, transform_matrix):
    # cms is a point, numpy array of size 2 [y, x]. This is the CMS of LV and RV. We want to rotate the image such that
    # RV is always left of LV. To center the image on the new center of mass, we need to now the rotated location
    # of the CMS. We construct a dummy slice with the CMS and then rotate. Because interpolation of one point in image
    # most probably leads to multiple points (resampling) we take the mean of the new position
    dummy_img = np.zeros(img_2d_shape)
    # set CMS in dummy images (one point).
    dummy_img[cms[0], cms[1]] = 1
    num_rows, num_cols = img_2d_shape
    dummy_img_rot = cv2.warpAffine(dummy_img, transform_matrix, (num_cols, num_rows),
                                    None, borderValue=0)  # cv2.INTER_NEAREST
    ax0, ax1 = np.nonzero(dummy_img_rot)

    return np.array([np.mean(ax0), np.mean(ax1)]).astype(np.int)


def det_translation_matrix(img_shape, new_center):
    # img_shape is numpy array or tuple of 2D image shape
    # new_center: numpy array 1D, size 2
    if not isinstance(img_shape, np.ndarray):
        img_shape = np.array(img_shape)
    img_center = (img_shape * 0.5).astype(np.int)
    rows, columns = (img_center - new_center).astype(np.int)
    # slightly confusing. But cv2 applies transformation in switched order, columns, rows
    return np.float32([[1, 0, columns], [0, 1, rows]])


def determine_transformations(labels):
    """
        param labels: is numpy 3D volume [z, y, x] with segmentation labels for at least RV and LV

    """
    # determine center of mass for LV and RV
    cms_lv, cms_rv = determine_center_of_mass_segmentation(labels)
    # determine midpoint of line between both CMS points. IMPORTANT avg_cms is 1D size 3 [z, y, x] !!!
    avg_cms = (0.5 * (cms_lv + cms_rv)).astype(np.int)
    # print("cms ", avg_cms)
    rot_angle_degrees = determine_rotation_angle(cms_lv, cms_rv)
    lbl_2d_shape = np.array(labels.shape[1:])
    num_rows, num_cols = lbl_2d_shape
    rotation_matrix = cv2.getRotationMatrix2D((num_cols / 2, num_rows / 2), rot_angle_degrees, 1)
    # IMPORTANT avg_cms is 1D size 3 [z, y, x] !!!
    # Determine new rotated CMS
    rot_avg_cms = det_transformed_cmass(lbl_2d_shape, avg_cms[1:], rotation_matrix)

    translation_matrix = det_translation_matrix(lbl_2d_shape, rot_avg_cms)
    # print(translation_matrix)
    return rotation_matrix, translation_matrix


def get_patid_data(patid, src_dir="~/data/ACDC/all_cardiac_phases"):

    pat_dir = os.path.join(src_dir, patid)
    fname_img = os.path.join(pat_dir, "{}_4d.nii.gz".format(patid))
    fname_es = os.path.join(pat_dir, "{}_frame01_gt.nii.gz".format(patid))
    if not os.path.isfile(fname_es):
        # try frame number 4, e.g. patient090
        fname_es = os.path.join(pat_dir, "{}_frame04_gt.nii.gz".format(patid))
        if not os.path.isfile(fname_es):
            raise ValueError("ERROR - file does not exist {}".format(fname_es))

    if not os.path.isfile(fname_img):
        raise ValueError("ERROR - file does not exist {}".format(fname_img))
    img = sitk.ReadImage(fname_img)
    labels = sitk.ReadImage(fname_es)
    return img, labels


def apply_transformation_to_4d(img4d, rot_matrix, trans_matrix):
    f_nums, s_nums, rows, cols = img4d.shape
    trans_img4d = np.zeros(img4d.shape)

    for f in np.arange(f_nums):
        for s in np.arange(s_nums):
            i_slice = img4d[f, s]
            mvalue = int(np.min(i_slice))
            num_rows, num_cols = i_slice.shape
            rot_img = cv2.warpAffine(i_slice, rot_matrix, (num_cols, num_rows),
                                     borderValue=mvalue)

            trans_img4d[f, s] = cv2.warpAffine(rot_img, trans_matrix, (num_cols, num_rows), cv2.INTER_CUBIC,
                                                borderValue=mvalue)
    return trans_img4d


def create_centered_dataset(src_dir="~/data/ACDC/all_cardiac_phases", out_dir="~/data/ACDC/centered",
                            patid=None):
    src_dir = os.path.expanduser(src_dir)
    out_dir = os.path.expanduser(out_dir)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=False)
    for pid in np.arange(1, 101):
        patid = "patient{:03d}".format(pid)
        img4d, labels_es = get_patid_data(patid, src_dir)
        spacing = np.array(img4d.GetSpacing()).astype(np.float64)[::-1]
        direction = np.array(img4d.GetDirection()).astype(np.float64)
        rotation_matrix, translation_matrix = determine_transformations(sitk.GetArrayFromImage(labels_es))
        t_img4d = apply_transformation_to_4d(sitk.GetArrayFromImage(img4d),
                                             rotation_matrix, translation_matrix)
        pat_outdir = os.path.join(out_dir, "{}".format(patid))
        if not os.path.isdir(pat_outdir):
            os.makedirs(pat_outdir, exist_ok=False)
        sitk_save(os.path.join(pat_outdir, "{}_4d.nii.gz".format(patid)), t_img4d, spacing=spacing,
                  origin=img4d.GetOrigin(), direction=direction)
        print("INFO - Saved {} images to {}".format(patid, pat_outdir), t_img4d.shape)