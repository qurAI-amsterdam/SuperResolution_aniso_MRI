import SimpleITK as sitk
import os
import numpy as np
from os import path
from datasets.common import apply_2d_zoom_3d
import matplotlib.pyplot as plt
from datasets.shared_transforms import AdjustToPatchSize
from skimage import exposure
from evaluate.metrics import compute_ssim_for_batch, compute_psnr_for_batch
import copy


class DicomImageSlice(object):
    def __init__(self, fname):
        reader = sitk.ImageFileReader()
        reader.SetFileName(fname)
        reader.LoadPrivateTagsOn()
        reader.ReadImageInformation()

        self.imgnumber = int(reader.GetMetaData('0020|0013'))
        self.pixelspacing = tuple(float(s) for s in reader.GetMetaData('0028|0030').split('\\'))
        self.slicethickness = float(reader.GetMetaData('0018|0050'))
        self.slicespacing = float(reader.GetMetaData('0018|0088'))
        self.repetition_time = float(reader.GetMetaData('0018|0080'))  # TR
        self.echo_time = float(reader.GetMetaData('0018|0081'))  # TE
        self.heart_rate = float(reader.GetMetaData('0018|1088'))  # Heart rate
        self.sliceposition = float(reader.GetMetaData('0020|1041'))
        self.patid = str(reader.GetMetaData('0010|0020'))
        self.series_descr = str(reader.GetMetaData('0008|103e'))  # Series discription
        # Patient orientation, world coordinates (?)
        self.orientation = tuple(float(s) for s in reader.GetMetaData('0020|0037').split('\\'))
        # x,y,z coordiantes of upper left corner
        self.position = tuple(float(s) for s in reader.GetMetaData('0020|0032').split('\\'))


def dim3_to_4(oldim, slices, timepoints):
    oldarr = sitk.GetArrayFromImage(oldim)
    old_shape = oldarr.shape
    new_shape = (slices, timepoints) + old_shape[1:]
    newarr = oldarr.reshape(new_shape).transpose((1, 0, 2, 3))

    images = list()
    for im in newarr:
        img = sitk.GetImageFromArray(im)
        img.SetSpacing(oldim.GetSpacing())
        img.SetDirection(oldim.GetDirection())
        img.SetOrigin(oldim.GetOrigin())
        images.append(img)
    return images


def create_nifti(dicom_dir, outdir=None):
    reader = sitk.ImageSeriesReader()
    reader.SetLoadPrivateTags(True)
    dicom_names = reader.GetGDCMSeriesFileNames(dicom_dir)

    image_numbers = list()
    slice_idcs = list()
    slice_slicespacings = list()
    patient_orientation = list()
    patient_position = {}
    slice_position = []
    patids = list()

    for s, fname in enumerate(dicom_names):
        imslice = DicomImageSlice(fname)
        image_numbers.append(imslice.imgnumber)
        slice_slicespacings.append(imslice.slicespacing)
        patient_position[s] = imslice.position
        slice_position.append(imslice.sliceposition)
        patient_orientation.append(imslice.orientation)
        patids.append(imslice.patid)

    unique_patids = np.unique(patids)
    assert (len(unique_patids) == 1)
    unique_slicespacing = np.unique(slice_slicespacings)
    assert (len(unique_slicespacing) == 1)
    patid = imslice.patid
    slicespacing = imslice.slicespacing
    min_sliceposition, max_sliceposition = min(slice_position), max(slice_position)
    max_imagenbr = max(image_numbers)
    # print(unique_slicespacing, min_sliceposition, max_sliceposition, (max_sliceposition - min_sliceposition))
    num_of_slices = int((round(max_sliceposition - min_sliceposition)) / unique_slicespacing) + 1
    num_of_frames = int(max_imagenbr / num_of_slices)

    sorted_fnames = list()
    sorted_orientation = list()
    sorted_position = list()
    for idx in np.argsort(image_numbers):
        sorted_fnames.append(dicom_names[idx])
        sorted_orientation.append(patient_orientation[idx])
        sorted_position.append(patient_position[idx])
    try:
        assert (num_of_slices * num_of_frames == len(sorted_fnames))  # check if it makes sense.
    except AssertionError:
        print('Warning - skipped {} #frames * #slices does not match total #slices'.format(dicom_dir))
        print("{} = {} = {} * {}".format(max_imagenbr, len(image_numbers), num_of_slices, num_of_frames))
        return None, False
    reader.SetFileNames(sorted_fnames)
    image = reader.Execute()

    volumes = dim3_to_4(image, num_of_slices, num_of_frames)
    image4d = sitk.JoinSeries(volumes)
    # GetSpacing has shape [x, y, z] but spacing in z-direction is 1. It was originally stored in the dicom tag
    # under 0018|0088 and lost in the process. So we append it here again
    new_spacing = image.GetSpacing()[:2] + (slicespacing, 1.)
    image4d.SetSpacing(new_spacing)
    # print("Patient_id {}: new shape & spacing: {}".format(patid, fname), image4d.GetSize())
    if outdir is not None:
        if not os.path.isdir(outdir):
            os.makedirs(outdir, exist_ok=False)
        fname = path.join(outdir, '{}.nii.gz'.format(patid.strip()))
        sitk.WriteImage(image4d, fname, True)

    return image4d, sorted_position   # True


def get3d_img_direction(img4d: sitk.Image):
    direc = np.reshape(np.array(img4d.GetDirection()), (4, 4))
    # direc is matrix 4x4 with column1 = [Xx, Xy, Xz, 0], column2=[Yx, Yy, Yz, 0] etc.
    # we're only interested in 3x3 matrix
    return direc[:3, :3]


def compute_origin_image3d_slice(image: sitk.Image, slice_id: int, axis=0):
    direc = np.reshape(np.array(image.GetDirection()).astype(np.float64), (3, 3))
    origin_slice0 = np.array(image.GetOrigin()).astype(np.float64)
    sp_axis = image.GetSpacing()[::-1][axis]  # axis=0 -> z-axis, slice direction of sax images
    z_cosine = direc[:, -1]
    diff_xyz = z_cosine * sp_axis * slice_id  # we assume first slice has index 0, hence no origin change
    return origin_slice0 + diff_xyz


def get_sitk_transformation(image3d: sitk.Image, target_img3d: sitk.Image):
    dimension = 3
    composite = sitk.Transform(dimension, sitk.sitkComposite)
    D = np.array(image3d.GetDirection()).reshape((3, 3)).astype(np.float64)
    T_source = np.array(image3d.GetOrigin()).astype(np.float64)
    T_target = np.array(target_img3d.GetOrigin()).astype(np.float64)
    S = np.diag(np.array(image3d.GetSpacing()).astype(np.float64))
    D_inv = np.linalg.inv(np.dot(D, S))
    # D_inv = np.linalg.inv(D)
    translation = sitk.TranslationTransform(dimension)
    translation.SetOffset(T_source)
    composite.AddTransform(translation)
    affine_transform = sitk.AffineTransform(dimension)
    affine_transform.SetMatrix(D_inv.ravel())
    composite.AddTransform(affine_transform)
    # now roll forward to target image orientation
    D = np.array(target_img3d.GetDirection()).reshape((3, 3)).astype(np.float64)

    affine_transform_t = sitk.AffineTransform(dimension)
    affine_transform_t.SetMatrix(np.dot(D, S).ravel())
    composite.AddTransform(affine_transform_t)
    translation_t = sitk.TranslationTransform(dimension)
    translation_t.SetOffset(T_target)
    composite.AddTransform(translation_t)
    return composite


def create_sitk_from_data_dict(data_dict: dict) -> sitk.Image:
    assert "origin" in data_dict.keys() and "direction" in  data_dict.keys() and "spacing" in data_dict.keys()
    img_arr = data_dict['image']
    spacing = data_dict['spacing']
    if img_arr.ndim == 4:
        # 4d array
        if len(spacing) == 3:
            spacing = np.array([1, spacing[0], spacing[1], spacing[2]]).astype(np.float64)
        volumes = [sitk.GetImageFromArray(img_arr[v].astype(img_arr.dtype), False) for v in range(img_arr.shape[0])]
        img = sitk.JoinSeries(volumes)
    else:
        img = sitk.GetImageFromArray(img_arr)

    img.SetSpacing(spacing[::-1])
    # Important: we assume origin and direction are not reversed in order
    img.SetOrigin(data_dict['origin'])
    img.SetDirection(data_dict['direction'])
    return img


def get_2d_from3d(image: sitk.Image, slice_id: int) -> sitk.Image:
    size = list(image.GetSize())
    size[2] = 0
    index = [0, 0, slice_id]
    Extractor = sitk.ExtractImageFilter()
    Extractor.SetSize(size)
    Extractor.SetIndex(index)
    return Extractor.Execute(image)


def get_3d_from4d(image: sitk.Image, frame_id: int, new_spacing: np.ndarray = None,
                  patch_size=None) -> sitk.Image:
    size = list(image.GetSize())
    size[3] = 0
    index = [0, 0, 0, frame_id]
    Extractor = sitk.ExtractImageFilter()
    Extractor.SetSize(size)
    Extractor.SetIndex(index)
    return Extractor.Execute(image)


def generate_resampled_image(source_img: sitk.Image, target_img: sitk.Image,
                             interpolator=sitk.sitkCosineWindowedSinc, default_value=0):
    identity = sitk.Transform(3, sitk.sitkIdentity)
    # identity = sitk.AffineTransform(3)
    # identity.SetMatrix(np.array([1,0,0  , 0,0,1,  0, 1, 0]).astype(np.float64))
    # resampled_img = sitk.Resample(source_img, target_img, identity, interpolator, default_value)

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(target_img)
    resampler.SetInterpolator(sitk.sitkCosineWindowedSinc)
    resampler.SetDefaultPixelValue(default_value)
    resampler.SetTransform(identity)
    resampled_img = resampler.Execute(source_img)
    return resampled_img


def create_dummy_image(source_image: sitk.Image, spacing: tuple) -> sitk.Image:
    new_shape = source_image.GetSize()[::-1]
    arr_np = np.zeros(new_shape)
    new_img = sitk.GetImageFromArray(arr_np)
    new_img.SetOrigin(source_image.GetOrigin())
    new_img.SetDirection(source_image.GetDirection())
    new_img.SetSpacing(spacing)
    return new_img


def create_cross_section_3d(source_img: sitk.Image, target_img: sitk.Image):
    """
        We assume source_img is a 3D Short-axis cardiac image in sitk format [x, y, z]
        Further, target_img is the LAX (sitk) image [x, y, z] to which we want to resample the SAX image
        Both images are from the same patient
        I.e., we want a cross section through the SAX volume in exactly the same orientation as the LAX images

    """
    identity = sitk.Transform(3, sitk.sitkIdentity)
    ref_img = create_dummy_image(target_img, target_img.GetSpacing())
    resampled_sax = None
    # loop over slices in SAX image
    # for s in range(source_img.GetSize()[::-1][0]):
    # sax_img3d_slice = get_3dslice_from3d(source_img, s)
    resampled_sax = resample(source_img, identity, ref_img)
    # resampled_sax = resampled_slice if resampled_sax is None else resampled_sax + resampled_slice

    return resampled_sax


def resample(image, transform, reference, interpolator=sitk.sitkCosineWindowedSinc):
    # Output image Origin, Spacing, Size, Direction are taken from the reference
    # image in this call to Resample
    reference_image = reference
    default_value = 0
    return sitk.Resample(image, reference_image, transform, interpolator, default_value)


def myshow(img, title=None, margin=0.05, dpi=80, slice_id=None):
    if slice_id is not None:
        # important we assume one dummy dimension for 3rd dim, slices in 3d space
        nda = sitk.GetArrayViewFromImage(img)[slice_id]
        spacing = np.array(img.GetSpacing()).astype(np.float64)[:-1]
    else:
        nda = sitk.GetArrayViewFromImage(img)
        spacing = img.GetSpacing()

    ysize = nda.shape[0]
    xsize = nda.shape[1]

    figsize = (1 + margin) * ysize / dpi, (1 + margin) * xsize / dpi

    fig = plt.figure(title, figsize=figsize, dpi=dpi)
    ax = fig.add_axes([margin, margin, 1 - 2 * margin, 1 - 2 * margin])

    extent = (0, xsize * spacing[1], 0, ysize * spacing[0])

    t = ax.imshow(nda,
                  interpolation='nearest',
                  extent=extent,
                  cmap='gray')
    ax.axis("off")
    if title:
        plt.title(title)


def myshow_compare(img1, img2, img3, title=None, margin=0.05, dpi=80, do_show=True, frame_id=None):

    if frame_id is None:
        frame_range = np.arange(len(img1))
    else:
        frame_range = np.arange(frame_id, frame_id + 1)
    ssim_res1, ssim_res2, psnr_res1, psnr_res2 = [], [], [], []
    for f in frame_range:
        # important we assume one dummy dimension for 3rd dim, slices in 3d space
        nda1 = sitk.GetArrayViewFromImage(img1[f])[0]
        mask = nda1 == 0
        nda1 = exposure.equalize_hist(np.uint8(np.clip(nda1 * 255., 0, 255)))
        spacing1 = np.array(img1[f].GetSpacing()).astype(np.float64)[:-1]
        nda2 = sitk.GetArrayViewFromImage(img2[f])[0]
        nda2 = exposure.equalize_hist(np.uint8(np.clip(nda2 * 255., 0, 255)))
        spacing2 = np.array(img2[f].GetSpacing()).astype(np.float64)[:-1]
        nda3 = copy.deepcopy(sitk.GetArrayViewFromImage(img3)[f].squeeze())
        spacing3 = np.array(img3.GetSpacing()).astype(np.float64)[:-2]
        nda3 = exposure.equalize_hist(np.uint8(np.clip(nda3 * 255., 0, 255)))

        nda3[mask] = 0
        ssim1, ssim2 = compute_ssim_for_batch(nda1, nda3), compute_ssim_for_batch(nda2, nda3)
        print(ssim1, ssim2)
        psnr1, psnr2 = compute_psnr_for_batch(nda1, nda3), compute_psnr_for_batch(nda2, nda3)
        ysize = nda1.shape[0]
        xsize = nda1.shape[1]
        extent1 = (0, xsize * spacing1[1], 0, ysize * spacing1[0])
        extent2 = (0, xsize * spacing2[1], 0, ysize * spacing2[0])
        extent3 = (0, xsize * spacing3[1], 0, ysize * spacing3[0])

        figsize = 4 * (1 + margin) * ysize / dpi, 4 * (1 + margin) * xsize / dpi
        # fig, axarr = plt.subplots(num_frames, 3, figsize=(10, 5), dpi=dpi)
        ax_counter = 0
        fig, axarr = plt.subplots(1, 3, figsize=figsize, dpi=dpi)
        t = axarr[ax_counter].imshow(nda1,
                      extent=extent1,
                      cmap='gray')
        axarr[ax_counter].axis("off")

        ax_counter += 1
        t = axarr[ax_counter].imshow(nda2,
                            extent=extent2,
                            cmap='gray')
        axarr[ax_counter].axis("off")
        ax_counter += 1

        t = axarr[ax_counter].imshow(nda3,
                            extent=extent3,
                            cmap='gray')
        axarr[ax_counter].axis("off")
        if do_show:
            plt.show()
        plt.close()
        ssim_res1.append(ssim1)
        ssim_res2.append(ssim2)
        if not (np.isnan(psnr1) or np.isinf(psnr1)):
            psnr_res1.append(psnr1)
        if not (np.isnan(psnr2) or np.isinf(psnr2)):
            psnr_res2.append(psnr2)

    if (title):
        plt.title(title)
    mean_ssim1, mean_ssim2 = np.mean(np.array(ssim_res1)), np.mean(np.array(ssim_res2))
    mean_psnr1, mean_psnr2 = np.mean(np.array(psnr_res1)), np.mean(np.array(psnr_res2))
    print("Mean SSIM ACAI : CONV - {:.3f} : {:.3f}".format(mean_ssim1, mean_ssim2))
    print("Mean PSNR ACAI : CONV - {:.3f} : {:.3f}".format(mean_psnr1, mean_psnr2))
