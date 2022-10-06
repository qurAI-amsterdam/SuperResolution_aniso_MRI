import argparse
import os
from pathlib import Path
import numpy as np
import torch
from kwatsch.get_trainer import get_trainer_dynamic
import SimpleITK as sitk
from typing import Tuple, List
from tqdm import tqdm, tqdm_notebook


def create_super_volume(trainer, images: torch.tensor, alpha_range: np.ndarray, use_original=False, labels=None) -> dict:
    """

    :param trainer: Assuming images is tensor of patient volume with shape [z, 1, y, x] or [z, y, x]

    :param images:
    :param alpha_range:
    :param use_original: boolean, create sr volume with original slices that are not reconstructed or reconstruct all slices
    :return:
    """

    with_labels = False if labels is None else True
    if images.dim() == 3:
        # make sure [z, 1, y, x]
        images = torch.unsqueeze(images, dim=1)
    num_slices, _, w, h = images.size()
    if with_labels and labels.dim() == 3:
        labels = torch.unsqueeze(labels, dim=1)
    images = images if labels is None else torch.cat([images, labels], dim=1)

    if not use_original:
        if images.dtype != torch.cuda.FloatTensor:
            images = images.to('cuda')
        recon_dict = trainer.predict(images)
        if with_labels:
            recon_volume = recon_dict['image'].detach().cpu()
            recon_labels = recon_dict['pred_labels'].detach().cpu()

        else:
            recon_volume = recon_dict.detach().cpu()
            recon_labels = None
    else:
        recon_volume = images
        recon_labels = None if not with_labels else labels
    images2 = images[1:]
    images1 = images[:-1]
    interp_slices, interp_slice_labels = None, None
    for i, alpha in enumerate(alpha_range):
        inter_result_dict = latent_space_interp(alpha, trainer, images2, images1, with_labels=with_labels)
        interp_img = inter_result_dict['inter_image']
        inter_label = None if not with_labels else inter_result_dict['inter_label']
        interp_slices = interp_img if interp_slices is None else torch.cat([interp_slices, interp_img], dim=1)
        if with_labels:
            interp_slice_labels = inter_label if interp_slice_labels is None else torch.cat([interp_slice_labels, inter_label], dim=1)
    # interp_slices = rescale_tensor(interp_slices)
    new_volume, new_labels = None, None
    for i in range(num_slices - 1):
        new_volume = torch.cat([recon_volume[i], interp_slices[i]]) if new_volume is None else torch.cat \
                ([new_volume, recon_volume[i], interp_slices[i]], dim=0)
        if with_labels:
            new_labels = torch.cat([recon_labels[i], interp_slice_labels[i]]) if new_labels is None else torch.cat \
                ([new_labels, recon_labels[i], interp_slice_labels[i]], dim=0)
    # add last slice
    new_volume = torch.cat([new_volume, recon_volume[i + 1]])
    new_labels = None if not with_labels else torch.cat([new_labels, recon_labels[i + 1]])
    new_volume = torch.clamp(new_volume, min=0, max=1.)

    return {'upsampled_image': new_volume, 'upsampled_labels': new_labels}


def latent_space_interp(alpha, trainer, img1, img2, device='cuda', with_labels=False) -> dict:

    # june 2022 with new pytorch version this seems to be required otherwise getting error:
    # RuntimeError: Input type (torch.cuda.DoubleTensor) and weight type (torch.cuda.FloatTensor) should be the same
    img1 = img1.float()
    img2 = img2.float()
    # latent vector of first image
    if not img1.is_cuda:
        img1 = img1.to(device)
    latent_1 = trainer.encode(img1, use_sr_model=True)

    # latent vector of second image
    if not img2.is_cuda:
        img2 = img2.to(device)
    latent_2 = trainer.encode(img2, use_sr_model=True)

    inter_latent = alpha * latent_1 + (1 - alpha) * latent_2

    # reconstruct interpolated image
    decode_result = trainer.decode(inter_latent, use_sr_model=True)
    if with_labels:
        inter_image = decode_result['image']
        inter_label = decode_result['pred_labels']
        inter_label = inter_label.detach().cpu()
    else:
        inter_image = decode_result
        inter_label = None
    inter_image = inter_image.detach().cpu()

    return {'inter_image': inter_image, 'inter_label': inter_label}


def sitk_to_torch(input_image: sitk.Image) -> torch.FloatTensor:
    np_img = sitk.GetArrayFromImage(input_image).astype(np.float32)
    if np_img.max() > 1 or np_img.min() < 0:
        np_img = normalize_img(np_img)

    img_tensor = torch.from_numpy(np_img).float()
    img_tensor = img_tensor.unsqueeze(dim=1)
    return img_tensor


def numpy_to_sitk(image_resolved: np.ndarray, input_image: sitk.Image, new_spacing: np.ndarray = None) -> sitk.Image:
    if image_resolved.ndim == 4:
        # 4d array
        volumes = [sitk.GetImageFromArray(image_resolved[v], False) for v in range(image_resolved.shape[0])]
        image_resolved = sitk.JoinSeries(volumes)
    else:
        image_resolved = sitk.GetImageFromArray(image_resolved)
    image_resolved.SetOrigin(input_image.GetOrigin())
    image_resolved.SetDirection(input_image.GetDirection())
    if new_spacing is not None:
        image_resolved.SetSpacing(new_spacing)
    else:
        image_resolved.SetSpacing(input_image.GetSpacing())
    return image_resolved


def normalize_img(img: np.ndarray, perc=(1, 99)) -> np.ndarray:
        min_val, max_val = np.percentile(img, perc)
        im = ((img.astype(img.dtype) - min_val) / (max_val - min_val)).clip(0, 1)
        return im


def load_images(input_dir: Path, suffix='.nii*') -> List[Tuple[Path, sitk.Image]]:
    file_list = [f for f in input_dir.rglob("*" + suffix)]
    if len(file_list) == 0:
        file_list = [f for f in input_dir.rglob("*.mha")]
        if len(file_list) == 0:
            file_list = [f for f in input_dir.rglob("*.mhd")]
        if len(file_list) == 0:
            raise FileNotFoundError("Error - no files found in {} with extensions nii, mha, mhd")
    images = []
    for fname in file_list:
        images.append((fname, sitk.ReadImage(str(fname))))

    return images


def save_images(images_hr: List[Tuple[Path, sitk.Image]]):
    global output_dir

    for (fname, sitk_img) in images_hr:
        sitk.WriteImage(sitk_img, str(fname))
        print("Save image HR {}".format(str(fname)))


def main():
    global trainer, input_images

    alpha_range = np.linspace(0, 1, args.num_interpolations + 2, endpoint=True)[1:-1]
    images_hr = []
    for (fname, sitk_img) in tqdm(input_images, desc='Process images'):
        num_frames = 1 if len(sitk_img.GetSize()) == 3 else sitk_img.GetSize()[-1]
        np_img_hr = None if num_frames == 1 else []

        for f_id in np.arange(num_frames):
            img = sitk_img if num_frames == 1 else sitk_img[:, :, :, int(f_id)]
            torch_image = sitk_to_torch(img)
            result_dict = create_super_volume(trainer, torch_image, alpha_range, use_original=True, labels=None)
            if num_frames == 1:
                np_img_hr = result_dict["upsampled_image"].detach().cpu().numpy().squeeze()
            else:
                np_img_hr.append(result_dict["upsampled_image"].detach().cpu().numpy().squeeze())
            if num_frames == 1 or (num_frames != 1 and f_id + 1 == num_frames):
                new_spacing_z = img.GetSpacing()[-1] / (args.num_interpolations + 1)
                new_spacing_z = (new_spacing_z,) if num_frames == 1 else (new_spacing_z, 1,)
                new_spacing = np.asarray(img.GetSpacing()[:2] + new_spacing_z).astype(np.float64)
                np_img_hr = np_img_hr if num_frames == 1 else np.stack(np_img_hr)
                sitk_img_hr = numpy_to_sitk(np_img_hr, sitk_img, new_spacing=new_spacing)
                images_hr.append((output_dir / fname.name, sitk_img_hr))
    return images_hr


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Find best SR model')

    parser.add_argument('--exper_dir', type=str, default=None)
    parser.add_argument('--model_nbr', type=int, default=None)
    parser.add_argument('--num_interpolations', type=int, default=6)
    parser.add_argument('--data_input_dir', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--save', action='store_true')

    args = parser.parse_args()
    if args.output_dir is None:
        args.exper_dir = args.exper_dir + os.sep + "ni0{}".format(args.num_interpolations)
    output_dir = Path(args.output_dir)
    if not output_dir.is_dir():
        output_dir.mkdir(parents=True)
    input_images = load_images(Path(args.data_input_dir))
    print("INFO - Found {} files to process in {}".format(len(input_images), args.data_input_dir))
    trainer, myargs = get_trainer_dynamic(src_path=args.exper_dir, model_nbr=args.model_nbr, model_nbr_sr=None,
                                          eval_mode=True)
    images_hr = main()
    if args.save:
        save_images(images_hr)