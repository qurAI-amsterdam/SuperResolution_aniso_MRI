import torch
import numpy as np


def synthesize_cardiac_features(normalized_frame_id, orig_num_slices, spacing, downsample_steps=2):
    # we assume that the volume for which we create the features is downsampled. We first determine the range
    # of slice IDs we're dealing with after downsampling. We use original num slices to determine
    # normalized slice ID
    # IMPORTANT: normalized_frame_id = (frame_id + 1) / num_frames
    slice_range = np.arange(0, orig_num_slices)[::downsample_steps]

    slice_id_from, slice_id_to = (slice_range[:-1] + 1) / orig_num_slices, (slice_range[1:] + 1) / orig_num_slices
    frame_id = np.array([normalized_frame_id] * slice_id_from.shape[0])
    spacing = np.array([spacing] * slice_id_from.shape[0])
    num_slices = np.array([orig_num_slices] * slice_id_from.shape[0])
    features = np.concatenate((slice_id_from[:, None], slice_id_to[:, None], frame_id[:, None], spacing[:, None],
                           num_slices[:, None]), axis=1)
    # print("-------------- Features -----------------")
    # print(slice_range)
    # print(features)
    return features


def create_features(feature_dict, downsample_steps=2):
    """
        IMPORTANT: can only be used in case we generate in-between-slices for ACDC evaluation (cardiac)
        assume that num_slices is the downsampled num_slices.
        So, if original volume had 10 slices and we downsampled with factor 2 we obtain a stack
        [0, 2, 4, 6, 8] (#=5) so we miss slice 9! --> (10 - 1) % 2 != 0
        To create the normalized slice_ids we have to compute a) original number of slices and
        b) original slice IDs
        and then normalize them

        Important!!! we assume frame_id is already normalized!!!
    """

    if feature_dict['anatomy'] == "cardiac":
        return synthesize_cardiac_features(feature_dict['norm_frame_id'], feature_dict['orig_num_slices'],
                                           feature_dict['spacing'], downsample_steps=downsample_steps)
    else:
        raise NotImplementedError()


def latent_space_interp_pred_alpha(trainer, img1, img2,
                                   downsample_steps, device='cuda', is_eval=True,
                                   feature_dict=None) -> [torch.FloatTensor, np.ndarray]:

    torch.cuda.empty_cache()
    # latent vector of first image
    if not img1.is_cuda:
        img1 = img1.to(device)
    latent_1 = trainer.encode(img1, eval=is_eval)
    # latent vector of second image
    if not img2.is_cuda:
        img2 = img2.to(device)
    latent_2 = trainer.encode(img2, eval=is_eval)
    features = create_features(feature_dict, downsample_steps)
    features = torch.from_numpy(features).float().to(device)
    combined_latent = torch.cat([latent_1, latent_2], dim=1)
    alphas = trainer.predict_alpha(combined_latent, features, eval=is_eval)

    if trainer.num_alphas == 16:
        # alphas = torch.cuda.FloatTensor([0.5], device=torch.device('cuda:0')).expand_as(latent_1)
        # print("Warning - fixed alpha ", alphas.shape)
        # inter_latent = alphas * latent_1 + (1 - alphas) * latent_2
        inter_latent = alphas[:, :, None, None] * latent_1 + (1 - alphas[:, :, None, None]) * latent_2
        reshaped_alphas = alphas
    elif trainer.num_alphas == 32:
        latent_dim = int(trainer.args['latent'])
        inter_latent = alphas[:, :latent_dim, None, None] * latent_1 + alphas[:, latent_dim:, None, None] * latent_2
        reshaped_alphas = torch.cat((alphas[:, None, :latent_dim], alphas[:, None, latent_dim:]), dim=1)
    elif trainer.num_alphas == 256:
        alphas = alphas.view(latent_1.size(0), 1, latent_1.size(2), latent_1.size(3))
        inter_latent = alphas * latent_1 + (1 - alphas) * latent_2
        reshaped_alphas = torch.cat((alphas, (1 - alphas)), dim=1)
    else:
        inter_latent = alphas[:, 0, None, None, None] * latent_1 + alphas[:, 1, None, None, None] * latent_2
        reshaped_alphas = torch.cat((alphas[:, 0, None], alphas[:, 1, None]), dim=1)

    # reconstruct interpolated image
    if hasattr(trainer, 'decoder_mix'):
        inter_image = trainer.decode_syn_slice(inter_latent, eval=is_eval)
    else:
        inter_image = trainer.decode(inter_latent, eval=is_eval)
    inter_image = inter_image.detach().cpu()

    return inter_image, reshaped_alphas.detach().cpu().numpy()
