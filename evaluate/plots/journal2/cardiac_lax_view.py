import os
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from torchvision import transforms
from datasets.shared_transforms import CenterCrop, AdjustToPatchSize


TRANSFORM224 = transforms.Compose([AdjustToPatchSize(tuple((224, 224))),
                                           CenterCrop(224)])


def generate_lax_views(img4d_ref, img4d_caisr, img4d_linear, img4d_bspline, img4d_lanczos, do_save = False):

    fig_dir = os.path.expanduser("~/expers/sr/ACDC/results/lax/")
    frame_id, slice_id = 4, 110
    slice_x, slice_y = slice(50, 180, None), slice(0, 224, None)
    # slice_x, slice_y = slice(90, 150, None), slice(40, 200, None)
    slice_y_ref = slice(1 + slice_y.start // 10, slice_y.stop // 10)
    fname = "patient16_lax_f{}_s{}.png".format(frame_id, slice_id)
    # cross_slice_1 = img4d_ae[frame_id, slice_y, slice_id, slice_x]
    # print(cross_slice_1.shape)
    # ysize, xsize = cross_slice_1.shape
    # extent_sr = (0, xsize * 1.4, 0, ysize * 1)
    # fig = plt.figure(figsize=(6, 4))
    # plt.imshow(cross_slice_1, cmap=cm.gray, vmin=0, vmax=1, interpolation="nearest",
    #            extent=extent_sr)
    # plt.axis("off")
    # fig.tight_layout(rect=[0.03, 0.03, 0.97, 0.97])
    # if do_save:
    #     plt.savefig(os.path.join(fig_dir, "ae_" + fname), bbox_inches='tight', pad_inches=0)
    # plt.show()
    cross_slice_1_caisr = img4d_caisr[frame_id, slice_y, slice_id, slice_x]
    print("caisr ", cross_slice_1_caisr.shape)
    ysize, xsize = cross_slice_1_caisr.shape
    extent_sr = (0, xsize * 1.4, 0, ysize * 1)
    fig = plt.figure(figsize=(6, 4))
    plt.imshow(cross_slice_1_caisr, cmap=cm.gray, vmin=0, vmax=1, interpolation="nearest",
               extent=extent_sr)
    plt.axis("off")
    fig.tight_layout(rect=[0.03, 0.03, 0.97, 0.97])
    if do_save:
        plt.savefig(os.path.join(fig_dir, "caisr_" + fname), bbox_inches='tight', pad_inches=0)
    plt.show()

    conv_offset = 10
    cross_slice_l = img4d_linear[frame_id, conv_offset:, slice_id, slice_x]
    cross_slice_l = cross_slice_l[slice_y]
    print("Linear ", cross_slice_l.shape)
    ysize, xsize = cross_slice_l.shape
    extent_sr = (0, xsize * 1.4, 0, ysize * 1)
    fig = plt.figure(figsize=(6, 4))
    plt.imshow(cross_slice_l, cmap=cm.gray, vmin=0, vmax=1, interpolation="nearest",
               extent=extent_sr)
    plt.axis("off")
    fig.tight_layout(rect=[0.03, 0.03, 0.97, 0.97])
    if do_save:
        plt.savefig(os.path.join(fig_dir, "linear_" + fname), bbox_inches='tight',
                    pad_inches=0)
    plt.show()

    cross_slice_bsp = img4d_bspline[frame_id, conv_offset:, slice_id, slice_x]
    cross_slice_bsp = cross_slice_bsp[slice_y]
    print("B-spline ", cross_slice_bsp.shape)
    ysize, xsize = cross_slice_bsp.shape
    extent_sr = (0, xsize * 1.4, 0, ysize * 1)
    fig = plt.figure(figsize=(6, 4))
    plt.imshow(cross_slice_bsp, cmap=cm.gray, vmin=0, vmax=1, interpolation="nearest",
               extent=extent_sr)
    plt.axis("off")
    fig.tight_layout(rect=[0.03, 0.03, 0.97, 0.97])
    if do_save:
        plt.savefig(os.path.join(fig_dir, "bspline_" + fname), bbox_inches='tight',
                    pad_inches=0)
    plt.show()

    cross_slice_lcz = img4d_lanczos[frame_id, conv_offset:, slice_id, slice_x]
    cross_slice_lcz = cross_slice_lcz[slice_y]
    print("Lanczos ", cross_slice_lcz.shape)
    ysize, xsize = cross_slice_lcz.shape
    extent_sr = (0, xsize * 1.4, 0, ysize * 1)
    fig = plt.figure(figsize=(6, 4))
    plt.imshow(cross_slice_lcz, cmap=cm.gray, vmin=0, vmax=1, interpolation="nearest",
               extent=extent_sr)
    plt.axis("off")
    fig.tight_layout(rect=[0.03, 0.03, 0.97, 0.97])
    if do_save:
        plt.savefig(os.path.join(fig_dir, "lanczos_" + fname), bbox_inches='tight',
                    pad_inches=0)
    plt.show()

    img3d_ref = TRANSFORM224({'image': img4d_ref[frame_id]})['image']
    img3d_ref = (img3d_ref - np.min(img3d_ref)) / (np.max(img3d_ref) - np.min(img3d_ref))
    cross_slice_ref = img3d_ref[slice_y_ref, slice_id, slice_x]
    print(cross_slice_ref.shape)
    ysize, xsize = cross_slice_ref.shape
    extent = (0, xsize * 1.4, 0, ysize * 10)
    print("Extents ", extent_sr, extent)
    fig = plt.figure(figsize=(6, 4))
    plt.imshow(cross_slice_ref, cmap=cm.gray, vmin=0, vmax=1, extent=extent, interpolation="nearest")
    plt.axis("off")
    fig.tight_layout(rect=[0.03, 0.03, 0.97, 0.97])
    if do_save:
        plt.savefig(os.path.join(fig_dir, "ref_" + fname), bbox_inches='tight',
                    pad_inches=0)
    plt.show()