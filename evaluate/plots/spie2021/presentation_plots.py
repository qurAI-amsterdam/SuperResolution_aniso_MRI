import os
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from collections import defaultdict
from evaluate import get_slices

title_font_small = {'fontname': 'Monospace', 'size': '16', 'color': 'black', 'weight': 'normal'}
title_font_medium = {'fontname': 'Monospace', 'size': '20', 'color': 'black', 'weight': 'normal'}


def plot_references(img_slice, meth, patid, frame_id, slice_id, width=5, height=5, fig_dir=None, do_save=False,
                    do_show=True):
    fig = plt.figure(figsize=(width, height))
    plt.imshow(img_slice, cmap=cm.gray, vmin=0, vmax=1, interpolation="nearest")
    plt.axis("off")
    fig.tight_layout(rect=[0.03, 0.03, 0.97, 0.97])
    if do_save:
        if frame_id is not None:
            fig_file_name = os.path.join(fig_dir, meth + "_sax_p{}_f{}_s{}.png".format(patid,
                                                                                                frame_id,
                                                                                                slice_id))
        else:
            fig_file_name = os.path.join(fig_dir, meth + "_sax_p{}_s{}.png".format(patid,
                                                                                            slice_id))
        plt.savefig(fig_file_name, bbox_inches='tight')
        print("Saved to {}".format(fig_file_name))
    if do_show:
        plt.show()
    plt.close()


def plot_synth_plus_diff(img_slice, img_diff, meth, patid, frame_id, slice_id, width=5, height=5, fig_dir=None, do_save=False,
                    do_show=True):
    fig = plt.figure(figsize=(width, height * 2))
    ax = fig.subplots(2)
    ax[0].imshow(img_slice, cmap=cm.gray, vmin=0, vmax=1, interpolation="nearest")
    ax[0].axis("off")
    ax[1].imshow(img_diff, cmap=cm.bwr, vmin=-0.5, vmax=0.5, interpolation="nearest")
    ax[1].axis("off")
    fig.tight_layout(rect=[0.03, 0.03, 0.97, 0.97])
    if do_save:
        if frame_id is not None:
            fig_file_name = os.path.join(fig_dir, meth + "_sax_diff_p{}_f{}_s{}.png".format(patid,
                                                                                                frame_id,
                                                                                                slice_id))
        else:
            fig_file_name = os.path.join(fig_dir, meth + "_sax_diff_p{}_s{}.png".format(patid,
                                                                                            slice_id))
        plt.savefig(fig_file_name, bbox_inches='tight')
        print("Saved to {}".format(fig_file_name))
    if do_show:
        plt.show()
    plt.close()


def compare_synthesized_slices(images_dict, patid, slice_id, width=5, height=5, do_save=False, fig_name=None,
                           methods=None, slice_x=None, slice_y=None, axis=0,
                           fig_dir="/home/jorg/images/", do_show=True, frame_id=None, transform=None,
                           upsample=None):
    if not os.path.isdir(fig_dir):
        os.makedirs(fig_dir, exist_ok=False)
    if methods is None:
        methods = list(images_dict.keys())
    print("Do show {}".format(do_show))
    rsme = defaultdict(dict)
    slice_dict = {}
    slice_dict[-1], plane = get_slices(patid, slice_id-1, images_dict, axis,
                                   slice_x=slice_x, slice_y=slice_y, frame_id=frame_id,
                                   transform=transform, upsample=upsample)
    slice_dict[0], plane = get_slices(patid, slice_id, images_dict, axis,
                                   slice_x=slice_x, slice_y=slice_y, frame_id=frame_id,
                                   transform=transform, upsample=upsample)
    slice_dict[1], plane = get_slices(patid, slice_id+1, images_dict, axis,
                                   slice_x=slice_x, slice_y=slice_y, frame_id=frame_id,
                                   transform=transform, upsample=upsample)

    for i, meth in enumerate(methods):
        if meth == 'reference':
            for i in np.arange(-1, 2, 1):
                img_slice = slice_dict[i][meth]
                print("Reference slice {}".format(slice_id - i))
                plot_references(img_slice, meth, patid, frame_id, slice_id - i, width=width, height=height,
                                fig_dir=fig_dir, do_save=do_save, do_show=do_show)
        else:
            # plot synthesized image only + difference image
            img_slice = slice_dict[0][meth]
            img_diff = slice_dict[0]['reference'] - img_slice
            print("{} slice {}".format(meth, slice_id))
            plot_synth_plus_diff(img_slice, img_diff, meth, patid, frame_id, slice_id, width=width, height=height,
                                 fig_dir=fig_dir, do_save=do_save, do_show=do_show)