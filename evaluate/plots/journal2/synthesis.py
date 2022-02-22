import os
import matplotlib.pyplot as plt
from matplotlib import cm
from copy import deepcopy
import numpy as np
from collections import defaultdict
from mpl_toolkits.axes_grid.inset_locator import (InsetPosition,
                                                  mark_inset)
from evaluate.plots.journal2.evaluate_qualitatively import get_slices, get_diff

title_font_small = {'fontname': 'Monospace', 'size': '16', 'color': 'black', 'weight': 'normal'}
title_font_medium = {'fontname': 'Monospace', 'size': '20', 'color': 'black', 'weight': 'normal'}


def plot_inset(slice_image, ax):
    ax2 = plt.axes([0, 0, 1, 1])
    # Manually set the position and relative size of the inset axes within ax1
    ip = InsetPosition(ax, [0.4, 0.2, 0.5, 0.5])
    ax2.set_axes_locator(ip)
    # Mark the region corresponding to the inset axes on ax1 and draw lines
    # in grey linking the two axes.
    mark_inset(ax, ax2, loc1=2, loc2=1, fc="none", ec='0.5')
    ax2.imshow(slice_image[20:40, 20:40], cmap=cm.gray, interpolation='nearest', aspect='equal',
               vmin=0, vmax=1)
    ax2.axis("off")


def plot_interpol_sequence(image_dict, slice_info_dict, interpolation_steps, do_save=False, meth_name=None, height=5, width=15,
                           fig_dir=None, do_show=True, dpi=300, show_diff=False, dict_key='image'):
    if do_save and fig_dir is None:
        raise ValueError("Error - do_save is yes and fig_dir is None")
    for patid, slice_details in slice_info_dict.items():
        frame_id, slice_id, x_off, y_off, p_size = slice_details.values()
        if isinstance(p_size, tuple):
            x_p_size, y_p_size = p_size
        else:
            x_p_size, y_p_size = p_size, p_size
        if x_off is not None and y_off is not None and p_size is not None:
            slice_x, slice_y = slice(x_off, x_off + x_p_size, None), slice(y_off, y_off + y_p_size, None)
        else:
            slice_x, slice_y = None, None
        image = image_dict[patid][dict_key]

        for i in np.arange(interpolation_steps + 2):
            fig = plt.figure(figsize=(width, height))
            ax = plt.gca()
            if frame_id is None:
                slice_image = image[slice_id + i]
            else:
                slice_image = image[frame_id, slice_id + i]
                
            slice_image = slice_image if slice_x is None else slice_image[slice_x, slice_y]
            print("slice_image shape ", slice_image.shape)
            # print("slice_image shape ", slice_image.shape)
            if i == 0:
                first_image = deepcopy(slice_image)
                # print("WARNING - first slice : {}".format(slice_id + i))
            elif i == interpolation_steps - 1:
                last_image = deepcopy(slice_image)
                # print("WARNING - last slice : {}".format(slice_id + i))
            ax.imshow(slice_image, cmap=cm.gray, interpolation='nearest', aspect='equal',
                      vmin=0, vmax=1)
            ax.axis("off")
            fig.tight_layout(rect=[0.03, 0.03, 0.97, 0.97])
            if do_save:
                fig_dir = os.path.expanduser(fig_dir)
                str_patid = patid if isinstance(patid, str) else str(patid)
                pat_out_dir = os.path.join(fig_dir, str_patid)
                if do_save and not os.path.isdir(pat_out_dir):
                    os.makedirs(pat_out_dir)
                if dict_key == 'image':
                    if meth_name is not None:
                        fig_name = "p-{}_interpol_{}_s{}".format(str_patid, meth_name, slice_id + i) + ".png"
                    else:
                        fig_name = "p-{}_interpol_s{}".format(str_patid, slice_id + i) + ".png"
                else:
                    fig_name = "p-{}_ref_s{}".format(str_patid, slice_id + i) + ".png"
                fig_name = os.path.join(pat_out_dir, fig_name)
                plt.savefig(fig_name, bbox_inches='tight', pad_inches=0.0, dpi=dpi)
                print(("INFO - Successfully saved fig %s" % fig_name))
            if do_show:
                plt.show()
            plt.close()
        if show_diff:
            fig = plt.figure(figsize=(width, height))
            ax = plt.gca()
            ax.imshow(last_image - first_image, cmap=cm.bwr, vmin=-1, vmax=1, interpolation='nearest', aspect='equal')
            ax.axis("off")
            fig.tight_layout(rect=[0.03, 0.03, 0.97, 0.97])
            plt.show()
            plt.close()


def collect_slices(image_dict, slice_info_dict, axis=0, methods=None, downsample_steps=None, all_inbetween=False):

    slice_dict = {}
    for patid, slice_details in slice_info_dict.items():
        # 50: {'f':0, 's': 7, 'x': 70, 'y': 25, 'p_size': 100}
        # print(slice_details.values())
        frame_id, slice_id, x_off, y_off, p_size = slice_details.values()
        if axis == 0 and downsample_steps is not None:
            if slice_id in np.arange(image_dict['reference'][patid]['num_slices'])[::downsample_steps]:
                print("WARNING - !!!! - slice {} was reconstructed not synthesized".format(slice_id))
            slice_step_minus = slice_id % downsample_steps
            slice_below = slice_id - slice_step_minus
            slice_above = slice_id + (downsample_steps - slice_step_minus)
            if all_inbetween:
                slice_range_ref = np.arange(slice_below, slice_above + 1)
                slice_range_synth = np.arange(slice_below + 1, slice_above)
            else:
                slice_range_ref = [slice_below, slice_above]
                slice_range_synth = [slice_id]
        else:
            # non-axial slices
            slice_range_synth = [slice_id]
            slice_range_ref = None
        print("Collect slices - determined slice range / slice_range_ref: ", slice_range_synth, slice_range_ref)
        # !!!! Important: slice_id ARE SYNTHESIZED SLICES. But we want reconstructed here so
        # we add recon_idx which is +/- 1
        if x_off is not None and y_off is not None and p_size is not None:
            slice_x, slice_y = slice(x_off, x_off + p_size, None), slice(y_off, y_off + p_size, None)
        else:
            slice_x, slice_y = None, None
        for s_id in slice_range_synth:
            slice_dict[tuple((patid, s_id))], _ = get_slices(patid, s_id, image_dict, axis,
                                                              slice_x=slice_x, slice_y=slice_y, frame_id=frame_id,
                                                                transform=None, upsample=None, methods=methods)
        if axis == 0:
            for s_id in slice_range_ref:
                slice_dict[tuple((patid, s_id))], _ = get_slices(patid, s_id, image_dict, axis,
                                                                     slice_x=slice_x, slice_y=slice_y, frame_id=frame_id,
                                                                     transform=None, upsample=None, methods=methods)

    return slice_dict


def plot_separate_synthesis(images_dict, slice_info_dict, do_save=False, fig_name=None, height=5, width=15,
                                  methods=None, axis=0, fig_dir=None, do_show=True, dpi=300,
                                  downsample_steps=None, all_inbetween=False, verbose=False):
    if axis != 0:
        all_inbetween = False
    if not os.path.isdir(fig_dir):
        os.makedirs(fig_dir, exist_ok=False)
    rsme = defaultdict(dict)
    slice_dict = collect_slices(images_dict, slice_info_dict[axis], axis=axis, methods=methods,
                                downsample_steps=downsample_steps, all_inbetween=all_inbetween)

    for patid_sliceid, method_slice_dict in slice_dict.items():
        patid, slice_id = patid_sliceid
        str_patid = patid if isinstance(patid, str) else str(patid)
        pat_out_dir = os.path.join(os.path.join(fig_dir, str_patid), "axis{}".format(axis))
        if do_save and not os.path.isdir(pat_out_dir):
            os.makedirs(pat_out_dir)
        for meth, slice_image in method_slice_dict.items():
            fig = plt.figure(figsize=(width, height))
            ax = plt.gca()
            ax.imshow(slice_image, cmap=cm.gray, interpolation='nearest', aspect='equal',
                       vmin=0, vmax=1)
            ax.axis("off")
            if verbose:
                ax.set_title("Slice {} - method {}".format(slice_id, meth))
            fig.tight_layout(rect=[0.03, 0.03, 0.97, 0.97])
            if do_save:
                fig_name = "p-{}_s{}_{}_".format(str_patid, slice_id, meth) + "axis{}.png".format(axis)
                fig_name = os.path.join(pat_out_dir, fig_name)
                plt.savefig(fig_name, bbox_inches='tight', pad_inches=0.03, dpi=dpi)
                print(("INFO - Successfully saved fig %s" % fig_name))
            if do_show:
                plt.show()
            plt.close()
            if meth != "reference":
                diff, rsme[meth][slice_id] = get_diff(method_slice_dict['reference'], slice_image)
                fig = plt.figure(figsize=(width, height))
                ax = plt.gca()
                ax.imshow(diff, cmap=cm.bwr, vmin=-1, vmax=1, interpolation='nearest', aspect='equal')
                ax.xaxis.set_ticklabels([]), ax.yaxis.set_ticklabels([])
                ax.xaxis.set_ticks([]), ax.yaxis.set_ticks([])
                fig.tight_layout(rect=[0.03, 0.03, 0.97, 0.97])
                if do_save:
                    fig_name = "p-{}_s{}_{}_diff_".format(str_patid, slice_id, meth) + "axis{}.png".format(axis)
                    fig_name = os.path.join(pat_out_dir, fig_name)
                    plt.savefig(fig_name, bbox_inches='tight', pad_inches=0.03, dpi=dpi
                                )
                    print(("INFO - Successfully saved fig %s" % fig_name))
                if do_show:
                    plt.show()
                plt.close()
