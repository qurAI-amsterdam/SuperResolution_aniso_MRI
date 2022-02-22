import os
import matplotlib.pyplot as plt
from matplotlib import cm
from copy import deepcopy
from collections import defaultdict
from evaluate.plots.journal2.metric_boxplots import METHOD_LABELS

from evaluate.plots.journal2.evaluate_qualitatively import get_slices, get_diff

title_font_small = {'fontname': 'Monospace', 'size': '16', 'color': 'black', 'weight': 'normal'}
title_font_medium = {'fontname': 'Monospace', 'size': '20', 'color': 'black', 'weight': 'normal'}


def collect_slices(image_dict, slice_info_dict, recon_idx=1, axis=0, methods=None):
    slice_dict = {}
    for patid, slice_details in slice_info_dict.items():
        # 50: {'f':0, 's': 7, 'x': 70, 'y': 25, 'p_size': 100}
        frame_id, slice_id, x_off, y_off, p_size = slice_details.values()
        # !!!! Important: slice_id ARE SYNTHESIZED SLICES. But we want reconstructed here so
        # we add recon_idx which is +/- 1
        slice_id += recon_idx
        slice_x, slice_y = slice(x_off, x_off + p_size, None), slice(y_off, y_off + p_size, None)
        slice_dict[tuple((patid, slice_id))], _ = get_slices(patid, slice_id, image_dict, axis,
                                                          slice_x=slice_x, slice_y=slice_y, frame_id=frame_id,
                                                            transform=None, upsample=None, methods=methods)
    return slice_dict


def plot_separate_reconstructions(images_dict, slice_info_dict, do_save=False, fig_name=None, height=5, width=15,
                                  methods=None, axis=0, recon_idx=1, fig_dir=None, do_show=True, dpi=300,
                                  ):

    if not os.path.isdir(fig_dir):
        os.makedirs(fig_dir, exist_ok=False)
    rsme = defaultdict(dict)
    slice_dict = collect_slices(images_dict, slice_info_dict, recon_idx, axis=axis, methods=methods)

    for patid_sliceid, method_slice_dict in slice_dict.items():
        patid, slice_id = patid_sliceid
        str_patid = patid if isinstance(patid, str) else str(patid)
        pat_out_dir = os.path.join(fig_dir, str_patid)
        if do_save and not os.path.isdir(pat_out_dir):
            os.makedirs(pat_out_dir)
        for meth, slice_image in method_slice_dict.items():
            fig = plt.figure(figsize=(width, height))
            ax = plt.gca()
            ax.imshow(slice_image, cmap=cm.gray, interpolation='nearest', aspect='equal',
                       vmin=0, vmax=1)
            ax.axis("off")
            fig.tight_layout(rect=[0.03, 0.03, 0.97, 0.97])
            if do_save:
                fig_name = "p-{}_s{}_{}_recon_".format(str_patid, slice_id, meth) + "axis{}.png".format(axis)
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
                ax.imshow(diff, cmap=cm.bwr, vmin=-0.5, vmax=0.5, interpolation='nearest', aspect='equal')
                # ax.axis("off")
                ax.xaxis.set_ticklabels([]), ax.yaxis.set_ticklabels([])
                ax.xaxis.set_ticks([]), ax.yaxis.set_ticks([])
                fig.tight_layout(rect=[0.03, 0.03, 0.97, 0.97])
                if do_save:
                    fig_name = "p-{}_s{}_{}_diff_".format(str_patid, slice_id, meth) + "axis{}.png".format(axis)
                    fig_name = os.path.join(pat_out_dir, fig_name)
                    plt.savefig(fig_name, bbox_inches='tight', dpi=dpi, pad_inches=0.03
                                )
                    print(("INFO - Successfully saved fig %s" % fig_name))
                if do_show:
                    plt.show()
                plt.close()


def get_fig_row(meth):
    if meth == "reference":
        row = 4
    elif meth == "ae":
        row = 2
    elif meth == "caisr":
        row = 6
    else:
        raise ValueError("Error - get_fig_row - unknown {} method".format(meth))
    return row


def filter_image_dict(images_dict, methods):
    m_keys = deepcopy(list(images_dict.keys()))
    for meth in m_keys:
        if meth not in methods:
            del images_dict[meth]
    return images_dict


def plot_grid_reconstructions(images_dict, slice_info_dict, do_save=False, fig_name=None, height=5, width=15,
                           methods=None, slice_x=None, slice_y=None, axis=0, recon_idx=1,
                           fig_dir=None, do_show=True, downsample_steps=None):

    if not os.path.isdir(fig_dir):
        os.makedirs(fig_dir, exist_ok=False)
    if methods is not None:
        images_dict = filter_image_dict(images_dict, methods)
    col_span, row_span = 2, 2
    # columns we need for plot.
    num_columns = len(slice_info_dict) * col_span
    num_rows = 5 * row_span
    # width = column_width * num_columns
    # height = int(width * ((num_rows // row_span) / (num_columns // col_span)))
    rsme = defaultdict(dict)
    print("Rows/Columns {}/{}   Width/Height {}/{}".format(num_rows, num_columns, width, height))
    fig = plt.figure(figsize=(width, height))
    col = 0
    slice_dict = collect_slices(images_dict, slice_info_dict, recon_idx, axis=axis)
    for pat_slice_id, method_slice_dict in slice_dict.items():
        patid, slice_id = pat_slice_id
        for meth, slice_image in method_slice_dict.items():
            row = get_fig_row(meth)
            ax1 = plt.subplot2grid((num_rows, num_columns), (row, col), rowspan=2, colspan=2)
            ax1.imshow(slice_image, cmap=cm.gray, interpolation='nearest', aspect='equal',
                       vmin=0, vmax=1)
            if col == 0:
                ax1.set_ylabel("{}".format(METHOD_LABELS[meth]), **title_font_medium)
                ax1.set_yticks([])
                ax1.set_xticks([])
            else:
                ax1.axis("off")
            if meth != "reference":
                diff, rsme[meth][slice_id] = get_diff(method_slice_dict['reference'], slice_image)
                diff_row = 0 if meth == "ae" else 8
                ax1 = plt.subplot2grid((num_rows, num_columns), (diff_row, col), rowspan=2, colspan=2)
                ax1.imshow(diff, cmap=cm.bwr, vmin=-0.5, vmax=0.5, interpolation='nearest', aspect='equal')
                ax1.xaxis.set_ticklabels([]), ax1.yaxis.set_ticklabels([])
                ax1.xaxis.set_ticks([]), ax1.yaxis.set_ticks([])
                if meth == "caisr":
                    ax1.set_xlabel("RMSE={:.4f}".format(rsme[meth][slice_id]), **title_font_small)
                else:
                    ax1.set_title("RMSE={:.4f}".format(rsme[meth][slice_id]), **title_font_small)

        col += col_span
    fig.tight_layout(rect=[0.03, 0.03, 0.97, 0.97])

    if fig_name is None:
        fig_name = "reconstructions_" + "axis{}.png".format(axis)
    fig_name = os.path.join(fig_dir, fig_name)
    if do_save:
        plt.savefig(fig_name, bbox_inches='tight', pad_inches=0)
        print(("INFO - Successfully saved fig %s" % fig_name))
    if do_show:
        plt.show()
    plt.close()