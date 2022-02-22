import os
import torch
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from collections import defaultdict
from evaluate.plots.journal2.metric_boxplots import METHOD_LABELS

title_font_small = {'fontname': 'Monospace', 'size': '16', 'color': 'black', 'weight': 'normal'}
title_font_medium = {'fontname': 'Monospace', 'size': '20', 'color': 'black', 'weight': 'normal'}


def plot_reference_coronal_sagittal(slice_dict, num_rows=2, num_columns=8):
    ax1 = plt.subplot2grid((num_rows, num_columns), (0, 0), rowspan=2, colspan=2)
    ax1.imshow(slice_dict[0]['reference'], cmap=cm.gray, interpolation='nearest', aspect='equal',
               vmin=0, vmax=1)
    ax1.set_title("Original", **title_font_small)
    ax1.axis("off")


def plot_reference_images(slice_dict, slice_id_in_between, downsample_steps, num_rows=6, num_columns=8):

    ax1 = plt.subplot2grid((num_rows, num_columns), (0, 0), rowspan=2, colspan=2)
    ax1.imshow(slice_dict[-1]['reference'], cmap=cm.gray, interpolation='nearest', aspect='equal',
               vmin=0, vmax=1)
    ax1.set_ylabel("Original\nslice 1", **title_font_small)
    ax1.set_yticks([])
    ax1.set_xticks([])
    # ax1.axis("off")
    ax1 = plt.subplot2grid((num_rows, num_columns), (2, 0), rowspan=2, colspan=2)
    ax1.imshow(slice_dict[0]['reference'], cmap=cm.gray, interpolation='nearest', aspect='equal',
               vmin=0, vmax=1)
    ax1.set_ylabel("Original\nslice {}".format(slice_id_in_between), **title_font_small)
    ax1.set_yticks([])
    ax1.set_xticks([])
    # ax1.axis("off")
    ax1 = plt.subplot2grid((num_rows, num_columns), (4, 0), rowspan=2, colspan=2)
    ax1.imshow(slice_dict[1]['reference'], cmap=cm.gray, interpolation='nearest', aspect='equal',
               vmin=0, vmax=1)
    ax1.set_ylabel("Original\nslice {}".format(1 + downsample_steps), **title_font_small)
    ax1.set_yticks([])
    ax1.set_xticks([])
    # ax1.axis("off")


def determine_slice_step(num_slices, downsample_steps):
    slices_in_lr_volume = np.arange(num_slices)[:: downsample_steps]


def compare_methods_slices(images_dict, patid, slice_id, do_save=False, fig_name=None, height=5, width=15,
                           methods=None, slice_x=None, slice_y=None, axis=0,
                           fig_dir=None, do_show=True, downsample_steps=None,
                           exper_tag=None, frame_id=None, transform=None,
                           upsample=None, file_suffix=".png"):
    print(images_dict.keys())
    if not os.path.isdir(fig_dir):
        os.makedirs(fig_dir, exist_ok=False)
    if methods is None:
        methods = list(images_dict.keys())
    col_span, row_span = 2, 2
    num_columns = (len(methods) + 1) * col_span
    num_rows = 3 * row_span if axis == 0 else 2 * row_span
    # width = column_width * num_columns
    # height = int(width * ((num_rows // row_span) / (num_columns // col_span)))
    rsme = defaultdict(dict)
    print("Rows/Columns {}/{}   Width/Height {}/{}".format(num_rows, num_columns, width, height))
    fig = plt.figure(figsize=(width, height))
    # One row per method:
    slice_dict = {}
    if axis == 0 and downsample_steps is not None:
        if slice_id in np.arange(images_dict['reference'][patid]['num_slices'])[::downsample_steps]:
            print("WARNING - !!!! - slice {} was reconstructed not synthesized".format(slice_id))
        slice_step_minus = slice_id % downsample_steps
        slice_below = slice_id - slice_step_minus
        slice_above = slice_id + (downsample_steps - slice_step_minus)
        print("WARNING - slice below {} - {} - slice above {}".format(slice_below, slice_id, slice_above))
    else:
        slice_step_minus, slice_step_plus = 1, 1
        slice_below, slice_above = slice_id - slice_step_minus, slice_id + slice_step_plus

    slice_dict[-1], plane = get_slices(patid, slice_below, images_dict, axis,
                                       slice_x=slice_x, slice_y=slice_y, frame_id=frame_id,
                                       transform=transform, upsample=upsample)
    slice_dict[0], plane = get_slices(patid, slice_id, images_dict, axis,
                                      slice_x=slice_x, slice_y=slice_y, frame_id=frame_id,
                                      transform=transform, upsample=upsample)
    slice_dict[1], plane = get_slices(patid, slice_above, images_dict, axis,
                                      slice_x=slice_x, slice_y=slice_y, frame_id=frame_id,
                                      transform=transform, upsample=upsample)
    # row1/column1: Original, row2: method, row3, diff image
    if axis == 0:
        row = 2
        col = 2
        in_between_slice_id = 1 + slice_id - slice_below
        plot_reference_images(slice_dict, in_between_slice_id, downsample_steps, num_rows, num_columns)
    else:
        row = 0
        col = 2
        plot_reference_coronal_sagittal(slice_dict, num_rows, num_columns)
    for i, meth in enumerate(methods):
        ax1 = plt.subplot2grid((num_rows, num_columns), (row, col), rowspan=2, colspan=2)
        ax1.imshow(slice_dict[0][meth], cmap=cm.gray, interpolation='nearest', aspect='equal',
                   vmin=0, vmax=1)
        if axis == 0 and meth in ['ae', 'caisr']:
            mytitle = r'{} ($\alpha={:.2f}$)'.format(METHOD_LABELS[meth], (in_between_slice_id - 1) / downsample_steps)
        else:
            mytitle = METHOD_LABELS[meth]
        ax1.set_title("{}".format(mytitle), **title_font_small)
        ax2 = plt.subplot2grid((num_rows, num_columns), (row+2, col), rowspan=2, colspan=2)
        diff, rsme[meth][slice_id] = get_diff(slice_dict[0]['reference'], slice_dict[0][methods[i]])
        ax2.imshow(diff, cmap=cm.bwr, vmin=-0.5, vmax=0.5, interpolation='nearest', aspect='equal')
        ax1.axis("off")
        ax2.xaxis.set_ticklabels([]), ax2.yaxis.set_ticklabels([])
        ax2.xaxis.set_ticks([]), ax2.yaxis.set_ticks([])
        ax2.set_xlabel("RMSE={:.4f}".format(rsme[meth][slice_id]), **title_font_small)
        col += 2

    fig.tight_layout(rect=[0.03, 0.03, 0.97, 0.97])

    if fig_name is None:
        if exper_tag is None:
            exper_tag = "compare"
        fig_name = str(patid) + "_" + plane + "_s" + str(slice_id) + "_" + exper_tag + file_suffix
    fig_name = os.path.join(fig_dir, fig_name)
    if do_save:
        plt.savefig(fig_name, bbox_inches='tight')
        print(("INFO - Successfully saved fig %s" % fig_name))
    if do_show:
        plt.show()
    plt.close()
    print(rsme)


def plot_interpol_example(interpol_grid, orig_slices, width=14, height=8, do_save=False, do_show=True,
                          fig_dir=None, patid=None, frame_id=None, slice_id=None):
    """
        interpol_grid: numpy array with shape [interpol + 2 (slices), y, x]
        created with function interpol_2 from kwatsch.acai_utils
        e.g.: in case we generate 3 additional slices between two slices:
                interpol_grid [5, 128, 128]
                orig_slices [3, 128, 128]
              Assuming that we interpolate between 2 slices where one is left out (the in-between one)
              e.g. slice 1 and 3. Then orig slice contains slice 1,2,3.

        orig_slices:
    """
    fig_dir = os.path.expanduser(fig_dir)
    if not os.path.isdir(fig_dir):
        os.makedirs(fig_dir, exist_ok=False)
    fig = plt.figure(figsize=(width, height))
    for i in range(interpol_grid.shape[0]):
        ax1 = plt.subplot2grid((6, 6), (2, i * 2), rowspan=2, colspan=2)
        ax1.xaxis.set_ticklabels([]), ax1.yaxis.set_ticklabels([])
        ax1.xaxis.set_ticks([]), ax1.yaxis.set_ticks([])
        if i == 0 or i == 2:
            ax1.set_title(r"Reconstructed", ha='center', **title_font_small)
        else:
            ax1.set_title(r'Synthesized', ha='center',
                          **title_font_small)
        ax1.imshow(interpol_grid[i], cmap=cm.gray, aspect='equal', interpolation='nearest')

    ax1 = plt.subplot2grid((6, 6), (0, 0), rowspan=2, colspan=2)
    ax1.imshow(orig_slices[0], cmap=cm.gray, interpolation='nearest', aspect='equal', vmin=0, vmax=1)
    ax1.set_title("Original" "\n" "slice 1", ha='center', **title_font_small)
    ax1.xaxis.set_ticklabels([]), ax1.yaxis.set_ticklabels([])
    ax1.xaxis.set_ticks([]), ax1.yaxis.set_ticks([])
    ax1 = plt.subplot2grid((6, 6), (0, 2), rowspan=2, colspan=2)
    ax1.imshow(orig_slices[1], cmap=cm.gray, interpolation='nearest', aspect='equal', vmin=0, vmax=1)
    ax1.set_title("Original" "\n" "slice 2", ha='center', **title_font_small)
    ax1.xaxis.set_ticklabels([]), ax1.yaxis.set_ticklabels([])
    ax1.xaxis.set_ticks([]), ax1.yaxis.set_ticks([])
    ax1 = plt.subplot2grid((6, 6), (0, 4), rowspan=2, colspan=2)
    ax1.imshow(orig_slices[2], cmap=cm.gray, interpolation='nearest', aspect='equal', vmin=0, vmax=1)
    ax1.set_title("Original" "\n" "slice 3", ha='center', **title_font_small)
    ax1.xaxis.set_ticklabels([]), ax1.yaxis.set_ticklabels([])
    ax1.xaxis.set_ticks([]), ax1.yaxis.set_ticks([])
    # Diff images
    vmin, vmax = -0.7, 0.7
    ax1 = plt.subplot2grid((6, 6), (4, 0), rowspan=2, colspan=2)
    diff1, rmse = get_diff(orig_slices[0], interpol_grid[0])
    ax1.xaxis.set_ticklabels([]), ax1.yaxis.set_ticklabels([])
    ax1.xaxis.set_ticks([]), ax1.yaxis.set_ticks([])
    ax1.imshow(diff1, cmap=cm.bwr, vmin=vmin, vmax=vmax, interpolation='nearest',
               aspect='equal')
    ax1 = plt.subplot2grid((6, 6), (4, 2), rowspan=2, colspan=2)
    diff2, rmse = get_diff(orig_slices[1], interpol_grid[1])
    ax1.xaxis.set_ticklabels([]), ax1.yaxis.set_ticklabels([])
    ax1.xaxis.set_ticks([]), ax1.yaxis.set_ticks([])

    ax1.imshow(diff2, cmap=cm.bwr, vmin=vmin, vmax=vmax, interpolation='nearest',
               aspect='equal')
    # ax1.axis("off")
    ax1 = plt.subplot2grid((6, 6), (4, 4), rowspan=2, colspan=2)
    diff3, rmse = get_diff(orig_slices[2], interpol_grid[2])  # was [2] and [4]
    ax1.xaxis.set_ticklabels([]), ax1.yaxis.set_ticklabels([])
    ax1.xaxis.set_ticks([]), ax1.yaxis.set_ticks([])
    ax1.imshow(diff3, cmap=cm.bwr, vmin=vmin, vmax=vmax, interpolation='nearest',
               aspect='equal')
    fig.tight_layout(rect=[0.03, 0.03, 0.97, 0.97])
    if do_save:
        if frame_id is None:
            fig_name = "patient{}_s{}.png".format(patid, slice_id)
        else:
            fig_name = "patient{}_f{}_s{}.png".format(patid, frame_id, slice_id)
        fig_name = os.path.join(fig_dir, fig_name)
        plt.savefig(fig_name, bbox_inches='tight')
        print("INFO - Saved to {}".format(fig_name))
    if do_show:
        plt.show()
    plt.close()


# plot_interpol_example(img_grid[:, slice_x, slice_y], img_slices[:, slice_x, slice_y])

def get_diff_images(ref, recon):
    return ref - recon


def get_diff(ref, synthesized):

    diff = ref - synthesized
    rmse = np.sqrt(np.mean(diff**2))
    return diff, rmse


def apply_slicexy(aslice, slice_x, slice_y):
    if slice_x is not None:
        aslice = aslice[slice_x]
    if slice_y is not None:
        aslice = aslice[:, slice_y]
    return aslice


def do_upsample(slice_2d, up_func):
    slice_2d = torch.from_numpy(slice_2d[None, None])
    return up_func(slice_2d).squeeze().numpy()


def get_slices(pid, slice_id, images_dict, axis=0, methods=None,
               slice_x=None, slice_y=None, frame_id=None, transform=None, upsample=None):
    slice_dict = {}
    if axis == 0:
        # cardiac MRI has 4dim, time dim. We're only considering CMRIs for axial direction
        if frame_id is not None:
            myslice = np.s_[frame_id, slice_id]
        else:
            myslice = np.s_[slice_id]
        plane = 'axial'
    elif axis == 1:
        myslice = np.s_[:, slice_id]
        plane = 'coronal'
    elif axis == 2:
        myslice = np.s_[:, :, slice_id]
        plane = 'saggital'

    for m in images_dict.keys():
        if methods is not None and m not in methods:
            continue
        if transform is not None:
            images_dict[m][pid] = transform(images_dict[m][pid])
        if m == 'reference' and 'image_hr' in images_dict[m][pid].keys():
            print("!!!!! get_slices - Getting reference image ")
            s = images_dict[m][pid]['image_hr'][myslice]
        else:
            s = images_dict[m][pid]['image'][myslice]
        s = apply_slicexy(s, slice_x, slice_y)
        if upsample:
            s = do_upsample(s, upsample)
        if axis == 1 or axis == 2:
            s = np.rot90(s, 2)
        slice_dict[m] = s
    return slice_dict, plane


def interpolate_between_slices(trainer, z, num_interpol=3, side=None, dim=None):
    if side is None:
        side = z.shape[0] // 2
    # these are not interpolated but just reconstructions. so don't add: use_sr_model=True
    x = trainer.decode(torch.FloatTensor(z).to('cuda'))

    # print("interpolate_2 !!!")
    a, b = z[:side], z[-side:]
    if dim is not None:
        z_interp = np.vstack([a] * num_interpol)
        z_mix = [a[:, dim:dim + 8] * (1 - t) + b[:, dim:dim + 8] * t for t in
                 np.linspace(0., 1., num_interpol + 2)[1:-1]]
        z_mix = np.vstack(z_mix)
        z_interp[:, dim:dim + 8] = z_mix
    else:
        z_interp = [a * (1 - t) + b * t for t in np.linspace(0., 1., num_interpol + 2)[1:-1]]
        z_interp = np.vstack(z_interp)
    # print("interpolate_2 ", a.shape, b.shape, z_interp.shape)
    x_interp = trainer.decode(torch.FloatTensor(z_interp).to('cuda'), use_sr_model=True)
    x_interp = x_interp.cpu().data.numpy()
    x_fixed = x.data.cpu().numpy()
    all = []
    all.extend(x_fixed[:side])
    all.extend(x_interp)
    all.extend(x_fixed[-side:])

    return all