import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, FuncFormatter
from matplotlib import cm
from collections import defaultdict
from evaluate.plots.journal2.metric_boxplots import METHOD_LABELS, MODEL_COLORS

title_font_small = {'size': '36', 'color': 'black', 'weight': 'normal'}  # 'fontname': 'Monospace',


def get_conventional_results(root_dir, upsample_factors, metric_str, eval_axis):
    # root dir e.g. : ~/expers/sr_redo/OASIS/
    # upsample_factors: list of factors [2, 3, 4, 5, 6]
    methods = ['linear', 'bspline', 'lanczos']
    result_dict = {}
    root_dir = os.path.expanduser(root_dir)
    print("INFO - loading conventional-method results from {}".format(root_dir))
    for m in methods:
        m_dir = os.path.join(root_dir, 'conventional' + os.sep + m + os.sep + 'results')
        y_means, y_errors = [], []
        for factor in upsample_factors:
            factor = int(factor)
            f = '{}_{}x.npz'.format(m, factor) if eval_axis == 0 else '{}_{}x_axis{}.npz'.format(m, factor, eval_axis)
            f = os.path.join(m_dir, f)
            if os.path.isfile(f):
                res_d = np.load(f)
                y_means.append(round(res_d[metric_str + "_res"][0], 3))
                y_errors.append(round(res_d[metric_str + "_res"][1], 3))
            else:
                print("WARNING - conventional result {} NOT FOUND".format(f))
        result_dict[m] = {'mean': y_means, 'std': y_errors}
    return result_dict


def plot_conv_results(conv_results, p_axis, x_labels):
    for meth, res_dict in conv_results.items():
        meth_lbl = METHOD_LABELS[meth]
        m_color, alpha = MODEL_COLORS[meth][0], MODEL_COLORS[meth][1]
        p_axis.plot(x_labels, res_dict['mean'], c='blue', alpha=alpha)
        p_axis.errorbar(x_labels, res_dict['mean'], c='blue', label=meth_lbl, yerr=res_dict['std'], fmt='o',
                        solid_capstyle='projecting', capsize=15, markersize=20, alpha=alpha)


def upsample_compare(res_dicts, eval_axis, height=5, width=7, filename=None, do_show=True, do_save=False,
                            dpi=300, conv_methods=None):
    tick_label_size = 36
    axis_label_size = 36
    rows, columns = 2, 6
    f1, f2 = lambda x, pos: "{:.3f}".format(x), lambda x, pos: "{:.2f}".format(x)
    fig = plt.figure(figsize=(width, height))
    ax1 = plt.subplot2grid((rows, columns), (0, 0), rowspan=2, colspan=2)
    ax2 = plt.subplot2grid((rows, columns), (0, 2), rowspan=2, colspan=2)
    ax3 = plt.subplot2grid((rows, columns), (0, 4), rowspan=2, colspan=2)
    ax1.set_title("SSIM", **title_font_small), ax1.yaxis.grid(True)
    ax2.set_title("PSNR", **title_font_small), ax2.yaxis.grid(True)
    ax3.set_title("VIF", **title_font_small), ax3.yaxis.grid(True)

    for p_axis, metric_str in zip([ax1, ax2, ax3], ['ssim', 'psnr', 'vif']):
        x_labels, y1_labels, y1_errors = [], [], []
        for x_key, res_d in res_dicts.items():
            x_labels.append(str(x_key))
            y1_labels.append(round(res_d[metric_str + "_res"][0], 3))
            y1_errors.append(round(res_d[metric_str + "_res"][1], 3))
        if conv_methods is not None:
            conv_results = get_conventional_results('~/expers/sr_redo/{}/'.format(conv_methods), x_labels, metric_str,
                                                    eval_axis)
        else:
            conv_results = None
        p_axis.set_xticklabels(x_labels)
        p_axis.set_xlabel(r'Upsampling factor (K)', fontsize=axis_label_size)
        p_axis.errorbar(x_labels, y1_labels, c='orange', label='AISR', yerr=y1_errors, fmt='o',
                        solid_capstyle='projecting', capsize=15, markersize=20)
        if conv_methods:
            plot_conv_results(conv_results, p_axis, x_labels)
            p_axis.legend(loc="lower left", bbox_to_anchor=[0.01, 0.05],
                          ncol=1, prop={'size': axis_label_size})
        p_axis.plot(x_labels, y1_labels, c='orange')  # label='Reconstructed slices')
        p_axis.tick_params(axis='both', which='major', labelsize=tick_label_size)

    fig.tight_layout(rect=[0.03, 0.03, 0.97, 0.97])
    if do_save:
        filename = os.path.expanduser(filename)
        plt.savefig(filename, bbox_inches='tight', dpi=dpi)
        print('INFO - figure saved to {} '.format(filename))
    if do_show:
        plt.show()


def lambda_compare(res_dicts, height=5, width=7, filename=None, do_show=True, do_save=False,
                            dpi=300):
    tick_label_size = 36
    axis_label_size = 36
    rows, columns = 2, 6
    f1, f2 = lambda x, pos: "{:.3f}".format(x), lambda x, pos: "{:.2f}".format(x)
    fig = plt.figure(figsize=(width, height))
    ax1 = plt.subplot2grid((rows, columns), (0, 0), rowspan=2, colspan=2)
    ax2 = plt.subplot2grid((rows, columns), (0, 2), rowspan=2, colspan=2)
    ax3 = plt.subplot2grid((rows, columns), (0, 4), rowspan=2, colspan=2)
    ax1.set_title("SSIM", **title_font_small), ax1.yaxis.grid(True)
    ax2.set_title("PSNR", **title_font_small), ax2.yaxis.grid(True)
    ax3.set_title("VIF", **title_font_small), ax3.yaxis.grid(True)

    for p_axis, metric_str in zip([ax1, ax2, ax3], ['ssim', 'psnr', 'vif']):
        x_labels, y1_labels, y1_errors, y2_labels, y2_errors = [], [], [], [], []
        for x_key, res_d in res_dicts.items():
            x_labels.append(str(x_key))
            y1_labels.append(round(res_d[metric_str + "_res_recon"][0], 3))
            y2_labels.append(round(res_d[metric_str + "_res_synth"][0], 3))
            y1_errors.append(round(res_d[metric_str + "_res_recon"][1], 3))
            y2_errors.append(round(res_d[metric_str + "_res_synth"][1], 3))
        p_axis.set_xticklabels(x_labels)
        p_axis.set_xlabel(r'$\lambda$', fontsize=axis_label_size)
        # ax1.scatter(x_labels, y1_labels, c='orange', )
        p_axis.plot(x_labels, y1_labels, c='orange')  # label='Reconstructed slices')
        p_axis.set_ylabel('Reconstructed', fontsize=axis_label_size)
        p_axis.errorbar(x_labels, y1_labels, c='orange', label='Reconstructed slices', yerr=y1_errors, fmt='o',
                        solid_capstyle='projecting', capsize=6)

        ax11 = p_axis.twinx()
        ax11.errorbar(x_labels, y2_labels, c='cornflowerblue', label='Synthesized slices', yerr=y2_errors, fmt='o',
                      solid_capstyle='projecting', capsize=6)
        ax11.plot(x_labels, y2_labels, c='cornflowerblue')  # label='Synthesized slices')
        # ax2.scatter(x_labels, y2_labels, c='cornflowerblue', )
        p_axis.tick_params(axis='both', which='major', labelsize=tick_label_size)
        ax11.tick_params(axis='both', which='major', labelsize=tick_label_size)
        ax11.set_ylabel('Synthesized', fontsize=axis_label_size)

        p_axis.legend(loc="center",  bbox_to_anchor=[0.5, 0.2],
                        ncol=1, prop={'size': axis_label_size})
        ax11.legend(loc="center", bbox_to_anchor=[0.5, 0.12],
                      ncol=1, prop={'size': axis_label_size})
        if metric_str == "ssim":
            # p_axis.set_ylim([0.60, 0.99])
            p_axis.set_ylim([0.3, 1])
            ax11.set_ylim([0.3, 1])
        elif metric_str == 'psnr':
            # p_axis.set_ylim([15, 37])
            p_axis.set_ylim([10., 40])
            ax11.set_ylim([10., 40])
        elif metric_str == 'vif':
            # p_axis.set_ylim([0.70, 0.95])
            p_axis.set_ylim([0.3, 1])
            ax11.set_ylim([0.3, 1])

    fig.tight_layout(rect=[0.03, 0.03, 0.97, 0.97])
    if do_save:
        filename = os.path.expanduser(filename)
        plt.savefig(filename, bbox_inches='tight', dpi=dpi)
        print('INFO - figure saved to {} '.format(filename))
    if do_show:
        plt.show()