import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, FuncFormatter
import numpy as np
from statannot import add_stat_annotation
import pandas as pd


FILL_COLOR = ['xkcd:periwinkle', 'xkcd:sea green', 'xkcd:sun yellow', 'xkcd:greeny grey',
                  'xkcd:orange', 'xkcd:deep purple', 'xkcd:dark turquoise']
METHOD_LABELS = {'ae': r'AISR$_{\lambda=0}$', 'linear': 'Linear', 'bspline': 'B-spline', 'lanczos': 'Lanczos',
                 'ae_combined': 'AISR', 'caisr': 'AISR', 'ae_caisr': 'Combined',
                 'reference': "Original"}
MODEL_COLORS = {'linear': (FILL_COLOR[4], 0.45), 'bspline': (FILL_COLOR[4], 0.6),
                    'lanczos': (FILL_COLOR[4], 0.95),  'ae_combined': (FILL_COLOR[5], 1.),
                    'caisr': (FILL_COLOR[5], 1.),
                    'ae': (FILL_COLOR[5], 0.7),
                    'ae_caisr': (FILL_COLOR[5], 0.5)}
METHOD_3_LABELS = {'ae': r'AISR$_{\lambda=0}$', 'ae_combined': r'AISR$_{\lambda=0.5}$',
                   'caisr': r'AISR$_{\lambda=0.5}$', 'ae_caisr': 'combined'}
MODEL_3_COLORS = {'ae_combined': (FILL_COLOR[5], 0.8),
                    'caisr': (FILL_COLOR[5], 0.8),
                    'ae': (FILL_COLOR[4], 0.8),
                    'ae_caisr': (FILL_COLOR[6], 0.8)}


def make_boxplots(result_dict, fig_file_name, do_show=True, do_save=False, verbose=0, width_box=0.5,
                  use_fill_color=False, show_means=False, df=None, method_key_filter=None, synth_only=False):
    """
        result_dict contains:
            keys: methods abbreviations: ae, linear, bspline, lanczos for now
            values: another dict with keys ['ssim', 'psnr', 'vif', 'lpips', 'ssim_res', 'psnr_res', 'vif_res']
    """
    # force pdf
    if '.pdf' not in fig_file_name:
        raise ValueError("Error - fig_file_name must have pdf extension!")
    specific_model_colors = MODEL_COLORS
    specific_model_labels = METHOD_LABELS
    if method_key_filter is not None and (len(method_key_filter) == 6 or len(method_key_filter) == 5):
        box_position = {'ae': 3, 'ae_combined': 4, 'linear': 0, 'bspline': 1, 'lanczos': 2, 'ae_caisr': 5}
        order = ['linear', 'bspline', 'lanczos', 'ae', 'ae_combined']
    elif method_key_filter is not None and len(method_key_filter) == 4:
        box_position = {'ae_combined': 3, 'linear': 0, 'bspline': 1, 'lanczos': 2}
        order = ['linear', 'bspline', 'lanczos', 'ae_combined']
    elif method_key_filter is not None and len(method_key_filter) == 2:
        box_position = {'ae_combined': 1, 'ae': 0}
        order = ['ae', 'ae_combined']
        specific_model_colors = MODEL_3_COLORS
        specific_model_labels = METHOD_3_LABELS
    else:
        raise ValueError("Error - check parameter method_key_filter and adjust box_position in method details")
    title_font_small = {'size': '36', 'color': 'black', 'weight': 'normal'}  # 'fontname': 'Monospace',
    tick_label_size = 34
    if df is None:
        fig = plt.figure(figsize=(23, 10))
    else:
        fig = plt.figure(figsize=(29, 13))
    rows, columns = 2, 6
    # fig.suptitle("{}".format("Compare methods"), **title_font_small)
    ax1 = plt.subplot2grid((rows, columns), (0, 0), rowspan=2, colspan=2)
    ax2 = plt.subplot2grid((rows, columns), (0, 2), rowspan=2, colspan=2)
    ax3 = plt.subplot2grid((rows, columns), (0, 4), rowspan=2, colspan=2)
    ax1.set_title("SSIM", **title_font_small), ax1.yaxis.grid(True)
    ax2.set_title("PSNR", **title_font_small), ax2.yaxis.grid(True)
    ax3.set_title("VIF", **title_font_small), ax3.yaxis.grid(True)
    f1, f2 = lambda x, pos: "{:.3f}".format(x), lambda x, pos: "{:.2f}".format(x)
    for meth, meth_dict in result_dict.items():
        meth = meth.replace("-", "").lower()
        if method_key_filter is not None:
            if meth not in method_key_filter:
                continue
        method = specific_model_labels[meth]
        c, alpha = specific_model_colors[meth]
        meth_box_pos = box_position[meth]
        metric = 'ssim_synth' if synth_only else "ssim"
        _ = ax1.boxplot(np.round(meth_dict[metric], decimals=3), positions=[meth_box_pos], notch=False, patch_artist=use_fill_color,
                          showmeans=show_means, labels=[method], widths=width_box,
                    boxprops=dict(color=c, alpha=alpha, linewidth=4),
                    capprops=dict(color=c, alpha=alpha, linewidth=4),
                    whiskerprops=dict(color=c, alpha=alpha, linewidth=4),
                    flierprops=dict(color=c, markeredgecolor=c, alpha=alpha, linewidth=4),
                    medianprops=dict(color=c, alpha=alpha, linewidth=4),
                          meanprops=dict(markeredgecolor=c, alpha=alpha, markersize=15,
                                              markerfacecolor=c))
        metric = 'psnr_synth' if synth_only else "psnr"
        _ = ax2.boxplot(np.round(meth_dict[metric], decimals=2), positions=[meth_box_pos], notch=False, patch_artist=use_fill_color,
                          showmeans=show_means, labels=[method], widths=width_box,
                          boxprops=dict(color=c, alpha=alpha, linewidth=4),
                          capprops=dict(color=c, alpha=alpha, linewidth=4),
                          whiskerprops=dict(color=c, alpha=alpha, linewidth=4),
                          flierprops=dict(color=c, markeredgecolor=c, alpha=alpha, linewidth=4),
                          medianprops=dict(color=c, alpha=alpha, linewidth=4),
                          meanprops=dict(markeredgecolor=c, alpha=alpha, markersize=15,
                                         markerfacecolor=c))
        metric = 'vif_synth' if synth_only else "vif"
        _ = ax3.boxplot(meth_dict[metric], positions=[meth_box_pos], notch=False, patch_artist=use_fill_color,
                          showmeans=show_means, labels=[method], widths=width_box,
                          boxprops=dict(color=c, alpha=alpha, linewidth=4),
                          capprops=dict(color=c, alpha=alpha, linewidth=4),
                          whiskerprops=dict(color=c, alpha=alpha, linewidth=4),
                          flierprops=dict(color=c, markeredgecolor=c, alpha=alpha, linewidth=4),
                          medianprops=dict(color=c, alpha=alpha, linewidth=4),
                          meanprops=dict(markeredgecolor=c, alpha=alpha, markersize=15,
                                         markerfacecolor=c))

        # ax1.set_ylim([np.min(df['ssim']), np.max(df['ssim'])])
        # ax2.set_ylim([np.min(df['psnr']), np.max(df['psnr'])])
        # ax3.set_ylim([np.min(df['vif']), np.max(df['vif'])])
    ax1.yaxis.set_major_formatter(FuncFormatter(f1))
    # ax2.yaxis.set_major_formatter(FuncFormatter(f2))
    # ax3.yaxis.set_major_formatter(FuncFormatter(f1))
    ax1.tick_params(axis='both', which='major', labelsize=tick_label_size)
    plt.setp(ax1.get_xticklabels(), rotation=90)
    ax2.tick_params(axis='both', which='major', labelsize=tick_label_size)
    plt.setp(ax2.get_xticklabels(), rotation=90)
    ax3.tick_params(axis='both', which='major', labelsize=tick_label_size)
    fig.tight_layout(rect=[0.01, 0.01, 0.99, 0.99])

    plt.setp(ax3.get_xticklabels(), rotation=90)
    if do_save:
        if synth_only:
            fig_file_name = fig_file_name.replace(".pdf", "_synth.pdf")
        plt.savefig(fig_file_name, bbox_inches='tight')  # , pad_inches=0)
        print(("INFO - Successfully saved fig %s" % fig_file_name))
    if df is not None:
        x, y = 'method', 'ssim'
        if 'ae' in method_key_filter and 'linear' in method_key_filter:
            add_stat_annotation(ax1, data=df, x=x, y=y, order=order,
                                box_pairs=[("linear", "ae"), ("bspline", "ae"), ("lanczos", "ae")],
                                test='Wilcoxon', text_format='star', loc='outside', verbose=verbose, fontsize=26,
                                stats_params={'alternative': 'less'})
        # for CAISR
        if 'ae_combined' in method_key_filter and 'linear' in method_key_filter:
            add_stat_annotation(ax1, data=df, x=x, y=y, order=order,
                                box_pairs=[("linear", "ae_combined"), ("bspline", "ae_combined"), ("lanczos", "ae_combined")],
                                test='Wilcoxon', text_format='star', loc='outside', verbose=verbose, fontsize=26,
                                stats_params={'alternative': 'less'})
        if 'ae' in method_key_filter:
            add_stat_annotation(ax1, data=df, x=x, y=y, order=order,
                                box_pairs=[("ae", "ae_combined"), ("ae", "ae_combined"), ("ae", "ae_combined")],
                                test='Wilcoxon', text_format='star', loc='outside', verbose=verbose, fontsize=26,
                                stats_params={'alternative': 'less'})

        x, y = 'method', 'psnr'
        if 'ae' in method_key_filter and 'linear' in method_key_filter:
            add_stat_annotation(ax2, data=df, x=x, y=y, order=order,
                                box_pairs=[("linear", "ae"), ("bspline", "ae"), ("lanczos", "ae")],
                                test='Wilcoxon', text_format='star', loc='outside', verbose=verbose, fontsize=26,
                                stats_params={'alternative': 'less'})
        if 'ae_combined' in method_key_filter and 'linear' in method_key_filter:
            add_stat_annotation(ax2, data=df, x=x, y=y, order=order,
                                box_pairs=[("linear", "ae_combined"), ("bspline", "ae_combined"), ("lanczos", "ae_combined")],
                                test='Wilcoxon', text_format='star', loc='outside', verbose=verbose, fontsize=26,
                                stats_params={'alternative': 'less'})
        if 'ae' in method_key_filter:
            add_stat_annotation(ax2, data=df, x=x, y=y, order=order,
                                box_pairs=[("ae", "ae_combined"), ("ae", "ae_combined"), ("ae", "ae_combined")],
                                test='Wilcoxon', text_format='star', loc='outside', verbose=verbose, fontsize=26,
                                stats_params={'alternative': 'less'})
        x, y = 'method', 'vif'
        if 'ae' in method_key_filter and 'linear' in method_key_filter:
            add_stat_annotation(ax3, data=df, x=x, y=y, order=order,
                                box_pairs=[("linear", "ae"), ("bspline", "ae"), ("lanczos", "ae")],
                                test='Wilcoxon', text_format='star', loc='outside', verbose=verbose, fontsize=26,
                                stats_params={'alternative': 'less'})
        if 'ae_combined' in method_key_filter and 'linear' in method_key_filter:
            add_stat_annotation(ax3, data=df, x=x, y=y, order=order,
                                box_pairs=[("linear", "ae_combined"), ("bspline", "ae_combined"),
                                           ("lanczos", "ae_combined")],
                                test='Wilcoxon', text_format='star', loc='outside', verbose=verbose, fontsize=26,
                                stats_params={'alternative': 'less'})
        if 'ae' in method_key_filter:
            add_stat_annotation(ax3, data=df, x=x, y=y, order=order,
                                box_pairs=[("ae", "ae_combined"), ("ae", "ae_combined"), ("ae", "ae_combined")],
                                test='Wilcoxon', text_format='star', loc='outside', verbose=verbose, fontsize=26,
                                stats_params={'alternative': 'less'})
        if do_save:
            fig.tight_layout(rect=[0.01, 0.01, 0.99, 0.99])
            fig_file_name = fig_file_name.replace(".pdf", "_with_stats.pdf")
            plt.savefig(fig_file_name, bbox_inches='tight') # , pad_inches=0)
            print(("INFO - Successfully saved fig with stats %s" % fig_file_name))
    if do_show:
        plt.show()


def convert_to_df(result_dict):
    cols = ['method', 'ssim', 'psrn', 'vif']
    df = None
    for meth, data_dict in result_dict.items():
        meth = meth.replace("-", "").lower()
        df_tmp = pd.DataFrame(
            {'method': [meth] * len(data_dict['ssim']), 'ssim': data_dict['ssim'], 'psnr': data_dict['psnr'],
             'vif': data_dict['vif']})
        df = df_tmp if df is None else df.append(df_tmp)
        print(meth, len(data_dict['ssim']), len(data_dict['psnr']), len(data_dict['vif']))
    return df