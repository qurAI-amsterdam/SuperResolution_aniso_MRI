import matplotlib.pyplot as plt
import numpy as np


def make_boxplots(res_vectors, m_labels, fig_file_name, do_show=True, do_save=False,
                  use_fill_color=True, show_means=False):
    """
        m_labels: ['AE+C-loss', 'Linear', 'B-spline', 'Lanczos']
    """

    fill_color = ['xkcd:periwinkle', 'xkcd:sea green', 'xkcd:sun yellow', 'xkcd:greeny grey',
                  'xkcd:orange', 'xkcd:deep purple', 'xkcd:dark turquoise']
    model_colors = {'Linear': (fill_color[4], 0.45), 'B-spline': (fill_color[4], 0.7),
                    'Lanczos': (fill_color[4], 0.95),  'ACAI': (fill_color[5], 1), 'AE': (fill_color[5], 1),
                    'AE_C-loss': (fill_color[5], 0.7)}
    box_position = {'AE': 4, 'Linear': 1, 'B-spline': 2, 'Lanczos': 3}
    title_font_small = {'fontname': 'Monospace', 'size': '26', 'color': 'black', 'weight': 'normal'}
    fig = plt.figure(figsize=(23, 10))
    rows, columns = 2, 6
    # fig.suptitle("{}".format("Compare methods"), **title_font_small)
    ax1 = plt.subplot2grid((rows, columns), (0, 0), rowspan=2, colspan=2)
    ax2 = plt.subplot2grid((rows, columns), (0, 2), rowspan=2, colspan=2)
    ax3 = plt.subplot2grid((rows, columns), (0, 4), rowspan=2, colspan=2)
    ax1.set_title("SSIM", **title_font_small), ax1.yaxis.grid(True)
    ax2.set_title("PSNR", **title_font_small), ax2.yaxis.grid(True)
    ax3.set_title("VIF", **title_font_small), ax3.yaxis.grid(True)
    for m_idx in np.arange(len(res_vectors['ssim'])):
        method = m_labels[m_idx]
        c, alpha = model_colors[method]
        # bp1 = ax1.boxplot(res_vectors['ssim'], labels=m_labels, showmeans=True, patch_artist=use_fill_color)
        bp1 = ax1.boxplot(res_vectors['ssim'][m_idx], positions=[m_idx + 1], notch=False, patch_artist=use_fill_color,
                          showmeans=show_means, labels=[method], widths=0.6,
                    boxprops=dict(color=c, alpha=alpha, linewidth=4),
                    capprops=dict(color=c, alpha=alpha, linewidth=4),
                    whiskerprops=dict(color=c, alpha=alpha, linewidth=4),
                    flierprops=dict(color=c, markeredgecolor=c, alpha=alpha, linewidth=4),
                    medianprops=dict(color=c, alpha=alpha, linewidth=4),
                          meanprops=dict(markeredgecolor=c, alpha=alpha, markersize=15,
                                              markerfacecolor=c))
        bp2 = ax2.boxplot(res_vectors['psnr'][m_idx], positions=[m_idx + 1], notch=False, patch_artist=use_fill_color,
                          showmeans=show_means, labels=[method], widths=0.6,
                          boxprops=dict(color=c, alpha=alpha, linewidth=4),
                          capprops=dict(color=c, alpha=alpha, linewidth=4),
                          whiskerprops=dict(color=c, alpha=alpha, linewidth=4),
                          flierprops=dict(color=c, markeredgecolor=c, alpha=alpha, linewidth=4),
                          medianprops=dict(color=c, alpha=alpha, linewidth=4),
                          meanprops=dict(markeredgecolor=c, alpha=alpha, markersize=15,
                                         markerfacecolor=c))

        bp3 = ax3.boxplot(res_vectors['vif'][m_idx], positions=[m_idx + 1], notch=False, patch_artist=use_fill_color,
                          showmeans=show_means, labels=[method], widths=0.6,
                          boxprops=dict(color=c, alpha=alpha, linewidth=4),
                          capprops=dict(color=c, alpha=alpha, linewidth=4),
                          whiskerprops=dict(color=c, alpha=alpha, linewidth=4),
                          flierprops=dict(color=c, markeredgecolor=c, alpha=alpha, linewidth=4),
                          medianprops=dict(color=c, alpha=alpha, linewidth=4),
                          meanprops=dict(markeredgecolor=c, alpha=alpha, markersize=15,
                                         markerfacecolor=c))


    ax1.tick_params(axis='both', which='major', labelsize=26)
    plt.setp(ax1.get_xticklabels(), rotation=90)
    ax2.tick_params(axis='both', which='major', labelsize=26)
    plt.setp(ax2.get_xticklabels(), rotation=90)
    ax3.tick_params(axis='both', which='major', labelsize=26)
    fig.tight_layout(rect=[0.03, 0.03, 0.97, 0.97])
    plt.setp(ax3.get_xticklabels(), rotation=90)
    if do_save:
        plt.savefig(fig_file_name, bbox_inches='tight')
        print(("INFO - Successfully saved fig %s" % fig_file_name))
    if do_show:
        plt.show()

