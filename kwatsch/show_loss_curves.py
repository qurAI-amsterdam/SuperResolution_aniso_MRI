import numpy as np
import matplotlib.pyplot as plt
from kwatsch.acai_utils import build_batches
from IPython.display import clear_output


def show_loss_curves(trainer):
    fig = plt.figure(facecolor='w', figsize=(12, 10))
    fig1_ax1 = fig.add_subplot(411)
    fig2_ax1 = fig.add_subplot(412)
    # fig3_ax1 = fig.add_subplot(413)
    fig1_ax2 = fig1_ax1.twinx()
    fig2_ax2 = fig2_ax1.twinx()
    # fig3_ax2 = fig3_ax1.twinx()
    for key in trainer.losses:
        total = len(trainer.losses[key])
        skip = 1 + (total // 1000)
        y = build_batches(trainer.losses[key], skip).mean(axis=-1)
        x = np.linspace(0, total, len(y))
        if key == "loss":
            fig1_ax1.plot(x, y, label=key, lw=0.5, c='r')
        if key == "loss_05":
            fig1_ax2.plot(x, y, label=key, lw=0.5, c='g')
        if key == "loss_recon":
            fig2_ax1.plot(x, y, label=key, lw=0.5, c='r')
        if key == "loss_recon_05":
            fig2_ax2.plot(x, y, label=key, lw=0.5, c='g')
    fig1_ax1.legend(loc='upper right')
    fig1_ax2.legend(loc='upper left')
    fig2_ax1.legend(loc='upper right')
    fig2_ax2.legend(loc='upper left')
    clear_output(wait=True)
    plt.show()