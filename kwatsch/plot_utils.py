from matplotlib import animation
from IPython.display import HTML
import os
import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np


root_dir = "/home/jorg/expers/cardiac_sr/acdc_new_simple/p128_l16_16_12/images_sr/"
fname = os.path.join(root_dir, "patient016/patient016_4d_acai.nii.gz")
img4d = sitk.GetArrayFromImage(sitk.ReadImage(fname))
frame_id = img4d.shape[0] // 2
np_img = img4d[frame_id]
print("#Slices {}".format(np_img.shape[0]))

first_slice = np_img[0]
frames = np_img.shape[0] - 3


def imslices_generator():
    for i in np.arange(1, np_img.shape[0]):
        yield np_img[i]


def init_frames():
    return np_img[0]


my_gen = imslices_generator()


def animate(j):
    imslice = next(my_gen)
    im1.set_data(imslice)
    return (im1,)


fig, ax = plt.subplots(1, 1, figsize=(10,5))
im1 = ax.imshow(first_slice, vmin=0, vmax=1, cmap='gray')

anim = animation.FuncAnimation(fig, animate, frames=frames, interval=200, blit=False)
HTML(anim.to_html5_video())
