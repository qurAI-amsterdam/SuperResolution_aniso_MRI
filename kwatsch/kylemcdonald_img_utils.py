import numpy as np
import math
import shutil

from io import BytesIO
import numpy as np
import PIL.Image
import IPython.display
from math import sqrt
from imageio import imsave


def find_rectangle(n):
    max_side = int(math.sqrt(n))
    for h in range(2, max_side + 1)[::-1]:
        w = n // h
        if (h * w) == n:
            return (h, w)
    return (n, 1)


def swapaxes(x, a, b):
    try:
        return x.swapaxes(a, b)
    except AttributeError:  # support pytorch
        return x.transpose(a, b)


# 1d images (n, h*w): no
# 2d images (n, h, w): yes
# 3d images (n, h, w, c): yes
def make_mosaic(x, nx=None, ny=None):
    if not isinstance(x, np.ndarray):
        x = np.asarray(x)
    n, h, w = x.shape[:3]
    has_channels = len(x.shape) > 3
    if has_channels:
        c = x.shape[3]

    if nx is None and ny is None:
        ny, nx = find_rectangle(n)
    elif ny is None:
        ny = n // nx
    elif nx is None:
        nx = n // ny

    end_shape = (w, c) if has_channels else (w,)
    mosaic = x.reshape(ny, nx, h, *end_shape)
    mosaic = swapaxes(mosaic, 1, 2)
    hh = mosaic.shape[0] * mosaic.shape[1]
    ww = mosaic.shape[2] * mosaic.shape[3]
    end_shape = (ww, c) if has_channels else (ww,)
    mosaic = mosaic.reshape(hh, *end_shape)
    return mosaic


# 1d images (n, h*w): no
# 2d images (n, h, w): yes
# 3d images (n, h, w, c): yes
# assumes images are square if underspecified
def unmake_mosaic(mosaic, nx=None, ny=None, w=None, h=None):
    hh, ww = mosaic.shape[:2]

    if nx is not None or ny is not None:
        if nx is None:
            h = hh // ny
            w = h
            nx = ww // w
        elif ny is None:
            w = ww // nx
            h = w
            ny = hh // h
        else:
            w = ww // nx
            h = hh // ny

    elif w is not None or h is not None:
        if w is None:
            w = h
        elif h is None:
            h = w
        nx = ww // w
        ny = hh // h

    end_shape = (w, mosaic.shape[2]) if len(mosaic.shape) > 2 else (w,)

    x = mosaic.reshape(ny, h, nx, *end_shape)
    x = swapaxes(x, 1, 2)
    x = x.reshape(-1, h, *end_shape)
    return x


def show_array(img, fmt='png', filename=None, retina=False, zoom=None):
    if img is None:
        raise TypeError('input image not provided')

    if len(img.shape) == 1:
        n = len(img)
        side = int(sqrt(n))
        if (side * side) == n:
            img = img.reshape(side, side)
        else:
            raise ValueError('input is one-dimensional', img.shape)
    if len(img.shape) == 3 and img.shape[-1] == 1:
        img = img.squeeze()
    img = np.uint8(np.clip(img, 0, 255))

    image_data = BytesIO()
    PIL.Image.fromarray(img).save(image_data, fmt)

    height, width = img.shape[:2]
    if zoom is not None:
        width *= zoom
        height *= zoom
    IPython.display.display(IPython.display.Image(data=image_data.getvalue(),
                                                  width=width,
                                                  height=height,
                                                  retina=retina))

    if filename is not None:
        imsave(filename, img)

##################### old version


# def find_rectangle(n):
#     max_side = int(math.sqrt(n))
#     for h in range(2, max_side+1)[::-1]:
#         w = n // h
#         if (w * h) == n:
#             return (w,h)
#     return (n, 1)
#
#
# # should work for 1d and 2d images, assumes images are square but can be overriden
# def make_mosaic(images, n=None, nx=None, ny=None, w=None, h=None):
#     if n is None and nx is None and ny is None:
#         nx, ny = find_rectangle(len(images))
#     else:
#         nx = n if nx is None else nx
#         ny = n if ny is None else ny
#     images = np.array(images)
#     if images.ndim == 2:
#         side = int(np.sqrt(len(images[0])))
#         h = side if h is None else h
#         w = side if w is None else w
#         images = images.reshape(-1, h, w)
#     else:
#         h = images.shape[1]
#         w = images.shape[2]
#     nx = int(nx)
#     ny = int(ny)
#     h = int(h)
#     w = int(w)
#     image_gen = iter(images)
#     # should replace this code with https://stackoverflow.com/a/42041135/940196
#     if len(images.shape) > 3:
#         mosaic = np.empty((h*ny, w*nx, images.shape[3]))
#     else:
#         mosaic = np.empty((h*ny, w*nx))
#     for i in range(ny):
#         ia = (i)*h
#         ib = (i+1)*h
#         for j in range(nx):
#             ja = j*w
#             jb = (j+1)*w
#             mosaic[ia:ib, ja:jb] = next(image_gen)
#     return mosaic