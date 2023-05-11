import numpy as np

def downsample(arr, factor, mode='sum'):
    """Converts a shape (h, w) array into a (h//f, w//f) array
    by summing or averaging the values in f*f blocks."""

    h, w = arr.shape

    if h % factor != 0 or w % factor != 0:
        raise ValueError('Array dimensions should be divisible by the downsampling factor.')

    downsampled_h, downsampled_w = h // factor, w // factor
    downsampled_arr = np.zeros((downsampled_h, downsampled_w))

    for i in range(downsampled_h):
        for j in range(downsampled_w):
            block = arr[i * factor: (i + 1) * factor, j * factor: (j + 1) * factor]
            if mode == 'sum':
                downsampled_arr[i, j] = block.sum(axis=(0, 1))
            elif mode == 'average':
                downsampled_arr[i, j] = block.mean(axis=(0, 1))
            else:
                raise ValueError(mode)

    return downsampled_arr
