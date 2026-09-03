from timeit import timeit

import numpy as np
from numpy.testing import assert_array_equal
from PIL import Image
from skimage.transform import resize_local_mean

# def test_resize_local_mean_labels():
a = np.arange(8)[..., np.newaxis] + np.arange(6)[::-1, np.newaxis].T

target_size = (a.shape[-2] * 3, a.shape[-1] * 3)

a = a.astype(np.uint8)
b = resize_local_mean(
    a, output_shape=target_size, grid_mode=True, preserve_range=True
).astype(a.dtype)
c = np.asarray(Image.fromarray(a).resize(target_size[::-1], resample=0))

assert_array_equal(b, c)

repeats = 100000
time_ski = timeit(
    stmt="resize_local_mean(a, output_shape=(a.shape[-2] * 3, a.shape[-1] * 3), grid_mode=True, preserve_range=True).astype(a.dtype)",
    setup="import numpy as np; from skimage.transform import resize_local_mean; a = (np.arange(8)[..., np.newaxis] + np.arange(6)[::-1, np.newaxis].T).astype(np.uint8)",
    number=repeats,
)

time_pil = timeit(
    stmt="np.asarray(Image.fromarray(a).resize((a.shape[-1] * 3, a.shape[-2] * 3), resample=0))",
    setup="import numpy as np; from PIL import Image; a = (np.arange(8)[..., np.newaxis] + np.arange(6)[::-1, np.newaxis].T).astype(np.uint8)",
    number=repeats,
)

print(time_ski)
print(time_pil)

assert a.dtype == b.dtype
