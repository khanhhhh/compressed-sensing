import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

from example.compressed_sensing import discrete_fourier_2d
from example.compressed_sensing.compressed_sensing import reconstruct
from example.compressed_sensing.measure import create_measure_matrix

if __name__ == "__main__":
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12.0, 4.0))
    ax = ax.flatten()
    ax_index = 0

    width, height = 0, 0


    def open(filename: str = "example_2d.png", num_pixels: int = 100) -> np.ndarray:
        im = Image.open(filename)
        h, w = im.size
        scale = np.sqrt(num_pixels / (h * w))
        global width, height
        width, height = int(w * scale), int(h * scale)
        im.thumbnail(size=(height, width), resample=Image.ANTIALIAS)
        im = np.array(im)
        im = im[:, :, 0]
        height, width = im.shape
        im = im.flatten()
        return im


    def draw(im: np.ndarray, title: str = "im"):
        im = im.real.astype(np.float64)  # take real part
        im += 0.5
        im[im < 0] = 0
        im[im > 255] = 255
        im = im.astype(np.uint8)
        im = im.reshape((height, width))
        global ax_index
        ax[ax_index].imshow(im)
        ax[ax_index].set_title(title)
        ax_index += 1


    true_signal = open(num_pixels=2000)

    draw(true_signal, "true signal")

    N = len(true_signal)
    D = int(0.6 * N)
    measure_matrix = create_measure_matrix(D, N)
    measure_signal = measure_matrix @ true_signal

    draw(measure_matrix.T @ measure_signal, "measure signal")

    reconstruct_signal = reconstruct(measure_signal, measure_matrix, discrete_fourier_2d.backward(height, width, height, width))

    draw(reconstruct_signal, "reconstruct signal")

    plt.tight_layout()
    plt.show()
