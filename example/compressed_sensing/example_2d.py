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

    width, height, channel = 0, 0, 0


    def open(filename: str, num_pixels: int) -> np.ndarray:
        im = Image.open(filename)
        h, w = im.size
        scale = np.sqrt(num_pixels / (h * w))
        im.thumbnail(size=(int(w * scale), int(h * scale)), resample=Image.ANTIALIAS)
        im = np.array(im)
        # im = im[:, :, 0]
        global width, height, channel
        height, width, channel = im.shape
        im = im.reshape((height*width, channel))
        return im


    def draw(im: np.ndarray, title: str = "im"):
        im = im.real.astype(np.float64)  # take real part
        im += 0.5
        im[im < 0] = 0
        im[im > 255] = 255
        im = im.astype(np.uint8)
        im = im.reshape((height, width, channel))
        global ax_index
        ax[ax_index].imshow(im)
        ax[ax_index].set_title(title)
        ax_index += 1


    true_signal = open(filename="example_2d.png", num_pixels=6000)

    draw(true_signal, "true signal")

    N = len(true_signal)
    D = int(0.3 * N)
    measure_matrix = create_measure_matrix(D, N)
    measure_signal = measure_matrix @ true_signal

    draw(measure_matrix.T @ measure_signal, "measure signal")

    reconstruct_signal = np.empty(shape=(height*width, channel), dtype=np.complex128)
    for c in range(channel):
        reconstruct_signal[:, c] = reconstruct(measure_signal[:, c], measure_matrix, discrete_fourier_2d.backward(2*height, 2*width, height, width))

    draw(reconstruct_signal, "reconstruct signal")

    plt.tight_layout()
    plt.show()
