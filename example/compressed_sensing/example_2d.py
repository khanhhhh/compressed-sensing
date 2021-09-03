import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

from example.compressed_sensing import fourier_2d, laplacian, compressed_sensing
from example.compressed_sensing.measure import create_measure_matrix

if __name__ == "__main__":
    fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(16.0, 4.0))
    ax = ax.flatten()
    ax_index = 0

    height, width, channel = 0, 0, 0


    def open_im(filename: str, num_pixels: int) -> np.ndarray:
        im = Image.open(filename)
        h, w = im.size
        scale = np.sqrt(num_pixels / (h * w))
        if scale < 1:
            im.thumbnail(size=(int(w * scale), int(h * scale)), resample=Image.ANTIALIAS)
        im = np.array(im)
        return im


    def draw_im(im: np.ndarray, title: str = "im"):
        global ax_index
        ax[ax_index].imshow(im)
        ax[ax_index].set_title(title)
        ax_index += 1


    def im2sig(im: np.ndarray) -> np.ndarray:
        signal = im.reshape((height * width, channel))
        return signal


    def sig2im(signal: np.ndarray) -> np.ndarray:
        signal = signal.real.astype(np.float64)  # take real part
        signal += 0.5
        signal[signal < 0] = 0
        signal[signal > 255] = 255
        signal = signal.astype(np.uint8)
        im = signal.reshape((height, width, channel))
        return im

    # open true im
    true_im = open_im(filename="example_2d.png", num_pixels=1000)
    draw_im(true_im, "true signal")
    height, width, channel = true_im.shape
    true_signal = im2sig(true_im)

    # create measure im
    N = len(true_signal)
    D = int(0.5 * N)
    measure_matrix = create_measure_matrix(D, N)
    measure_signal = measure_matrix @ true_signal
    measure_im = sig2im(measure_matrix.T @ measure_signal)
    draw_im(measure_im, "measure signal")

    # reconstruct true im (fourier basis)
    _, inverse_fourier_matrix = fourier_2d.basis(height, width, height, width)
    reconstruct_signal = np.empty(shape=(height * width, channel), dtype=np.complex128)
    for c in range(channel):
        reconstruct_signal[:, c] = compressed_sensing.reconstruct_complex(
            measure_signal=measure_signal[:, c],
            measure_matrix=measure_matrix,
            inverse_fourier_matrix=inverse_fourier_matrix,
        )
    reconstruct_im = sig2im(reconstruct_signal)
    draw_im(reconstruct_im, "reconstruct signal (fourier basis)")

    # reconstruct true im (laplacian basis)
    adj = laplacian.create_grid_adj_matrix(height, width)
    _, inverse_laplacian_matrix = laplacian.basis(adj)
    reconstruct_signal = np.empty(shape=(height * width, channel), dtype=np.complex128)
    for c in range(channel):
        reconstruct_signal[:, c] = compressed_sensing.reconstruct_real(
            measure_signal=measure_signal[:, c],
            measure_matrix=measure_matrix,
            inverse_transform_matrix=inverse_laplacian_matrix,
        )
    reconstruct_im = sig2im(reconstruct_signal)
    draw_im(reconstruct_im, "reconstruct signal (laplacian basis)")

    # draw
    fig.suptitle(f"image size: {height} x {width}")
    plt.tight_layout()
    plt.show()
