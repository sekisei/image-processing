import os
import numpy as np
from PIL import Image, UnidentifiedImageError
from numba import cuda
import cv2
import glob
from tqdm import tqdm

class SBSM:
    def check_image_files(self, dir_path):
        try:
            img_sizes = []
            file_paths = glob.glob(f"{dir_path}/*")
            for file_path in file_paths:
                with Image.open(file_path) as img:
                    img.verify()
                with Image.open(file_path) as img:
                    img_sizes.append(img.size)
            img_base_size = img_sizes[0]
            is_same_size = all(size == img_base_size for size in img_sizes)
            if not is_same_size:
                raise RuntimeError("All images must have the same size.")
            return np.asarray(Image.open(file_paths[0])).shape
        except UnidentifiedImageError as e:
            raise RuntimeError(f"Can't open image file: {e.filename}")

    # check_pixel_values is called by get_hist, and it is executed on GPU.
    # It calculates the frequency distribution of pixel values for each pixel position across all images.
    @staticmethod
    @cuda.jit
    def check_pixel_values(img_arrays, freq_dists):
        bIdx_x = cuda.blockIdx.x
        bIdx_y = cuda.blockIdx.y
        tIdx_x = cuda.threadIdx.x
        tIdx_y = cuda.threadIdx.y
        x = bIdx_x * cuda.blockDim.x + tIdx_x
        y = bIdx_y * cuda.blockDim.y + tIdx_y
        batch, height, width, _ = img_arrays.shape

        if x < width and y < height:
            for b in range(batch):
                freq_dists[y][x][0][img_arrays[b][y][x][0]] += 1
                freq_dists[y][x][1][img_arrays[b][y][x][1]] += 1
                freq_dists[y][x][2][img_arrays[b][y][x][2]] += 1

    def get_hist(self, dir_path, img_shape, batch_size):
        height, width, _ = img_shape

        # Use 16x16 blocks to cover the image, and calculate the grid size accordingly.
        # example: for a 3648x2736 image, we need 228x171 blocks to cover it with 16x16 threads.
        block_xy_size = (16, 16)
        grid_x_size = (width + block_xy_size[0] - 1) // block_xy_size[0]
        grid_y_size = (height + block_xy_size[1] - 1) // block_xy_size[1]

        # uint16 is enough for 65535 frames, and it can save memory compared to uint32.
        freq_dists = np.zeros((height, width, 3, 256), dtype = np.uint16) 
        img_paths = glob.glob(f"{dir_path}/*")
        img_paths_batch = [img_paths[i : i + batch_size] for i in range(0, len(img_paths), batch_size)]

        for batch in tqdm(img_paths_batch):
            img_lists = [Image.open(img_path) for img_path in batch]
            img_arrays = np.asarray(img_lists, dtype = np.uint8, copy = True) # copy=True to make it writeable to transfer to GPU
            SBSM.check_pixel_values[(grid_x_size, grid_y_size), block_xy_size](img_arrays, freq_dists) # 実装中(未完成)
        return freq_dists

    @staticmethod
    @cuda.jit
    def sbsm_gpu(input, hist, sum):
        bIdx_x = cuda.blockIdx.x
        bIdx_y = cuda.blockIdx.y
        tIdx_x = cuda.threadIdx.x
        tIdx_y = cuda.threadIdx.y
        x = bIdx_x * cuda.blockDim.x + tIdx_x
        y = bIdx_y * cuda.blockDim.y + tIdx_y
        height, width = input.shape[0], input.shape[1]
        if x < width and y < height:
            P_w0 = 0.95
            P_w1 = 0.05
            I = input

            # Sum of pixel values for each pixel position across all images, which is used to calculate the probability.
            sum_0 = sum[y][x][0]
            sum_1 = sum[y][x][1]
            sum_2 = sum[y][x][2]

            # Frequency of the pixel value at the current position for each color channel, which is used to calculate the probability.
            pixel_value_count_0 = hist[y][x][0][I[y][x][0]]
            pixel_value_count_1 = hist[y][x][1][I[y][x][1]]
            pixel_value_count_2 = hist[y][x][2][I[y][x][2]]
            
            # Indexs of image are opposite to Indexs about hist array. 
            P_I_given_by_w0_0 = pixel_value_count_0 / sum_0
            P_I_given_by_w0_1 = pixel_value_count_1 / sum_1
            P_I_given_by_w0_2 = pixel_value_count_2 / sum_2
            P_I_given_by_w1_0 = 1.0 / 256.0
            P_I_given_by_w1_1 = 1.0 / 256.0
            P_I_given_by_w1_2 = 1.0 / 256.0

            # Calculating the probability of the pixel value given the background model (w0) and the foreground model (w1) for each color channel, and then multiplying them together to get the overall probability for each model.
            P_I_given_by_w0 = P_I_given_by_w0_0 * P_I_given_by_w0_1 * P_I_given_by_w0_2
            P_I_given_by_w1 = P_I_given_by_w1_0 * P_I_given_by_w1_1 * P_I_given_by_w1_2
            
            # # 1 = p_w0 * P_I_given_by_w0 / P_I + p_w1 * P_I_given_by_w1 / P_I
            P_I = P_w0 * P_I_given_by_w0 + P_w1 * P_I_given_by_w1
            P_w0_given_by_I = (P_I_given_by_w0 * P_w0) / P_I
            P_w1_given_by_I = (P_I_given_by_w1 * P_w1) / P_I
            if P_w1_given_by_I > P_w0_given_by_I:
                input[y][x][0] = 255
                input[y][x][1] = 0
                input[y][x][2] = 0
    
if __name__ == '__main__':
    device = cuda.get_current_device()
    print("MAX_THREADS_PER_BLOCK =", device.MAX_THREADS_PER_BLOCK)
    print("MAX_BLOCK_DIM_X =", device.MAX_BLOCK_DIM_X)
    print("MAX_BLOCK_DIM_Y =", device.MAX_BLOCK_DIM_Y)
    print("MAX_BLOCK_DIM_Z =", device.MAX_BLOCK_DIM_Z)
    print("MAX_GRID_DIM_X =", device.MAX_GRID_DIM_X)
    print("MAX_GRID_DIM_Y =", device.MAX_GRID_DIM_Y)
    print("MAX_GRID_DIM_Z =", device.MAX_GRID_DIM_Z)
    print("WARP_SIZE =", device.WARP_SIZE)

    INPUT_DIR_PATH = 'data/input/bg_images'
    OUTPUT_IMAGE_PATH = 'data/output/sbsm_gpu.jpg'
    BATCH_SIZE = 128

    # Generate histogram data from background images and save it as a .npy file.
    # sbsm = SBSM()
    # img_shape = sbsm.check_image_files(INPUT_DIR_PATH)
    # hist = sbsm.get_hist(INPUT_DIR_PATH, img_shape, BATCH_SIZE)
    # np.save('data/hist_xy.npy', hist)

    hist = np.load('data/hist_xy.npy')
    sample_image = np.asarray(Image.open('data/input/sample.png'), copy = True)
    height, width, _ = sample_image.shape

    block_xy_size = (16, 16)
    grid_x_size = (width + block_xy_size[0] - 1) // block_xy_size[0]
    grid_y_size = (height + block_xy_size[1] - 1) // block_xy_size[1]
    SBSM.sbsm_gpu[(grid_x_size, grid_y_size), block_xy_size](sample_image, hist, np.sum(hist, axis=3))
    img_out = Image.fromarray(np.uint8(sample_image))
    img_out.save(OUTPUT_IMAGE_PATH)
