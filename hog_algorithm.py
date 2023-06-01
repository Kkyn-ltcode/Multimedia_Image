import numpy as np

def sobel_gradient(I):
    # Compute the gradients of the image in the x and y directions using the Sobel operator with kernel size = 3x3
    sobel_x = np.array([[-1, 0, 1], 
                        [-2, 0, 2], 
                        [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], 
                        [0, 0, 0], 
                        [1, 2, 1]])
    
    gx = np.zeros_like(I)
    gy = np.zeros_like(I)
    
    for i in range(1, I.shape[0] - 1):
        for j in range(1, I.shape[1] - 1):
            gx[i, j] = np.sum(sobel_x * I[i-1:i+2, j-1:j+2])
            gy[i, j] = np.sum(sobel_y * I[i-1:i+2, j-1:j+2])
    
    return gx, gy

def magnitude_and_orientation(gx, gy):
    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    orientation = np.arctan2(gy, gx) * 180 / np.pi

    return magnitude, orientation

def compute_cell_histogram(gradient_magnitudes, gradient_orientations, num_num_bins):
    # Create a histogram with the specified number of num_bins
    histogram = np.zeros(num_num_bins)
    
    # Compute the bin width in degrees
    bin_width = 180 / num_num_bins
    
    # Loop over all pixels in the cell
    for i in range(gradient_magnitudes.shape[0]):
        for j in range(gradient_magnitudes.shape[1]):
            # Compute the gradient orientation in degrees
            orientation = gradient_orientations[i, j]
            
            # Compute the bin index for this orientation
            bin_index = int(np.floor(orientation / bin_width))
            
            # Add the gradient magnitude to the corresponding bin
            histogram[bin_index] += gradient_magnitudes[i, j]
    
    return histogram

def normalize_block(histogram, block_size=(2, 2), epsilon=1e-7):
    block_height, block_width = block_size
    norm = np.zeros((histogram.shape[0] - block_height + 1, histogram.shape[1] - block_width + 1))
    for i in range(norm.shape[0]):
        for j in range(norm.shape[1]):
            hist_cells = histogram[i:i+block_height, j:j+block_width, :].flatten()
            norm[i, j] = np.sqrt(np.sum(hist_cells ** 2) + epsilon)
            histogram[i:i+block_height, j:j+block_width, :] /= norm[i, j]
    return histogram

def compute_hog(I, cell_size=(16, 16), block_size=(2, 2), num_bins=9):
    gx, gy = sobel_gradient(I)
    magnitude, orientation = magnitude_and_orientation(gx, gy)

    cell_height, cell_width = cell_size
    num_cells_x = int(I.shape[1] / cell_width)
    num_cells_y = int(I.shape[0] / cell_height)

    cell_hist = np.zeros((num_cells_y, num_cells_x, num_bins))
    for i in range(num_cells_y):
        for j in range(num_cells_x):
            # Extract the gradient magnitudes and orientations for the current cell
            cell_magnitudes = magnitude[i*cell_height:(i+1)*cell_height, 
                                        j*cell_width:(j+1)*cell_width]
            cell_orientations = orientation[i*cell_height:(i+1)*cell_height,
                                            j*cell_width:(j+1)*cell_width]

            # Compute the histogram of oriented gradients for the current cell
            cell_histogram = compute_cell_histogram(cell_magnitudes, cell_orientations, num_bins)

            # Store the histogram in the cells array
            cell_hist[i, j, :] = cell_histogram
    cell_hist = normalize_block(cell_hist, block_size)

    # Reshape the histogram into a feature vector
    hog_feature = cell_hist.reshape(-1)
    return hog_feature