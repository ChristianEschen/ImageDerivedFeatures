import numpy as np
import matplotlib.pyplot as plt

def create_2dp1_data():
    # Path: simulate_2dp1d_data.py
    # Function: create_2dp1_data
    # Purpose: This function creates a 2D+1 dataset
    # Input: None
    # Output: data
    data = np.zeros((10, 10, 12))
    data[9, 9, -0] = 1
    
    # draw a aline vertical in the second frame
    data[1:5, 2, 1] = 1
    # draw a longer line the the third frame
    data[1:7, 2, 2] = 1
    # draw a branch in the fourth frame
    data[1:7, 3, 3] = 1
    data[5, 3:6, 3] = 1
    
    # draw a branch in the fifth frame
    data[1:7, 3, 4] = 1
    data[5, 3:6, 4] = 1
    data[1:7, 2, 4] = 1
    data[5, 2:6, 4] = 1
    # draw a branch in the sixth frame
    data[1:7, 2, 5] = 1
    data[5, 2:6, 5] = 1
    
    # draw a branch in the seventh frame
    data[1:7, 3, 6] = 1
    
    # draw a branch in the eighth frame
    
    data[1:5, 3, 7] = 1
    
    data[0, 0, -1] = 1
    
    return data



def plot_2dp1_data(data):
    # Path: simulate_2dp1d_data.py
    # Function: plot_2dp1_data
    # Purpose: This function plots a 2D+1 dataset
    # Input: data
    # Output: None
    for i in range(data.shape[2]):
        plt.figure()
      #  plt.imshow(data[:, :, i])
        #plt.imshow(data[:, :, i], cmap='gray', interpolation='nearest')
        arr = data[:, :, i]
        plt.imshow(arr, cmap='gray', interpolation='nearest', aspect='auto', extent=[0, arr.shape[1], 0, arr.shape[0]], origin='upper')

        # Set the x and y axis limits to match the size of the binary mask
        plt.xlim(0, arr.shape[1])
        plt.ylim(0, arr.shape[0])

        # Plot a grid of 1x1 pixels on top of the binary mask
        plt.grid(visible=True, linewidth=1, color='black', which='both', axis='both', linestyle='-', alpha=1)
        plt.show()
    

if __name__ == '__main__':
    data = create_2dp1_data()
    plot_2dp1_data(data)