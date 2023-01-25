from dataclasses import dataclass
import nibabel as nib
import numpy as np
import os
from skimage.morphology import skeletonize, disk
from skimage.measure import label
import sknw
import matplotlib.pyplot as plt
import scipy

@dataclass
class IDP_2D_plus_t:
    root_path: str
    relative_path: str
    pixel_size_x: float
    pixel_size_y: float
    pixel_size_z: float
    output_path: str

    def __post_init__(self):
        self.seg_path = os.path.join(self.root_path, self.relative_path)

    def load_nib_image(self, root_path):
        # Path: ImageDerivedFeatures.py
        # Function: load_nib_image
        # Purpose: This function loads a nibabel image
        # Input: image
        # Output: image_data
        image_data = nib.load(self.seg_path)
        data = image_data.get_fdata()
        return data

    def get_sum_contrast_pr_frame(self, seg_data):
        contrast_vector = np.sum(np.sum(seg_data, axis=1), axis=0)
        return contrast_vector

    def max_contrast(self, seg_data):
        # Path: ImageDerivedFeatures.py
        # Function: max_contrast
        # Purpose: This function calculates the maximum contrast of an image
        # Input: image
        # Output: max_contrast
        max_c = np.max(self.get_sum_contrast_pr_frame(seg_data))
        max_c = max_c * self.pixel_size_x * self.pixel_size_y
        return max_c
    
    def perimeter_area_ratio_max_contrast(self, seg_data):
        max_c = self.max_contrast(seg_data)
        perimeter_max_c = self.perimeter_at_max_contrast(seg_data)
        ratio = perimeter_max_c / max_c
        return ratio
        
    def perimeter_at_max_contrast(self, seg_data):
        frame = self.max_contrast_frame(seg_data)
        data = seg_data[:, :, frame]
        perimeter_max_c = self.calculate_perimeter(data)
        return perimeter_max_c

    def calculate_perimeter(self, seg_data):
        # Add a border of zeros around the array
        padded_arr = np.pad(seg_data, [(1, 1), (1, 1)], mode='constant')

        perimeter = np.sum(
            padded_arr[:, 1:] != padded_arr[:, :-1]) \
            + np.sum(padded_arr[1:, :] != padded_arr[:-1, :])
        return perimeter
    
    def max_contrast_frame(self, seg_data):
        # Path: ImageDerivedFeatures.py
        # Function: max_contrast_frame
        # Purpose: This function calculates the maximum contrast of an image
        # Input: image
        # Output: max_contrast
        max_c = np.argmax(self.get_sum_contrast_pr_frame(seg_data))
        return max_c
    
    def init_contrast(self, seg_data, init_blood_vessel_threshold=48):
        # Path: ImageDerivedFeatures.py
        # Function: init_contrast
        # Purpose: This function calculates the initimum contrast of an image
        # Input: image
        # Output: init_contrast
        contrast_sum_vector = self.get_sum_contrast_pr_frame(seg_data)
        contrast_sum_vector_mm = contrast_sum_vector * self.pixel_size_x * \
            self.pixel_size_y
        # Threshold the contrast vector
        # contrast_sum_vector_mm_copy = contrast_sum_vector_mm.copy()
        contrast_sum_vector_mm, index = self.remove_small_objects(
            contrast_sum_vector_mm, init_blood_vessel_threshold)
       #init_c = np.init(contrast_sum_vector_mm)
        return contrast_sum_vector_mm[0]

    def remove_small_objects(self, contrast_sum_vector_mm, init_blood_vessel_threshold=48):
        # Threshold the contrast vector
        # contrast_sum_vector_mm_copy = contrast_sum_vector_mm.copy()
        contrast_sum_vector_mm[contrast_sum_vector_mm < init_blood_vessel_threshold] = 0
        index = np.where(contrast_sum_vector_mm != 0)[0]
        contrast_sum_vector_mm = contrast_sum_vector_mm[contrast_sum_vector_mm != 0]
        
        return contrast_sum_vector_mm, index
    
    def total_contrast(self, seg_data, init_blood_vessel_threshold=48):
        # Path: ImageDerivedFeatures.py
        # Function: total_contrast
        # Purpose: This function calculates the total contrast of an image
        # Input: image
        # Output: total_contrast
        contrast_sum_vector = self.get_sum_contrast_pr_frame(seg_data)
        contrast_sum_vector_mm = contrast_sum_vector * self.pixel_size_x * self.pixel_size_y
        contrast_sum_vector_mm, _ = self.remove_small_objects(contrast_sum_vector_mm, init_blood_vessel_threshold)
        
        total_c = np.sum(contrast_sum_vector_mm)
        frames_with_contrast = len(contrast_sum_vector_mm)
        return total_c, frames_with_contrast
    
    def total_contrast_pr_time(self, seg_data, init_blood_vessel_threshold=48):
        # Path: ImageDerivedFeatures.py
        # Function: total_contrast_pr_time
        # Purpose: This function calculates the total contrast of an image
        # Input: image
        # Output: total_contrast_pr_time
        total_contrast, frames_with_contrast = self.total_contrast(seg_data, init_blood_vessel_threshold)
        total_c_pr_time = \
            total_contrast / (frames_with_contrast * self.pixel_size_z)
        return total_c_pr_time

    def perimeter_all_frames(self, seg_data, init_blood_vessel_threshold=48):
        # Path: ImageDerivedFeatures.py
        # Function: perimeter_all_frames
        # Purpose: This function calculates the perimeter of an image
        # Input: image
        # Output: perimeter_all_frames
        
        # first remove small objects with less than 48 pixels
        contrast_sum_vector = self.get_sum_contrast_pr_frame(seg_data)
        contrast_sum_vector_mm = contrast_sum_vector * self.pixel_size_x * self.pixel_size_y
        _, index= self.remove_small_objects(contrast_sum_vector_mm, init_blood_vessel_threshold)
        seg_data = seg_data[:, :, index]
        
        
        
        
        perimeter_all_frames = []
        for frame in range(seg_data.shape[2]):
            data = seg_data[:, :, frame]
            perimeter_all_frames.append(self.calculate_perimeter(data))
            
        return perimeter_all_frames
    
    def sum_perimeter_all_frames_pr_time(self, seg_data, init_blood_vessel_threshold=48):
        # Path: ImageDerivedFeatures.py
        # Function: perimeter_all_frames_pr_time
        # Purpose: This function calculates the perimeter of an image
        # Input: image
        # Output: perimeter_all_frames_pr_time
        
        perimeter_all_frames = self.perimeter_all_frames(seg_data, init_blood_vessel_threshold)
        perimeter_all_frames_pr_time = np.sum(np.array(perimeter_all_frames)) / self.pixel_size_z
        return perimeter_all_frames_pr_time
  
    def ratio_perimeter_all_frames_pr_time_to_total_contrast_pr_time(self, seg_data, init_blood_vessel_threshold=48):
        # Path: ImageDerivedFeatures.py
        # Function: ratio_perimeter_all_frames_pr_time_to_total_contrast_pr_time
        # Purpose: This function calculates the ratio of perimeter to total contrast
        # Input: image
        # Output: ratio_perimeter_all_frames_pr_time_to_total_contrast_pr_time
        perimeter_all_frames_pr_time = self.perimeter_all_frames_pr_time(seg_data, init_blood_vessel_threshold)
        total_c_pr_time = self.total_contrast_pr_time(seg_data, init_blood_vessel_threshold)
        ratio_perimeter_all_frames_pr_time_to_total_contrast_pr_time = perimeter_all_frames_pr_time / total_c_pr_time
        return ratio_perimeter_all_frames_pr_time_to_total_contrast_pr_time

    def endpoints(self, seg_data):
        skeletonized = skeletonize(seg_data)
        # Find row and column locations that are non-zero
        (rows,cols) = np.nonzero(skeletonized)

        # Initialize empty list of co-ordinates
        x_skel_coords = []
        y_skell_coords = []

        # For each non-zero pixel...
        for (r,c) in zip(rows,cols):

            # Extract an 8-connected neighbourhood
            (col_neigh,row_neigh) = np.meshgrid(np.array([c-1,c,c+1]), np.array([r-1,r,r+1]))

            # Cast to int to index into image
            col_neigh = col_neigh.astype('int')
            row_neigh = row_neigh.astype('int')

            # Convert into a single 1D array and check for non-zero locations
            pix_neighbourhood = skeletonized[row_neigh,col_neigh].ravel() != 0

            # If the number of non-zero locations equals 2, add this to 
            # our list of co-ordinates
            if np.sum(pix_neighbourhood) == 2:
                x_skel_coords.append(r)
                y_skell_coords.append(c)
        
        # plt.imshow(seg_data, cmap='gray')
        # plt.colorbar()
        # plt.plot(y_skell_coords, x_skel_coords, 'r.')
        # plt.title("Branch graph")
        # plt.axis('off')
        # plt.show()
        return len(x_skel_coords)
    
    def endpoints_all_frames(self, seg_data, init_blood_vessel_threshold=48):
        # Path: ImageDerivedFeatures.py
        # Function: endpoints_all_frames
        # Purpose: This function calculates the endpoints of an image
        # Input: image
        # Output: endpoints_all_frames
        # first remove small objects
        contrast_sum_vector = self.get_sum_contrast_pr_frame(seg_data)
        contrast_sum_vector_mm = contrast_sum_vector * \
            self.pixel_size_x * self.pixel_size_y
        _, index = self.remove_small_objects(
            contrast_sum_vector_mm, init_blood_vessel_threshold)
        seg_data = seg_data[:, :, index]
        endpoints_all_frames = []
        for frame in range(seg_data.shape[2]):
            data = seg_data[:, :, frame]
            endpoints_all_frames.append(self.endpoints(data))
        return endpoints_all_frames

    def ratio_endpoints_max_contrast_at_max_contrast(self, seg_data):
        max_contrast = self.max_contrast(seg_data)
        max_contrast_frame = self.max_contrast_frame(seg_data)
        endpoints_max_contrast_at_max_contrast = self.endpoints(seg_data[:, :, max_contrast_frame])
        ratio_endpoints_max_contrast_at_max_contrast = endpoints_max_contrast_at_max_contrast / max_contrast
        return ratio_endpoints_max_contrast_at_max_contrast
    
    def mean_endpoints_pr_total_contrast(self, seg_data, threshold=48):
        
        endpoints_all_frames = self.endpoints_all_frames(seg_data, init_blood_vessel_threshold=threshold)
        mean_endpoints = np.mean(endpoints_all_frames)
        total_contrast, _ = self.total_contrast(seg_data, init_blood_vessel_threshold=threshold)
        endpoints_pr_total_contrast = mean_endpoints / total_contrast
        return endpoints_pr_total_contrast
        
    def time_to_max_contrast(self, seg_data, init_blood_vessel_threshold=48):
        # Path: ImageDerivedFeatures.py
        # Function: time_to_max_contrast
        # Purpose: This function calculates the time to max contrast
        # Input: image
        # Output: time_to_max_contrast
        contrast_sum_vector = self.get_sum_contrast_pr_frame(seg_data)
        contrast_sum_vector_mm = contrast_sum_vector * self.pixel_size_x * \
            self.pixel_size_y
        # Threshold the contrast vector
        # contrast_sum_vector_mm_copy = contrast_sum_vector_mm.copy()
        contrast_sum_vector_mm, index = self.remove_small_objects(
            contrast_sum_vector_mm, init_blood_vessel_threshold)
        max_contrast_frame = self.max_contrast_frame(seg_data)
        nr_frames_loading_arteries = max_contrast_frame - min(index)
        estimated_time_to_max_contrast = nr_frames_loading_arteries * self.pixel_size_z
        return estimated_time_to_max_contrast
    
    def max_endpoints_all_frames(self, seg_data):
        endpoints_all_frames = self.endpoints_all_frames(seg_data)
        max_endpoints_all_frames = np.max(endpoints_all_frames)
        return max_endpoints_all_frames
    
    def time_to_max_endpoints(self, seg_data, init_blood_vessel_threshold=48):
        # first get index with init contrast
        contrast_sum_vector = self.get_sum_contrast_pr_frame(seg_data)
        contrast_sum_vector_mm = contrast_sum_vector * self.pixel_size_x * \
            self.pixel_size_y
        # Threshold the contrast vector
        # contrast_sum_vector_mm_copy = contrast_sum_vector_mm.copy()
        contrast_sum_vector_mm, index = self.remove_small_objects(
            contrast_sum_vector_mm, init_blood_vessel_threshold)
        start = min(index)
        
        # get index with max endpoints
        endpoints_all_frames = self.endpoints_all_frames(seg_data, init_blood_vessel_threshold=init_blood_vessel_threshold)
        max_endpoints_frame = np.argmax(endpoints_all_frames)
        
        # calculate time to max endpoints
        time_to_max_endpoints = (max_endpoints_frame - start) * self.pixel_size_z
        return time_to_max_endpoints
    
    def closeness_of_arteries(self, seg_data):
        #skeleton_h = skeletonize(seg_data)
        skeleton_h = seg_data
        disk_ = disk(1)
        convolution_h = scipy.signal.convolve2d(skeleton_h, disk_, mode='same')
        out_h = skeleton_h * convolution_h
        # plt.imshow(out_h, cmap='magma')
        # plt.axis('off')
        # plt.colorbar()
        # plt.show()
        return out_h
        
        
    def closeness_of_arteries_all_frames(self, seg_data):
        closeness_of_arteries_all_frames = []
        for frame in range(seg_data.shape[2]):
            data = seg_data[:, :, frame]
            closeness_of_arteries_all_frames.append(self.closeness_of_arteries(data))
        return closeness_of_arteries_all_frames
    
    def sum_closeness_arteries_at_max_contrast(self, seg_data):
        max_contrast_frame = self.max_contrast_frame(seg_data)
        closeness_arteries_at_max_contrast = self.closeness_of_arteries(seg_data[:, :, max_contrast_frame])
        closeness_arteries_at_max_contrast_mm = closeness_arteries_at_max_contrast * self.pixel_size_x * self.pixel_size_y
        sum_closeness = np.sum(closeness_arteries_at_max_contrast_mm)
        return sum_closeness
       
       
    def closeness_of_arteries_pr_time(self, seg_data, init_blood_vessel_threshold=48):
        contrast_sum_vector = self.get_sum_contrast_pr_frame(seg_data)
        contrast_sum_vector_mm = contrast_sum_vector * self.pixel_size_x * \
            self.pixel_size_y
        # Threshold the contrast vector
        # contrast_sum_vector_mm_copy = contrast_sum_vector_mm.copy()
        _, index = self.remove_small_objects(
            contrast_sum_vector_mm, init_blood_vessel_threshold)
        seg_data = seg_data[:, :, index[0]:]
        closeness_of_arteries_all_frames = self.closeness_of_arteries_all_frames(seg_data)
        sum_closseness_arteries_all_frames = np.sum(np.array(closeness_of_arteries_all_frames))
        sum_closseness_arteries_all_frames_mm = sum_closseness_arteries_all_frames * self.pixel_size_x * self.pixel_size_y
        sum_closseness_arteries_all_frames_mm_pr_time = sum_closseness_arteries_all_frames_mm / (seg_data.shape[2] * self.pixel_size_z)
        return sum_closseness_arteries_all_frames_mm_pr_time
        
        
        
        
        
        