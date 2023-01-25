import numpy as np
from simulate_2dp1d_data import create_2dp1_data
from ImageDerivedFeatures import IDP_2D_plus_t


def test_max_contrast():
    idp = IDP_2D_plus_t(
        "path/to/root", "path/to/relative", 0.5, 0.5, 1, "path/to/output")
    seg_data = create_2dp1_data()
    assert idp.max_contrast(seg_data) == 3.5


def test_max_contrast_frame():
    idp = IDP_2D_plus_t(
        "path/to/root", "path/to/relative", 0.5, 0.5, 1, "path/to/output")
    seg_data = create_2dp1_data()
    assert idp.max_contrast_frame(seg_data) == 4

def test_init_contrast():
    idp = IDP_2D_plus_t(
        "path/to/root", "path/to/relative", 4, 4, 1, "path/to/output")
    seg_data = create_2dp1_data()
    assert idp.init_contrast(seg_data) == 64
    

def test_total_contrast():
    idp = IDP_2D_plus_t(
        "path/to/root", "path/to/relative", 4, 4, 1, "path/to/output")
    seg_data = create_2dp1_data()
    total_contrast, _ = idp.total_contrast(seg_data, init_blood_vessel_threshold=48)
    assert total_contrast == 816
    
    idp = IDP_2D_plus_t(
        "path/to/root", "path/to/relative", 0.5, 1, 1, "path/to/output")
    seg_data = create_2dp1_data()
    total_contrast, _ = idp.total_contrast(seg_data, init_blood_vessel_threshold=2)
    assert total_contrast == 25.5 #26.5

def test_total_contrast_pr_time():
    idp = IDP_2D_plus_t(
        "path/to/root", "path/to/relative", 4, 4, 1, "path/to/output")
    seg_data = create_2dp1_data()
    total_contrast_pr_time = idp.total_contrast_pr_time(seg_data, init_blood_vessel_threshold=48)
    assert total_contrast_pr_time  == 116.57142857142857
    
    idp = IDP_2D_plus_t(
        "path/to/root", "path/to/relative", 0.5, 1, 0.5, "path/to/output")
    seg_data = create_2dp1_data()
    total_contrast_pr_time = idp.total_contrast_pr_time(seg_data, init_blood_vessel_threshold=2)

    assert total_contrast_pr_time == 7.285714285714286
    
    
    idp = IDP_2D_plus_t(
        "path/to/root", "path/to/relative", 0.5, 1, 2, "path/to/output")
    seg_data = create_2dp1_data()
    total_contrast_pr_time = idp.total_contrast_pr_time(seg_data, init_blood_vessel_threshold=2)

    assert total_contrast_pr_time == 1.8214285714285714

def test_perimeter_at_max_contrast():
    idp = IDP_2D_plus_t(
        "path/to/root", "path/to/relative", 1, 1, 1, "path/to/output")
    seg_data = create_2dp1_data()
    perimeter_at_max_contrast = idp.perimeter_at_max_contrast(seg_data)
    assert perimeter_at_max_contrast == 20
    
def test_perimeter_area_ratio_max_contrast():
    idp = IDP_2D_plus_t(
        "path/to/root", "path/to/relative", 1, 1, 1, "path/to/output")
    seg_data = create_2dp1_data()
    perimeter_area_ratio = idp.perimeter_area_ratio_max_contrast(seg_data)
    assert perimeter_area_ratio == 1.4285714285714286
    
def test_perimeter_all_frames_sum_pr_time():
    idp = IDP_2D_plus_t(
        "path/to/root", "path/to/relative", 4, 4, 1, "path/to/output")
    seg_data = create_2dp1_data()
    perimeter_all_frames_sum_pr_time = idp.sum_perimeter_all_frames_pr_time(seg_data)
    assert perimeter_all_frames_sum_pr_time == 106

def test_ratio_perimeter_all_frames_pr_time_to_to_total_contrast_pr_time():
    idp = IDP_2D_plus_t(
        "path/to/root", "path/to/relative", 4, 4, 1, "path/to/output")
    seg_data = create_2dp1_data()
    ratio_perimeter_all_frames_pr_time_to_to_total_contrast_pr_time = idp.ratio_perimeter_all_frames_pr_time_to_to_total_contrast_pr_time(seg_data)
    assert ratio_perimeter_all_frames_pr_time_to_to_total_contrast_pr_time == 0.9093137254901961

def test_endpoints_all_frames():
    idp = IDP_2D_plus_t(
        "path/to/root", "path/to/relative", 4, 4, 1, "path/to/output")
    seg_data = create_2dp1_data()
    endpoints = idp.endpoints_all_frames(seg_data)
    assert endpoints == [2, 2, 2, 2, 2, 2, 2]
    
def test_endpoints():
    idp = IDP_2D_plus_t(
        "path/to/root", "path/to/relative", 4, 4, 1, "path/to/output")
    
    binary_image = np.zeros((10, 10))
    binary_image[3, 3] = 1
    binary_image[2, 3] = 1
    binary_image[4, 3] = 1
    binary_image[1, 3] = 1
    binary_image[5, 3] = 1
    binary_image[0, 3] = 1
    binary_image[6, 3] = 1
    binary_image[3, 2] = 1
    binary_image[3, 4] = 1
    binary_image[3, 1] = 1
    binary_image[3, 5] = 1
    binary_image[3, 6] = 1
    binary_image[2, 6] = 1
    binary_image[1, 6] = 1
    binary_image[3, 7] = 1
    binary_image[3, 8] = 1
    
    nr_endpoints = idp.endpoints(binary_image)

    
    assert nr_endpoints == 5
    
    
def test_ratio_endpoints_max_contrast_at_max_contrast():
    idp = IDP_2D_plus_t(
        "path/to/root", "path/to/relative", 0.5, 0.5, 1, "path/to/output")
    seg_data = create_2dp1_data()
    assert idp.ratio_endpoints_max_contrast_at_max_contrast(seg_data) == 2/3.5

def test_mean_endpoints_pr_total_contrast():
    idp = IDP_2D_plus_t(
        "path/to/root", "path/to/relative", 4, 4, 1, "path/to/output")
    seg_data = create_2dp1_data()
    assert idp.mean_endpoints_pr_total_contrast(seg_data) == 2/816

def test_time_to_max_contrast():
    idp = IDP_2D_plus_t(
        "path/to/root", "path/to/relative", 4, 4, 0.5, "path/to/output")
    seg_data = create_2dp1_data()
    assert idp.time_to_max_contrast(seg_data) == 1.5
    
def test_time_to_max_endpoints():
    idp = IDP_2D_plus_t(
        "path/to/root", "path/to/relative", 4, 4, 0.5, "path/to/output")
    seg_data = create_2dp1_data()
    seg_data[3,0:8, 6] = 1
    assert idp.time_to_max_endpoints(seg_data) == 2
    
def test_closeness_of_arteries_all_frames():
    
    idp = IDP_2D_plus_t(
        "path/to/root", "path/to/relative", 4, 4, 0.5, "path/to/output")
    seg_data = create_2dp1_data()
    idp.closeness_of_arteries_all_frames(seg_data)
    
def test_sum_closeness_arteries_at_max_contrast():
    
    idp = IDP_2D_plus_t(
        "path/to/root", "path/to/relative", 4, 4, 0.5, "path/to/output")
    seg_data = create_2dp1_data()
    sum_closeness = idp.sum_closeness_arteries_at_max_contrast(seg_data)
    assert sum_closeness == 50.0 * 4 * 4
    
def test_closeness_of_arteries_pr_time():
    idp = IDP_2D_plus_t(
        "path/to/root", "path/to/relative", 4, 4, 0.5, "path/to/output")
    seg_data = create_2dp1_data()
    sum_closeness_pr_time = idp.closeness_of_arteries_pr_time(seg_data)
    assert sum_closeness_pr_time == 436.3636363636364

if __name__ == '__main__':
    test_max_contrast()
    test_max_contrast_frame() # helper function
    test_init_contrast() # helper function
    test_total_contrast()
    test_total_contrast_pr_time()
    test_perimeter_at_max_contrast()
    test_perimeter_area_ratio_max_contrast()
    test_perimeter_all_frames_sum_pr_time()
    test_endpoints_all_frames()
    test_endpoints() # helper function
    test_ratio_endpoints_max_contrast_at_max_contrast()
    test_mean_endpoints_pr_total_contrast()
    test_time_to_max_contrast()
    test_time_to_max_endpoints()
    test_closeness_of_arteries_all_frames() # helper function
    test_sum_closeness_arteries_at_max_contrast()
    test_closeness_of_arteries_pr_time()
    
    
    
    


    
    
    
    