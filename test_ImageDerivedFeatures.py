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

if __name__ == '__main__':
    test_max_contrast()
    test_max_contrast_frame()
    test_init_contrast()
    test_total_contrast()
    test_total_contrast_pr_time()
    test_perimeter_at_max_contrast()
    test_perimeter_area_ratio_max_contrast()
    test_perimeter_all_frames_sum_pr_time()
    
    
    
    