import numpy as np
from importance import *
from utils import *

test_val_1 = np.array([[15,16,17], [15,16,17], [15,16,17], [15,16,17]])
expected_output_1 = np.zeros(test_val_1.shape)

test_val_2 = np.array([[15,0,17], [15,16,17], [15,16,17], [15,16,17]])
expected_output_2 = np.zeros(test_val_1.shape)
expected_output_2[0,1] = 1.0

def test_interval_overlap():
    assert intervals_overlap(1, 5, 1, 5)
    assert intervals_overlap(1, 5, 0, 5)
    assert intervals_overlap(1, 5, 0, 6)
    assert intervals_overlap(1, 5, 2, 4)
    assert not intervals_overlap(1, 5, 6, 7)
    assert not intervals_overlap(6, 7, 1,5)

def test_confidence_interval():
    assert not confidence_interval_overlap([1,1,1],[1.5,1.5,1.5], .1, std_to_use = .5)
    assert confidence_interval_overlap([1,1,1],[1.5,1.5,1.5], .5, std_to_use = .5)


def test_cluster():
    pass

def test_set_get():
    u = ImportanceMatrix(test_val_1)
    u[0,4] = 5.0
    assert u[0,4] == 5.0

def test_proportion_within():
    assert count_within_threshold_importance(0, [0,1,1,1], 0) == 1.0
    assert count_within_threshold_importance(1, [0,1,1,1], 0) == 0.25
    assert count_within_threshold_importance(0, [0,1,1,1], 1) == 0.0

def test_normal_cdf():
    assert normal_cdf(0,[16,15,15,15],threshold=.05) == 1.0
    assert normal_cdf(0,[16,15,15,15],threshold=.05) == 0.0
    assert normal_cdf(0,[16,15,15,15]) == 1.0
    assert normal_cdf(0,[16,15]) == .5

