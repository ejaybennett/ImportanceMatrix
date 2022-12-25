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
    assert not confidence_interval_overlap([1,1,1],[1.5,1.5,1.5], .7, std_to_use = .5)
    assert confidence_interval_overlap([1,1,1],[1.5,1.5,1.5], .3, std_to_use = .5)


def test_cluster():
    val1 = cluster([15.9, 16, 16,16, 16, 17], p_value = .05)
    val2 = cluster([15.9, 16, 16,16, 16, 17], p_value = .5)
    assert([17] in val1 and [15.9, 16,16,16,16] in val1)
    assert([15.9] in val2 and [17] in val2 and [16,16,16,16] in val2)

def test_determine_importance():
    assert all(determine_importance(np.ndarray([1,1,1,2]),.05) == np.ndarray([0,0,0,1]))

def test_set_get():
    u = ImportanceMatrix(test_val_1)
    u[0,2] = 5.0
    assert u[0,2] == 5.0

def test_proportion_within():
    assert proportion_within(0, [0,1,1,1], 0) == 1.0
    assert abs(proportion_within(1, [0,1,1,1], 0) - .3333)<.001
    assert proportion_within(0, [0,1,1,1], 1) == 0.0

def test_normal_cdf():
    print(normal_cdf(0,np.ndarray([16,15,15,15]),threshold=.05))
    print(normal_cdf(0,np.ndarray([16,15,15,15]),threshold=.05))
    print(normal_cdf(0,np.ndarray([16,15,15,15])))
    print(normal_cdf(0,np.ndarray([16,15])))
    assert normal_cdf(0,np.ndarray([16,15,15,15]),threshold=.05) == 1.0
    assert normal_cdf(0,np.ndarray([16,15,15,15]),threshold=.05) == 0.0
    assert normal_cdf(0,np.ndarray([16,15,15,15])) == 1.0
    assert normal_cdf(0,np.ndarray([16,15])) == .5

