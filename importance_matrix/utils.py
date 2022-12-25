import numpy as np
from statistics import NormalDist

delta = 10**-14

def scale_to_max_min(array : np.ndarray,  max_val : float, min_val : float) -> np.ndarray:
    if max(array) == min(array):
        return (max_val + min_val)/2 + np.zeros(array.shape)
    m = (max_val - min_val) / (max(array) - min(array))
    b = min_val - m * min(array)
    return m * array + b

def intervals_overlap(min1 : float, max1 : float, min2 : float, max2 : float)-> bool:
        return (min2 <=min1 and  min1<=max2) or (min2 <= max1 and  max1<=max2) or \
            (min1 <=min2 and  min2<=max1) or (min1 <= max2 and  max2<=max1) 

def confidence_interval_overlap(array1 : np.ndarray, array2 : np.ndarray,\
     p : float, std_to_use : float = None) -> bool:
    if std_to_use == None:
        norm1 = NormalDist(np.mean(array1), max(np.std(array1), delta))
        norm2 = NormalDist(np.mean(array2),  max(np.std(array1), delta))
    else:
        norm1 = NormalDist(np.mean(array1),  max(std_to_use, delta))
        norm2 = NormalDist(np.mean(array2),  max(std_to_use, delta))
    return intervals_overlap(norm1.inv_cdf(p),norm1.inv_cdf(1-p), norm2.inv_cdf(p),norm2.inv_cdf(1-p),)