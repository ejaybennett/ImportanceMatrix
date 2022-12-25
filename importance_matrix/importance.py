import numpy as np
from statistics import NormalDist
from utils import confidence_interval_overlap, scale_to_max_min
gamma = .00000001


#Approach: Cluster nearby values together, then use a CI to add any additional points. 
#We will start with the initial standard deviation of each value as p*standard deviation
#Clusters will be combined if the values in their 1-p confidence interval overlap
#p is approximately the probability required to consider two clusters seperate
#IE, if p=.05, then two clusters will be considered seperate thier 95% confidence intervals
#overlap
def cluster_normal(array : np.ndarray, p_value : float):
    clusters = []
    starting_STD = np.std(array)*(1-p_value)/2
    #First iteration- add values within pvalue*std of eachother to the same cluster
    for a in array:
        found_cluster = False
        for cluster in clusters:
            if confidence_interval_overlap(cluster, [a], p_value, std_to_use=starting_STD):
                found_cluster = True
                cluster.append(a)
                break
        if not(found_cluster):
            clusters.append([a])
    changed = None
    #scecond iteration: combine clusters whose 1-p% confidence intervals overlap until no changes have been made
    while changed==None or changed == True:
        i = 0
        changed = False
        while i < len(clusters):
            j = i+1
            while j < len(clusters):
                if(confidence_interval_overlap(clusters[i], clusters[j], p_value)):
                    changed = True
                    clusters[i] = clusters[i] + clusters.pop(j)
                j += 1
            i += 1
    return clusters


    

def determine_importance(array: np.ndarray, threshold):
    #the smaller the cluster you are in and the farther away from the center of teh cluster,
    #the higher the importance score. 
    #If all values are equal, return 0
    if len(array) ==0 or (array==array[0]).all():
        return np.zeros(array.shape)
    
    clusters = cluster_normal(array, threshold)
    #We find the the smallest possible seperations between clusters
    #so we can rank based on z-score within a cluster
    scale_size = 1/(max([len(c) for c in clusters]))- 1/(max([len(c) for c in clusters])+1)
    #Then scale within a cluster on the absolute value of the z-score
    scores = {}
    for c in clusters:
        cluster = np.array(c)
        if (cluster == cluster[0]).all():
            z_scores = np.zeros(cluster.shape)
        else:
            z_scores = abs((cluster-np.mean(cluster)))/np.std(cluster)
        scaled_cluster_scores = scale_to_max_min(z_scores, 1/len(cluster)+scale_size,\
              1/len(cluster)-scale_size)
        for a, score in zip(cluster, scaled_cluster_scores):
            scores[a] = score
    return scale_to_max_min(np.array([scores[a] for a in array]),1,0)
  

def proportion_distances(array : np.ndarray, threshold=0):
    total_distance = []
    distances = np.ndarray([[abs(a-b) for b in array] for a in array])
    distances[distances<threshold] = 0
    total_distances = np.sqeeze(np.apply_along_axis(sum, 1))
    if total_distances[0] == 0:
        return np.zeros(array.shape)
    else:
        scale_to_max_min(total_distances, 1, 0)


def proportion_within(array : np.ndarray, threshold=0):
    if (array == array[0]).all():
        return np.zeros(array.shape)
    count = 0
    out = np.zeros(array.shape)
    for i in range(len(array)):
        out[i] = len(abs(array-array[i]) <= threshold)
    return out

def normal_cdf(idx : int, array : np.ndarray, threshold=0) -> float:
    """Only a useful measure for normally distributed (or at least continous and unimodal) data"""
    if len(array) < 2:
        return 0
    elif len(array) == 2:
        return float(not array[0] == array[1]) 
    most_extreme = max(array, key=lambda x: abs(x-np.mean(array)) )
    rest = array[abs(array-most_extreme)>threshold]
    if len(rest)==0:
        return 0.0
    mean = np.mean(rest)
    stdev_sample = np.std(rest, ddof=1)
    if stdev_sample < gamma:
        return float(array[idx] != mean)
    zscore = (array[idx]-mean)/stdev_sample
    cdf = 2*NormalDist().cdf(-abs(zscore))
    return cdf
    

class ImportanceMatrix:
    def __init__(self, input_data : np.ndarray, threshold : float = 0.05, \
            importance_function : "( np.array, float) -> float" =  normal_cdf):
        """ """
        if(not(isinstance(input_data, np.ndarray)) or len(input_data.shape) != 2 or \
            input_data.shape[0] <= 1 or input_data.shape[1] < 1):
            raise ValueError("input_data should be a 2 dimensional np.ndarray")
        rows = input_data.shape[0]
        cols = input_data.shape[1]
        self.threshold = threshold
        self.output_data = np.apply_along_axis(,1)
        # self.output_data = np.zeros(input_data.shape)
        # for c in range(cols):
        #     for r in range(rows):
        #         self.output_data[r,c] = importance_function(r, np.squeeze(input_data[:,c]), threshold)
    def __getitem__(self, key):
        return self.output_data.__getitem__(key)
    def __setitem__(self, key, value):
        return self.output_data.__setitem__(key, value)
    def __iter__(self):
        return self.output_data.__iter__()

print(scale_to_max_min(np.array([1,1,1,1]), 1, 0))

print(determine_importance(np.array([1,1,1,1,1,1,1,2, 3]),.05))

#cluster([15.1, 16, 16,16, 16, 17], p_value = .5)