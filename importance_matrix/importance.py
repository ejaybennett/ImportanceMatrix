import numpy as np
from statistics import NormalDist
from utils import conf_interval_overlap, scale_to_max_min
import scipy.stats as stats

class ImportanceMatrix:
    """Creates a 2D matrix from given input data and ranks the importance of each value in
    distinguishing it from the rest of the column. To access this output, treat as a numpy 
    matrix or access the object attribute importance, a numpy matrix."""
    def __init__(self, input_data : np.ndarray, threshold : float = 0.05, \
            importance_function : "(np.ndarray, float) ->  np.ndarray" =  "count_within_threshold"):
        """Parameters:
            array                    2-D float array to rank "importance" of values by column
            threshold                threshold to pass to importance_function
            importance_function      function to be applied to each column to determine the importance
                                    Can be "count_within_threshold", "normal_dist_clustering", or a user-supplied 
                                    function taking in a 1D array and a float and returning a 1D ndarray.
            Use count_within_threshold with threshold=0 for discrete data and with a non-zero threshold
            if you want values within threshold of eachother to decrease eachothers importance,
            and normal_dist_clustering with a p-value if data occurs in clusters, with the p-value  representing
            the probability required to consider two clusters to represent different data. 
                                     """
        if(not(isinstance(input_data, np.ndarray)) or len(input_data.shape) != 2 or \
            input_data.shape[0] <= 1 or input_data.shape[1] < 1):
            raise ValueError("input_data should be a 2 dimensional np.ndarray")
        if importance_function == "count_within_threshold":
            self.importance_function = ImportanceMatrix.count_within_threshold_importance
        elif importance_function == "normal_dist_clustering":
            self.importance_function = ImportanceMatrix.gaussian_clusters_importance
        else:
            self.importance_function = importance_function
        rows = input_data.shape[0]
        cols = input_data.shape[1]
        self.threshold = threshold
        #modify the importance function to squeeze the incoming array and 
        #return 0 if all the values are the same
        column_function = lambda array : self.importance_function(np.squeeze(array), threshold) if \
            not (array==array[0]).all() else np.zeros(len(array))
        self.importance = np.apply_along_axis(column_function,0,input_data)
    
    def normal_dist_clustering(array : np.ndarray, p : float)-> list[list[float]]:
        """ Clustering algorithm based on making confidence intervals with normal distributions.
        First creates clusters by grouping values wihtin standard-deviation*(1-p)/2,
        then combining clusters which student's t-test returns a pvalue less than p for
        Parameters:
            array : ndarray     Array to be clustered
            p_value : float     1-p is the width of the confidence intervals used for clustering
        Returns
            list[list[float]]   where each item in the outer list is a cluster. All items in array
                                will be clustered
        """
        clusters = []
        starting_STD = np.std(array)*(1-p)/2
        #First iteration- add values within pvalue*std of eachother to the same cluster
        for a in array:
            found_cluster = False
            for cluster in clusters:
                if conf_interval_overlap(cluster, [a], p, std_to_use=starting_STD):
                    found_cluster = True
                    cluster.append(a)
                    break
            if not(found_cluster):
                clusters.append([a])
        changed = None
        #second iteration: combine clusters whose 1-p% confidence intervals overlap until no changes have been made
        while changed==None or changed == True:
            i = 0
            changed = False
            while i < len(clusters):
                j = i+1
                while j < len(clusters):
                    print(stats.ttest_ind( clusters[i], clusters[j]).pvalue)
                    if stats.ttest_ind( clusters[i], clusters[j]).pvalue < p:
                        changed = True
                        clusters[i] = clusters[i] + clusters.pop(j)
                    j += 1
                i += 1
        return clusters
    
    def count_within_threshold_importance(array : np.ndarray, threshold=0) -> np.ndarray:
        """Importance inverse proportionally to amount of array within threshold then rescaled between 0 and 1.
        Parameters:
            array       1-D float array to rank importance 
            threshold   distance to include values
        Returns
            ndarray     Importance values ranging [0,1] inclusive of items in array
        """
        count = 0
        out = np.zeros(array.shape)
        for i in range(len(array)):
            out[i] = 1-len(array[abs(array-array[i]) <= threshold])/len(array)
        return scale_to_max_min(out,1,0)
    
    def gaussian_clusters_importance(array: np.ndarray, threshold : float) -> np.ndarray:
        """ Rank values in array based on the size of a cluster they belong in
        and the distance to the center of their cluster. 
        Between clusters, importance values are higher for all values in a smaller cluster
        Within a cluster, importance values increase linearly with absolute value of z-score
        Importance values are then scaled to have the minimum importance value 0 and maximum 1. 
        Parameters:
            array           Array to be clustered
            p_value         1-p is the width of the confidence intervals used for clustering
        Returns
            ndarray         importance value for each value in the array between [0,1]
        """
        clusters = ImportanceMatrix.normal_dist_clustering(array, threshold)
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

    def __getitem__(self, key):
        return self.importance.__getitem__(key)
    def __setitem__(self, key, value):
        return self.importance.__setitem__(key, value)
    def __iter__(self):
        return self.importance.__iter__()
    def __repr__(self):
        return f"ImportanceMatrix: {self.importance}"

print(ImportanceMatrix.normal_dist_clustering([1,1,1,1,1,1.01,2,1.2,5], .9))