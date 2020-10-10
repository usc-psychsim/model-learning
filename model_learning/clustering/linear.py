import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
from model_learning.algorithms.max_entropy import THETA_STR
from model_learning.util.plot import format_and_save_plot

__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'


def cluster_linear_rewards(results, linkage, dist_threshold):
    """
    Performs hierarchical agglomerative clustering of a group of linear reward functions found through IRL.
    :param list[ModelLearningResult] results: a list of linear IRL results used for clustering.
    :param str linkage: the linkage criterion of clustering algorithm.
    :param float dist_threshold: the distance above which the clusters are not joined (determines final number of clusters).
    :rtype: (AgglomerativeClustering, np.ndarray)
    :return: a tuple containing the agglomerative clustering algorithm fit to the reward weight vectors, and the
    an array containing all the reward weight vectors).
    """
    # performs clustering of all reward weights
    thetas = np.array([result.stats[THETA_STR] for result in results])
    clustering = AgglomerativeClustering(
        n_clusters=None, linkage=linkage, distance_threshold=dist_threshold)
    clustering.fit(thetas)

    return clustering, thetas


def get_clusters_means(clustering, thetas):
    """
    Get clusters' mean weight vectors and standard deviations.
    :param AgglomerativeClustering clustering: the clustering algorithm with the resulting labels/indexes.
    :param np.ndarray thetas: an array containing the reward weight vectors, of shape (num_points, rwd_vector_size).
    :rtype: (dict[str, list[int]], dict[str, np.ndarray])
    :return: a tuple with a dictionary containing the list of indexes of datapoints in each cluster and a dictionary
    containing an array of shape (2, rwd_vector_size) containing the mean and std_dev of the reward vector for each
    cluster.
    """
    # gets clusters
    clusters = {}
    for idx, cluster in enumerate(clustering.labels_):
        if cluster not in clusters:
            clusters[cluster] = []
        clusters[cluster].append(idx)

    # mean weights within each cluster
    cluster_weights = {}
    for cluster in sorted(clusters):
        idxs = clusters[cluster]
        cluster_weights[cluster] = np.array([np.mean(thetas[idxs], axis=0), np.std(thetas[idxs], axis=0)])

    return clusters, cluster_weights


def save_mean_cluster_weights(cluster_weights, file_path, rwd_feat_names):
    """
    Saves the clusters' mean reward vectors to a CSV file.
    :param cluster_weights: a dictionary containing an array of shape (2, `rwd_vector_size`) containing the mean and
    std_dev of the reward vector for each cluster.
    :param str file_path: the path to the CSV file in which to save the reward vector means.
    :param list[str] rwd_feat_names: the names of each reward feature, of length `rwd_vector_size`.
    :return:
    """
    # file with cluster weights
    with open(file_path, 'w') as f:
        write = csv.writer(f)
        write.writerow(['Cluster'] + rwd_feat_names)
        for cluster in sorted(cluster_weights):
            write.writerow([cluster] + cluster_weights[cluster][0].tolist())


def save_clusters_info(clustering, extra_info, thetas, file_path, rwd_feat_names):
    """
    Saves the clusters' datapoint information, including extra information about each datapoint.
    :param AgglomerativeClustering clustering: the clustering algorithm with the resulting labels/indexes.
    :param dict[str, list] extra_info: a dictionary containing extra information about the datapoints, where keys are
    the labels for the information and the values are lists containing the info for each datapoint.
    :param np.ndarray thetas: an array containing the reward weight vectors, of shape (num_points, rwd_vector_size).
    :param str file_path: the path to the CSV file in which to save the clusters' info.
    :param list[str] rwd_feat_names: the names of each reward feature, of length `rwd_vector_size`.
    :return:
    """
    # file with cluster contents
    with open(file_path, 'w') as f:
        write = csv.writer(f)
        write.writerow(['Cluster'] + list(extra_info.keys()) + rwd_feat_names)
        write.writerows(list(zip(clustering.labels_, *list(extra_info.values()), *thetas.T.tolist())))


def plot_clustering_distances(clustering, file_path):
    """
    Saves a plot with the clustering distances resulting from the given clustering algorithm.
    :param AgglomerativeClustering clustering: the clustering algorithm with the resulting distances.
    :param str file_path: the path to the file in which to save the plot.
    :return:
    """
    # distances plot
    plt.figure()
    plt.plot(np.hstack(([0], clustering.distances_)))
    plt.xlim([0, len(clustering.distances_)])
    plt.ylim(ymin=0)
    plt.xticks(np.arange(len(clustering.distances_) + 1), np.flip(np.arange(len(clustering.distances_) + 1) + 1))
    plt.axvline(x=len(clustering.distances_) - clustering.n_clusters_ + 1, c='red', ls='--', lw=0.6)
    format_and_save_plot(plt.gca(), 'Reward Weights Clustering Distance', file_path,
                         x_label='Num. Clusters', show_legend=False)


def plot_clustering_dendrogram(clustering, file_path, dist_threshold, labels):
    """
    Saves a dendrogram plot with the clustering resulting from the given model.
    :param AgglomerativeClustering clustering: the clustering algorithm with the resulting labels and distances.
    :param str file_path: the path to the file in which to save the plot.
    :param float dist_threshold: the distance above which the clusters are not joined (determines final number of clusters).
    :param list[str] labels: a list containing a label for each clustering datapoint.
    :return:
    """
    linkage_matrix = get_linkage_matrix(clustering)
    dendrogram(linkage_matrix, clustering.n_clusters_, 'level', labels=labels, leaf_rotation=45, leaf_font_size=8)
    plt.axhline(y=dist_threshold, c='red', ls='--', lw=0.6)
    format_and_save_plot(plt.gca(), 'Reward Weights Clustering Dendrogram', file_path, show_legend=False)


def get_linkage_matrix(clustering):
    """
    Gets a linkage matrix from the `sklearn` clustering model.
    See: https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html
    :param AgglomerativeClustering clustering: the clustering model.
    :return:
    """
    # create the counts of samples under each node
    counts = np.zeros(clustering.children_.shape[0])
    n_samples = len(clustering.labels_)
    for i, merge in enumerate(clustering.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    return np.column_stack([clustering.children_, clustering.distances_, counts]).astype(float)
