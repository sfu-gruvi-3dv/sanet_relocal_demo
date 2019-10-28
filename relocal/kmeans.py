import numpy as np
import matplotlib.pyplot as plt
import torch

def k_means(samples, num_clusters, stop_epsion=1e-2, max_iter=100, verbose=False):
    """
    K-Mean Cluster with torch
    :param samples: samples, dim: (N, D), N is the number of samples, D is the dimension
    :param num_clusters: number of clusters, K
    :param stop_epsion: stop condition
    :param max_iter: allowed max iteration
    :return cluster_loc: center of each cluster, dim: (K, D)
    :return sample_cluster_index: cluster indices (LongTensor) for each sample points, dim: (N)
    """
    # cluster indices
    sample_cluster_index = torch.zeros(samples.shape[0], dtype=torch.long)

    # distance cache, dim: (num_clusters, num_samples)
    sample_cluster_dist = torch.zeros((num_clusters, samples.shape[0]))

    # Step 1: Random choose initial points as cluster center
    random_indices = torch.randperm(samples.shape[0])
    cluster_loc = samples[random_indices[:num_clusters], :]
    old_distance_var = -10000

    # Step 2:Iteration
    for itr in range(0, max_iter):

        # compute the distance towards the cluster center, you can use 'np.linalg.norm' to compute L2 distance
        for cluster_idx in range(0, num_clusters):
            sample_cluster_dist[cluster_idx, :] = torch.norm(samples - cluster_loc[cluster_idx, :], dim=1)

        # for each sample point, set the cluster center with minimal distance
        sample_cluster_index = torch.argmin(sample_cluster_dist, dim=0)

        # re-compute the distance by average the cluster sampled points
        for cluster_idx in range(0, num_clusters):
            cluster_loc[cluster_idx, :] = torch.mean(samples[sample_cluster_index == cluster_idx], dim=0)

        # compute total avg. distance variance
        sum_distance_var = 0.0
        for cluster_idx in range(0, num_clusters):
            sum_distance_var += torch.norm(
                samples[sample_cluster_index == cluster_idx] - cluster_loc[cluster_idx, :], dim=1).sum()
        avg_distance_var = sum_distance_var / num_clusters

        # check if the avg. distance variance has converged
        if abs(avg_distance_var - old_distance_var) < stop_epsion:
            break

        if verbose:
            print("[itr %d] avg. distance variance: %f" % (itr, avg_distance_var))
        old_distance_var = avg_distance_var

    return cluster_loc, sample_cluster_index


if __name__ == '__main__':

    # Load the sample points
    points = np.load('data/k-mean_samples.npy')
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    # Do K-Mean cluster
    points_tensor = torch.from_numpy(points)

    with torch.cuda.device(0):
        cluster_loc, cluster_indices = k_means(points_tensor.cuda(), num_clusters=6, verbose=True)
        cluster_loc = cluster_loc.cpu().numpy()
        cluster_indices = cluster_indices.cpu().numpy()

    # Draw the clusters
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'w']
    for cluster_idx in range(0, cluster_loc.shape[0]):
        sub_sample_set = points[cluster_indices == cluster_idx]
        plt.scatter(sub_sample_set[:, 0], sub_sample_set[:, 1], c=colors[cluster_idx], label='group %d' % cluster_idx)

    plt.scatter(cluster_loc[:, 0], cluster_loc[:, 1], c='k', label='center')
    plt.scatter(cluster_loc[:, 0], cluster_loc[:, 1], c='k', label='center')
    plt.legend()
    plt.grid(True)
    plt.title("K-Mean Cluster (%d centers)" % cluster_loc.shape[0])
    plt.show()