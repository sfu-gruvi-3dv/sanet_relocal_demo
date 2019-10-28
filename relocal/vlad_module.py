import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TripletRankingLoss(nn.Module):
    """ The triplet ranking loss
        [ref: https://arxiv.org/pdf/1511.07247.pdf]
    """
    def __init__(self, margin, triplet_config):
        super(TripletRankingLoss, self).__init__()
        self.margin = margin
        self.overlap_thres = triplet_config['overlap_thres']
        self.baseline_thres = triplet_config['baseline_thres']
        self.anchor_ref_idx = triplet_config['anchor_ref_idx']

    def forward(self, pred_embeddings, rel_overlap_ratio, rel_baseline_dist):
        """
        Compute triplet
        :param pred_embeddings:
        :param rel_overlap_ratio:
        :param rel_baseline_dist:
        :return:
        """

        N, W, X = pred_embeddings.shape

        # Compute distance between embeddings
        pairwise_dist = self._pairwise_distance(pred_embeddings.view(N*W, -1), squared=True)          # dim: (N*W, N*W)

        pairwise_dist = pairwise_dist.reshape(N, W, N * W)
        anchor2sample_dist = pairwise_dist[:, self.anchor_ref_idx, :].squeeze(dim=1)                  # dim: (N, N*W)

        # Compute the positive and negative distance
        anc_pos_dist = anchor2sample_dist.unsqueeze(dim=2)                                            # dim: (N, W, 1)
        anc_neg_dist = anchor2sample_dist.unsqueeze(dim=1)                                            # dim: (N, 1, W)

        # triplet_loss[i, j, k] will contain the triplet loss of anc=i (per batch), pos=j, neg=k
        # triplet loss defines as: max(pos - neg + margin, 0)
        loss = anc_pos_dist - anc_neg_dist + self.margin                                           # dim: (N, N*W, N*W)

        # select the valid triplet
        mask = self._mask_valid(rel_overlap_ratio, rel_baseline_dist, self.overlap_thres, self.baseline_thres, self.anchor_ref_idx)
        triplet_loss = loss * mask

        # Remove negative losses (i.e. the easy triplets)
        triplet_loss = F.relu(triplet_loss)

        # Count number of hard triplets (where triplet_loss > 0)
        hard_triplets = torch.gt(triplet_loss, 1e-16).float()
        num_hard_triplets = torch.sum(hard_triplets)

        triplet_loss = torch.sum(triplet_loss) / (num_hard_triplets + 1e-16)

        return triplet_loss

    @staticmethod
    def _mask_valid(rel_overlap_ratio: torch.tensor, rel_baseline_dist: torch.tensor, overlap_thres, baseline_thres, anchor_idx):
        """
        Mask out the invalid triplet
        :param rel_overlap_ratio: the relative overlap ratio array, dim: (N, W, W)
        :param rel_baseline_dist: the relative baseline array, dim: (N, W, W)
        :param overlap_thres:
        :param baseline_thres:
        :return: mask, dim: (N, N*W, N*W)
        """
        N, W = rel_overlap_ratio.shape[:2]

        # mask = (rel_overlap_ratio > overlap_thres) * (rel_baseline_dist < baseline_thres)           # dim: (N, W, W)
        # mask = mask.cpu().numpy()

        output = np.zeros((N, N*W, N*W), dtype=np.float32)
        for n in range(N):
            # select anchor index
            mask = (rel_overlap_ratio[n, anchor_idx, :] > overlap_thres) * (rel_baseline_dist[n, anchor_idx, :] < baseline_thres)

            if np.any(mask) == True:
                output_row_start_idx = n*W
                for row_idx in range(mask.shape[0]):
                    # mark all items in that row as valid triplet
                    if mask[row_idx] == True:
                        output[n, output_row_start_idx + row_idx, :] = 1.0

        return torch.tensor(output).cuda().detach()

    @staticmethod
    def _pairwise_distance(x, squared=False, eps=1e-16):
        """
        Compute the pairwise distance matrix:
            ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
        :param x: embedding vector
        :param squared: use square distance if needed
        :param eps:
        :return:
        """
        cor_mat = torch.matmul(x, x.t())                                            # dim: (N, N)
        norm_mat = cor_mat.diag()                                                   # dim: (N)

        # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
        distances = norm_mat.unsqueeze(1) - 2 * cor_mat + norm_mat.unsqueeze(0)
        distances = F.relu(distances)

        if not squared:
            # Because the gradient of sqrt is infinite when distances == 0.0 (ex: on the diagonal)
            # we need to add a small epsilon where distances == 0.0
            mask = torch.eq(distances, 0.0).float()
            distances = distances + mask * eps
            distances = torch.sqrt(distances)

            # Correct the epsilon added: set the distances on the mask to be exactly 0.0
            distances = distances * (1.0 - mask)

        return distances


class NetVLAD(nn.Module):
    """ NetVLAD layer
        [ref: https://arxiv.org/pdf/1511.07247.pdf]
    """

    def __init__(self, num_clusters=64, dim=128, alpha=100.0,
                 normalize_input=True):
        """
        Args:
            num_clusters : int
                The number of clusters
            dim : int
                Dimension of descriptors
            alpha : float
                Parameter of initialization. Larger value is harder assignment.
            normalize_input : bool
                If true, descriptor-wise L2 normalization is applied to input.
        """
        super(NetVLAD, self).__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        self.alpha = alpha
        self.normalize_input = normalize_input
        self.conv = nn.Conv2d(dim, num_clusters, kernel_size=(1, 1), bias=True)
        self.centroids = nn.Parameter(torch.rand(num_clusters, dim))
        self._init_params()

    def _init_params(self):
        self.conv.weight = nn.Parameter(
            (2.0 * self.alpha * self.centroids).unsqueeze(-1).unsqueeze(-1)
        )
        self.conv.bias = nn.Parameter(
            - self.alpha * self.centroids.norm(dim=1)
        )

    def forward(self, x):
        """
        Compute VLAD representation (N, K, D).
        :param x: input feature, dim: (N, C, L)
        :return:
        """
        N, C = x.shape[:2]

        if self.normalize_input:
            x = F.normalize(x, dim=1)                                                 # across descriptor dim

        # soft-assignment
        soft_assign = self.conv(x).view(N, self.num_clusters, -1)
        soft_assign = F.softmax(soft_assign, dim=1)

        x_flatten = x.view(N, C, -1)

        # calculate residuals to each clusters
        residual = x_flatten.expand(self.num_clusters, -1, -1, -1).permute(1, 0, 2, 3) - \
                   self.centroids.expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
        residual *= soft_assign.unsqueeze(2)
        vlad = residual.sum(dim=-1)

        vlad = F.normalize(vlad, p=2, dim=2)                                          # intra-normalization
        vlad = vlad.view(x.size(0), -1)                                               # flatten
        vlad = F.normalize(vlad, p=2, dim=1)                                          # L2 normalize, dim: (N, K, D)

        return vlad