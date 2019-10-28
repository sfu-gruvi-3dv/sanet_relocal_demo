import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class LabelTripletLoss(nn.Module):
    """ The triplet ranking loss
        [ref: https://arxiv.org/pdf/1511.07247.pdf]
    """
    def __init__(self, margin):
        super(LabelTripletLoss, self).__init__()
        self.margin = margin
        self.anchor_ref_idx = 0

    @staticmethod
    def pos2neg_dists(anchor2sample_dist, pos_neg_label, anchor_ref_idx=0):
        N, W = pos_neg_label.shape
        anchor2sample_dist = anchor2sample_dist.detach().cpu().numpy()
        anchor_dists = []

        for n in range(N):
            pos_labels = np.zeros(N * W)
            pos_labels[n * W + np.where(pos_neg_label[n, :] == 1.0)[0]] = 1.0

            # compute the max anchor 2 positive samples
            anchor2pos = anchor2sample_dist[n, np.where(pos_labels == 1.0)].squeeze()
            max_anchor2pos = max(anchor2pos)

            # compute the min anchor 2 positive samples
            pos_labels[n * W + anchor_ref_idx] = 1.0
            anchor2neg = anchor2sample_dist[n, np.where(pos_labels != 1.0)].squeeze()
            min_anchor2neg = min(anchor2neg)

            # compute pairwise distance
            anchor2neg = np.expand_dims(anchor2neg, axis=1)
            anchor2pos = np.expand_dims(anchor2pos, axis=0)
            neg2pos = anchor2neg - anchor2pos
            min_neg2pos = np.min(neg2pos)
            xx = np.unravel_index(neg2pos.argmin(), neg2pos.shape)
            accu_ratio = float(np.count_nonzero((neg2pos > 0))) / float(neg2pos.size)
            # print(xx)
            avg_neg2pos = np.average(neg2pos)

            anchor_dists.append((max_anchor2pos, min_anchor2neg, min_neg2pos, avg_neg2pos, xx[0], xx[1], accu_ratio))
        anchor_dists = np.asarray(anchor_dists)

        return {'max_anchor2pos': anchor_dists[:, 0],
                'min_anchor2neg': anchor_dists[:, 1],
                'min_neg2pos': anchor_dists[:, 2],
                'avg_neg2pos': anchor_dists[:, 3],
                'neg_idx': anchor_dists[:, 4],
                'pos_idx': anchor_dists[:, 5],
                'accu_ratio': anchor_dists[:, 6]}

    def cal_accuracy(self, pred_embeddings, pos_neg_label):
        N, W, X = pred_embeddings.shape
        pos_neg_label = pos_neg_label.detach().cpu().numpy()

        # Compute distance between embeddings
        anchor2sample_dist = self._anchor2sample_dist(pred_embeddings, squared=True, anchor_idx=0)     # dim: (N, N*W)
        # pairwise_dist = self._pairwise_distance(pred_embeddings.view(N*W, -1), squared=True)         # dim: (N*W, N*W)
        # pairwise_dist = pairwise_dist.reshape(N, W, N * W)

        # Compute the positive and negative distance
        return self.pos2neg_dists(anchor2sample_dist, pos_neg_label)

    def forward(self, pred_embeddings, pos_neg_label):
        """
        Compute triplet
        :param pred_embeddings: feature embeddings, dim: (N, W, X)
        :param pos_neg_label: positive and negative sample labels, dim: (N, W)
        :return:
        """

        N, W, X = pred_embeddings.shape
        pos_neg_label = pos_neg_label.detach().cpu().numpy()

        # compute distance between embeddings
        anchor2sample_dist = self._anchor2sample_dist(pred_embeddings, squared=True, anchor_idx=0)      # dim: (N, N*W)
        # pairwise_dist = self._pairwise_distance(pred_embeddings.view(N*W, -1), squared=True)          # dim: (N*W, N*W)
        # pairwise_dist = pairwise_dist.reshape(N, W, N * W)
        # anchor2sample_dist = pairwise_dist[:, 0, :].squeeze(dim=1)                                    # dim: (N, N*W)

        # compute the positive and negative distance
        anc_pos_dist = anchor2sample_dist.unsqueeze(dim=2)                                            # dim: (N, W, 1)
        anc_neg_dist = anchor2sample_dist.unsqueeze(dim=1)                                            # dim: (N, 1, W)

        # triplet_loss[i, j, k] will contain the triplet loss of anc=i (per batch), pos=j, neg=k
        # triplet loss defines as: max(pos - neg + margin, 0)
        loss = anc_pos_dist - anc_neg_dist + self.margin                                           # dim: (N, N*W, N*W)

        # select the valid triplet
        mask = self._mask_valid(pos_neg_label)
        triplet_loss = loss * mask

        # Remove negative losses (i.e. the easy triplets)
        triplet_loss = F.relu(triplet_loss)

        # Count number of hard triplets (where triplet_loss > 0)
        hard_triplets = torch.gt(triplet_loss, 1e-16).float()
        num_hard_triplets = torch.sum(hard_triplets)

        triplet_loss = torch.sum(triplet_loss) / (num_hard_triplets + 1e-16)

        # compute accuracy
        pos2neg_dist = self.pos2neg_dists(anchor2sample_dist, pos_neg_label)

        return triplet_loss, pos2neg_dist

    @staticmethod
    def _mask_valid(pos_neg_label, anchor_idx=0):
        """
        Mask out the invalid triplet
        :param pos_neg_label: positive and negative sample labels, dim: (N, W)
        :return: mask, dim: (N, N*W, N*W)
        """
        N, W = pos_neg_label.shape

        output = np.zeros((N, N*W, N*W), dtype=np.float32)
        for n in range(N):

            # select anchor index
            mask = pos_neg_label[n, :]
            if np.any(mask) == 1.0:
                output_row_start_idx = n*W
                pos_indices = np.where(mask == 1.0)
                pos_indices = pos_indices[0] + output_row_start_idx
                for pos_idx in pos_indices:
                    output[n, pos_idx, :] = 1.0
                    output[n, pos_idx, pos_indices] = 0.0
                    output[n, pos_idx, output_row_start_idx + anchor_idx] = 0.0
                # output[n, :, anchor_idx] = 0.0

        return torch.tensor(output).cuda().detach()

    @staticmethod
    def _pairwise_distance(x, squared=False, eps=1e-16):
        """
        Compute the pairwise distance matrix:
            ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
        :param x: embedding vector
        :param squared: use square distance if needed
        :param eps:
        :return: embedding distance from anchor to samples, dim: (N*W, N*W)
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

    @staticmethod
    def _anchor2sample_dist(x, squared=False, eps=1e-16, anchor_idx=0):
        """
        Compute the distance from anchor to other samples:
            ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
        :param x: embedding vector, dim: (N, W, X)
        :param squared: enable square distance
        :param eps:
        :param anchor_idx: the anchor distance
        :return: embedding distance from anchor to samples, dim: (N, N*W)
        """
        N, W, X = x.shape
        anchor_x = x[:, anchor_idx, :]

        a_dot_b = torch.matmul(anchor_x.view(N, X), x.view(N*W, X).t())                                  # dim: (N, N*W)
        a_dot_a = torch.bmm(anchor_x.view(N, 1, X), anchor_x.view(N, X, 1)).view(N)         # dim: (N)
        b_dot_b = torch.bmm(x.view(N*W, 1, X), x.view(N*W, X, 1)).view(N*W)                 # dim: (N*W)

        distances = a_dot_a.unsqueeze(1) - 2 * a_dot_b + b_dot_b.unsqueeze(0)
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