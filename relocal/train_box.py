import torch
import torch.nn as nn
import shutil
import os
import numpy as np
import random
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt

from visualizer.visualizer_2d import show_multiple_img
from relocal.triplet_losses import LabelTripletLoss
from relocal.vlad_net import NetVLADEmbed
from core_dl.train_params import TrainParameters
from core_dl.base_train_box import BaseTrainBox
from relocal.kmeans import k_means
from seq_data.plot_seq_2d import plot_array_seq_2d


class RelocTrainBox(BaseTrainBox):
    """ VALD training container
    """

    """ Configurations -------------------------------------------------------------------------------------------------
    """
    def __init__(self, train_params: TrainParameters, workspace_dir=None, checkpoint_path=None, comment_msg=None):
        super(RelocTrainBox, self).__init__(train_params, workspace_dir, checkpoint_path, comment_msg)

    def _save_net_def(self):
        super(RelocTrainBox, self)._save_net_def()
        shutil.copy(os.path.realpath(__file__), self.model_def_dir)

    def _set_network(self):
        super(RelocTrainBox, self)._set_network()
        self.model = NetVLADEmbed(num_clusters=64)

    def _set_optimizer(self):
        super(RelocTrainBox, self)._set_optimizer()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.train_params.START_LR)

    def _set_loss_func(self):
        super(RelocTrainBox, self)._set_loss_func()
        # loss_func_config = {'overlap_thres': 0.54, 'baseline_thres': 0.22, 'anchor_ref_idx': 2}
        self.criterion = LabelTripletLoss(margin=0.05)

    """ Training Routines ----------------------------------------------------------------------------------------------
    """
    def _prepare_train(self):
        # add more keys
        self._add_log_keys(['Loss(Train)/triplet_loss',
                            'Accuracy(Train)/max_anchor2pos',
                            'Accuracy(Train)/min_anchor2neg',
                            'Accuracy(Train)/min_neg2pos',
                            'Accuracy(Train)/avg_neg2pos',
                            'Image(Train)/anchor',
                            'Image(Train)/max_pos',
                            'Image(Train)/min_neg',
                            'Accuracy(Valid)/max_anchor2pos',
                            'Accuracy(Valid)/min_anchor2neg',
                            'Accuracy(Valid)/min_neg2pos',
                            'Accuracy(Valid)/avg_neg2pos',
                            'Accuracy(Valid)/accu_ratio'])

    @staticmethod
    def preprocess(sample_dict, max_pos_num=-1, max_neg_num=-1):

        anchor_img = sample_dict['anchor_img']
        pos_imgs = sample_dict['pos_img']
        neg_imgs = sample_dict['neg_img']

        anchor_ori_img = sample_dict['anchor_ori_img']
        pos_ori_imgs = sample_dict['pos_ori_img']
        neg_ori_imgs = sample_dict['neg_ori_img']

        N, C, H, W = anchor_img.shape

        n_pos_imgs = pos_imgs.shape[1]
        n_neg_imgs = neg_imgs.shape[1]

        anchor_labels = torch.zeros(N, 1)       # mark the anchor as the negative samples

        # gen random pos index
        if max_pos_num > 1:
            pos_rand_idx = np.asarray(random.sample(range(n_pos_imgs), max_pos_num))
            pos_imgs = pos_imgs[:, pos_rand_idx, :, :, :]
            pos_ori_imgs = pos_ori_imgs[:, pos_rand_idx, :, :, :]
            pos_labels = torch.ones(N, max_pos_num)
        else:
            pos_labels = torch.ones(N, n_pos_imgs)

        if max_neg_num > 1:
            neg_rand_idx = np.asarray(random.sample(range(n_neg_imgs), max_neg_num))
            neg_imgs = neg_imgs[:, neg_rand_idx, :, :, :]
            neg_ori_imgs = neg_ori_imgs[:, neg_rand_idx, :, :, :]
            neg_labels = torch.zeros(N, max_neg_num)
        else:
            neg_labels = torch.zeros(N, n_neg_imgs)

        # stack together
        img_samples = torch.cat([anchor_img.unsqueeze(1), pos_imgs, neg_imgs], dim=1)
        ori_img_samples = torch.cat([anchor_ori_img.unsqueeze(1), pos_ori_imgs, neg_ori_imgs], dim=1)
        labels_samples = torch.cat([anchor_labels, pos_labels, neg_labels], dim=1)

        return img_samples, labels_samples, ori_img_samples

    def _train_feed(self, train_sample, cur_train_epoch, cur_train_itr):
        super(RelocTrainBox, self)._train_feed(train_sample, cur_train_epoch, cur_train_itr)
        with torch.cuda.device(self.dev_id):

            img_samples, labels_samples, ori_img_samples = self.preprocess(train_sample, max_pos_num=-1, max_neg_num=-1)

            N, L, C, H, W = img_samples.shape
            x = self.model(img_samples.view(N*L, C, H, W).cuda())

            # loss
            loss, acc = self.criterion.forward(x.view(N, L, -1), pos_neg_label=labels_samples)

            # do backwards
            loss.backward()

            # setup the
            if self.check_log_step(cur_train_itr):
                log_dict = {'Loss(Train)/triplet_loss': loss.item(),
                            'Accuracy(Train)/max_anchor2pos': acc['max_anchor2pos'][0],# np.average(acc['max_anchor2pos']),
                            'Accuracy(Train)/min_anchor2neg': acc['min_anchor2neg'][0],# np.average(acc['min_anchor2neg']),
                            'Accuracy(Train)/min_neg2pos': acc['min_neg2pos'][0],  # np.average(acc['min_neg2pos']),
                            'Accuracy(Train)/avg_neg2pos': acc['avg_neg2pos'][0],  # np.average(acc['avg_neg2pos']),
                            }

                if self.check_vis_step(cur_train_itr):
                    anchor_img = ori_img_samples[0, 0]
                    max_pos_img = ori_img_samples[0, int(acc['pos_idx'][0]) + 1]
                    neg_batch_idx = int(1 + np.sum(labels_samples[0].cpu().numpy()) + acc['neg_idx'][0]) // L
                    neg_idx = int(1 + np.sum(labels_samples[0].cpu().numpy()) + acc['neg_idx'][0]) % L
                    min_neg_img = ori_img_samples[neg_batch_idx, neg_idx]
                    log_dict['Image(Train)/anchor'] = [anchor_img]
                    log_dict['Image(Train)/max_pos'] = [max_pos_img]
                    log_dict['Image(Train)/min_neg'] = [min_neg_img]

                return log_dict
            else:
                return None

    def _valid_loop(self, valid_loader, train_epoch, train_itr):
        self.model.eval()
        with torch.cuda.device(self.dev_id):
            max_anchor2pos = []
            min_anchor2neg = []
            min_neg2pos = []
            avg_neg2pos = []
            accu_ratios = []

            for valid_batch_idx, valid_sample in enumerate(valid_loader):

                img_samples, labels_samples, _ = self.preprocess(valid_sample, max_pos_num=-1, max_neg_num=-1)

                N, L, C, H, W = img_samples.shape
                x = self.model(img_samples.view(N * L, C, H, W).cuda())
                acc = self.criterion.cal_accuracy(x.view(N, L, -1), labels_samples)

                max_anchor2pos.append(np.average(acc['max_anchor2pos']))
                min_anchor2neg.append(np.average(acc['min_anchor2neg']))
                min_neg2pos.append(np.average(acc['min_neg2pos']))
                avg_neg2pos.append(np.average(acc['avg_neg2pos']))
                accu_ratios.append(np.average(acc['accu_ratio']))

                if valid_batch_idx > self.train_params.MAX_VALID_BATCHES_NUM:
                    break

            max_anchor2pos = np.asarray(max_anchor2pos)
            min_anchor2neg = np.asarray(min_anchor2neg)
            min_neg2pos = np.asarray(min_neg2pos)
            avg_neg2pos = np.asarray(avg_neg2pos)
            accu_ratios = np.asarray(accu_ratios)

            return {'Accuracy(Valid)/max_anchor2pos': np.average(max_anchor2pos),
                    'Accuracy(Valid)/min_anchor2neg': np.average(min_anchor2neg),
                    'Accuracy(Valid)/min_neg2pos': np.average(min_neg2pos),
                    'Accuracy(Valid)/avg_neg2pos': np.average(avg_neg2pos),
                    'Accuracy(Valid)/accu_ratio': np.average(accu_ratios)}

    def check_vis_step(self, itr):
        return itr % (200*self.train_params.LOG_STEPS) == 0

    def init_NetVLAD_by_cluster(self, train_set, num_samples):
        train_loader = torch.utils.data.DataLoader(train_set,
                                                   batch_size=self.train_params.BATCH_SIZE,
                                                   shuffle=True,
                                                   num_workers=self.train_params.TORCH_DATALOADER_NUM)
        samples_count = 0
        features = []
        for train_batch_idx, train_sample in enumerate(train_loader):
            img_samples, labels_samples, ori_img_samples = self.preprocess(train_sample, max_pos_num=-1, max_neg_num=-1)
            N, L, C, H, W = img_samples.shape
            with torch.no_grad():
                feature = self.model.base_model(img_samples.view(N * L, C, H, W).cuda())  # (N * L, f_C, f_H, f_W)
            _, f_C, _, _ = feature.shape
            features.append(feature.permute(0, 2, 3, 1).contiguous().view(-1, f_C))
            samples_count += self.train_params.BATCH_SIZE
            if samples_count >= num_samples:
                break
        features = torch.cat(features, dim=0)
        centroids, _ = k_means(features, self.model.net_vald.num_clusters, verbose=True)
        self.model.net_vald.centroids = nn.Parameter(centroids)
        self.model.net_vald._init_params()

    def plot_seq_VALD_dist(self, valid_loader):
        self.model.eval()
        with torch.cuda.device(self.dev_id):
            for valid_batch_idx, valid_sample in enumerate(valid_loader):
                Tcw = valid_sample['Tcw'][0]
                K = valid_sample['K'][0]
                I = valid_sample['img'][0]
                d = valid_sample['depth'][0]

                L, C, H, W = I.shape
                with torch.no_grad():
                    x = self.model(I.cuda()).view(L, -1)

                anchor_idx = np.random.choice(L)
                dist = torch.sum((x - x[anchor_idx : anchor_idx + 1]) ** 2, dim=-1).cpu().numpy()
                vmax = np.amax(dist)
                vmin = np.amin(dist)
                print('max_value:', vmax, '   min_value:', vmin)
                cmap = cm.jet
                norm = Normalize(vmin=vmin, vmax=vmax)
                arrow_color = cmap(norm(dist))
                Tcw_n = Tcw.numpy()
                fig, axs = plt.subplots(1, 2, figsize=(16, 9))
                plot_array_seq_2d(Tcw_n, plt_axes=axs[0], color=(0, 0, 1), show_view_direction=True, legend='GT', arrow_color=arrow_color)
                axs[1].plot(dist)
                plt.show()

                input('wait')
