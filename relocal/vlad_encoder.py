from relocal.vlad_net import NetVLAD
import torchvision.models as models
import torch.nn as nn
import torch
import numpy as np
import pickle
from sklearn.metrics import pairwise
import torchvision.transforms as transforms

class VLADEncoder:

    def __init__(self, checkpoint_path, dev_id=0):
        pretrained = True
        num_clusters = 64
        encoder_dim = 512

        # use VGG16 as basic encoder -----------------------------------------------------------------------------------
        encoder = models.vgg16(pretrained=pretrained)
        layers = list(encoder.features.children())[:-1]
        if pretrained:
            for l in layers[:-5]:
                for p in l.parameters():
                     p.requires_grad = False

        encoder = nn.Sequential(*layers)
        model = nn.Module()
        model.add_module('encoder', encoder)
        net_vlad = NetVLAD(num_clusters=num_clusters, dim=encoder_dim, vladv2=False)
        model.add_module('pool', net_vlad)

        # load from checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'])
        with torch.cuda.device(dev_id):
            model.cuda()
            model.eval()

        self.transform_func = transforms.Compose([
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])

        self.dev_id = dev_id
        self.model = model
        self.sample_infos = []
        self.sample_embeds = []

    def forward(self, sample):
        image_encoding = self.model.encoder(sample.cuda())
        vlad_encoding = self.model.pool(image_encoding)
        return vlad_encoding

    def add_sample(self, sample, sample_info):
        if isinstance(sample, np.ndarray):
            sample = torch.from_numpy(sample).permute(2, 0, 1)
            sample = self.transform_func(sample).unsqueeze(0)

        with torch.cuda.device(self.dev_id), torch.no_grad():
            image_encoding = self.model.encoder(sample.cuda())
            vlad_encoding = self.model.pool(image_encoding)
            self.sample_embeds.append(vlad_encoding.detach().cpu().numpy())
            self.sample_infos.append(sample_info)

    def find_close_samples(self, sample, top_k=5):
        if isinstance(sample, np.ndarray):
            sample = torch.from_numpy(sample).permute(2, 0, 1)
            sample = self.transform_func(sample).unsqueeze(0)

        with torch.cuda.device(self.dev_id), torch.no_grad():
            image_encoding = self.model.encoder(sample.cuda())
            vlad_encoding = self.model.pool(image_encoding)
            vlad_encoding = vlad_encoding.detach().cpu().numpy()
            dist = [pairwise.cosine_similarity(vlad_encoding,
                                               self.sample_embeds[idx]) for idx in range(len(self.sample_embeds))]
            dist = np.asarray(dist).ravel()
            sorted_indices = np.argsort((1 - dist))
            top_k_frames = [self.sample_infos[idx] for idx in sorted_indices[:top_k]]
        return top_k_frames

    def dump(self, path):
        with open(path, 'wb') as f:
            pickle.dump([self.sample_embeds, self.sample_infos], f)

    def load(self, path):
        with open(path, 'rb') as f:
            x = pickle.load(f)
            self.sample_embeds = x[0]
            self.sample_infos = x[1]
