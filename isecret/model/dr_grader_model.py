import torch
import torch.nn as nn
from torchvision import models

from isecret.model.backbone import DRMODEL_REGISTRY
from isecret.utils.transform import UnNormalize
from torchvision import transforms

@ DRMODEL_REGISTRY.register('resnext101')
class DRGrader(nn.Module):
    def __init__(self, args, gpu, network="resnext101"):
        super(DRGrader, self).__init__()
        self.args = args
        self.device = gpu
        self.model_path = args.model.dr_model_path
        self.model = self.create_model(network)
        self.load_weights()
        mean = [0.425753653049469, 0.29737451672554016, 0.21293757855892181]
        std = [0.27670302987098694, 0.20240527391433716, 0.1686241775751114]
        self.dr_norm = transforms.Normalize(mean, std)
        self.un_norm = UnNormalize(self.args.data.mean, self.args.data.std)


    def forward(self, input):
        x_shift = self.process_input(input)
        feature_map = None
        x = x_shift
        for name, module in self.model.named_children():
            x = module(x)
            # print(x.shape)
            if name == self.args.train.dr_return_layer:
                feature_map = x.data
                break
        return feature_map
    
    def process_input(self, x):
        x_unnorm = torch.cat([self.un_norm(x[k, ...]).unsqueeze(dim=0) for k in range(x.size(0))], dim=0)
        x_shift = torch.cat([self.dr_norm(x_unnorm[k, ...]).unsqueeze(dim=0) for k in range(x_unnorm.size(0))], dim=0)
        return x_shift

    def load_weights(self):
        self.state_dict = torch.load(self.model_path, map_location="cpu")
        self.model.load_state_dict(self.state_dict, strict=True)
        # no grad and start eval mode
        for param in self.model.parameters():
            param.requires_grad=False
        self.model.eval()


    def create_model(self, network):
        if network == "resnext101":
            model = models.resnext101_32x8d(pretrained=False)
        else:
            raise NotImplementedError
        model.fc = nn.Linear(model.fc.in_features, 1)
        model.to(self.device)
        return model
