import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import torch


class GraspNet(nn.Module):
    def __init__(self):
        super(GraspNet, self).__init__()
        self.resnet50_cam = models.resnet50(pretrained=True) #Input 3 channels, output = 1000 features
        self.resnet50_cam = nn.Sequential(*list(self.resnet50_cam.children())[:-1])

        self.resnet50_gel = models.resnet50(pretrained=True) #Input 3 channels, output = 1000 features
        self.resnet50_gel = nn.Sequential(*list(self.resnet50_gel.children())[:-1])

        self.fc1 = nn.Linear(12288, 1024)
        self.fc2 = nn.Linear(1024, 2)

    def forward(self, cam_bef, cam_dur, lgel_bef, lgel_dur, rgel_bef, rgel_dur):
        cam_bef = self.resnet50_cam(cam_bef)
        cam_dur = self.resnet50_cam(cam_dur)

        lgel_bef = self.resnet50_gel(lgel_bef)
        lgel_dur = self.resnet50_gel(lgel_dur)
        rgel_bef = self.resnet50_gel(rgel_bef)
        rgel_dur = self.resnet50_gel(rgel_dur)

        x = torch.cat([cam_bef, cam_dur, lgel_bef, lgel_dur, rgel_bef, rgel_dur], dim=1)
        x = torch.flatten(x, start_dim=1) # Batch_size, 12288
        x = F.relu(self.fc1(x)) 
        x = self.fc2(x)
        return x