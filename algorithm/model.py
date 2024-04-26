import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
# from utils.cnn import CNN
# from utils.utils import init_weights



def call_bn(bn, x):
    return bn(x)


class CNN(nn.Module):
    def __init__(self, input_channel=3, n_outputs=100, dropout_rate=0.25, top_bn=False):
        self.dropout_rate = dropout_rate
        self.top_bn = top_bn
        super(CNN, self).__init__()
        self.c1 = nn.Conv2d(input_channel, 128, kernel_size=3, stride=1, padding=1)
        self.c2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.c3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.c4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.c5 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.c6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.c7 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=0)
        self.c8 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=0)
        self.c9 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=0)
        self.l_c1 = nn.Linear(128, n_outputs)
        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(256)
        self.bn6 = nn.BatchNorm2d(256)
        self.bn7 = nn.BatchNorm2d(512)
        self.bn8 = nn.BatchNorm2d(256)
        self.bn9 = nn.BatchNorm2d(128)
        self.dropout2d = nn.Dropout2d(p=self.dropout_rate)

        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.detach())
                m.bias.detach().zero_()
            elif isinstance(m, torch.nn.Linear):
                nn.init.kaiming_normal_(m.weight.detach())
                m.bias.detach().zero_()

    def forward(self, x, ):
        h = x
        h = self.c1(h)
        h = F.leaky_relu(call_bn(self.bn1, h), negative_slope=0.01)
        h = self.c2(h)
        h = F.leaky_relu(call_bn(self.bn2, h), negative_slope=0.01)
        h = self.c3(h)
        h = F.leaky_relu(call_bn(self.bn3, h), negative_slope=0.01)
        h = F.max_pool2d(h, kernel_size=2, stride=2)
        h = self.dropout2d(h)

        h = self.c4(h)
        h = F.leaky_relu(call_bn(self.bn4, h), negative_slope=0.01)
        h = self.c5(h)
        h = F.leaky_relu(call_bn(self.bn5, h), negative_slope=0.01)
        h = self.c6(h)
        h = F.leaky_relu(call_bn(self.bn6, h), negative_slope=0.01)
        h = F.max_pool2d(h, kernel_size=2, stride=2)
        h = self.dropout2d(h)

        h = self.c7(h)
        h = F.leaky_relu(call_bn(self.bn7, h), negative_slope=0.01)
        h = self.c8(h)
        h = F.leaky_relu(call_bn(self.bn8, h), negative_slope=0.01)
        h = self.c9(h)
        h = F.leaky_relu(call_bn(self.bn9, h), negative_slope=0.01)
        h = F.avg_pool2d(h, kernel_size=h.data.shape[2])            
        h = h.view(h.size(0), h.size(1))        


        normalized_feature = nn.functional.normalize(h)
 
        logit = self.l_c1(h)

        
        if self.top_bn:
            logit = call_bn(self.bn_c1, logit)
        return logit, normalized_feature

class MLPNet(nn.Module):

    def __init__(self):
        super(MLPNet, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 256)
        self.fc2 = nn.Linear(256, 10)
        self.dropout = nn.Dropout(p=0.25)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x



class Model(nn.Module):
    def __init__(self, arch='resnet18', num_classes=200, mlp_hidden=2, pretrained=True):
        super().__init__()
        self.encoder = Encoder(arch, num_classes, pretrained)
        self.classifier = MLPHead(self.encoder.feature_dim, mlp_hidden, num_classes)

    def forward(self, x):
        feature = self.encoder(x)
        logit = self.classifier(feature)
        normalized_feature = nn.functional.normalize(feature)
        return logit , normalized_feature




class Encoder(nn.Module):
    def __init__(self, arch='cnn', num_classes=200, pretrained=True):
        super().__init__()
        if arch.startswith('resnet') and arch in torchvision.models.__dict__.keys():
            resnet = torchvision.models.__dict__[arch](pretrained=pretrained)
            self.encoder = nn.Sequential(*list(resnet.children())[:-1])
            self.feature_dim = resnet.fc.in_features
      
        else:
            raise AssertionError(f'{arch} is not supported!')

    def forward(self, x):
        h = self.encoder(x)
        return h.view(h.shape[0], -1)


class MLPHead(nn.Module):
    def __init__(self, in_channels, mlp_hidden, projection_size, init_method='He'):
        super().__init__()

        mlp_hidden_size = round(mlp_hidden * in_channels)
        self.mlp_head = nn.Sequential(
            nn.Linear(in_channels, mlp_hidden_size),
            nn.BatchNorm1d(mlp_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden_size, projection_size)
        )
        init_weights(self.mlp_head, init_method)

    def forward(self, x):
        return self.mlp_head(x)

def init_weights(module, init_method='He'):
    for _, m in module.named_modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            if init_method == 'He':
                nn.init.kaiming_normal_(m.weight.data)
            elif init_method == 'Xavier':
                nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, val=0)