import torch.nn as nn
import torch.nn.functional as F
import timm
import torchvision.models as models

class EfficientNet_new(nn.Module):

    def __init__(self, num_classes, in_channels=3):
        super(EfficientNet_new, self).__init__()

        self.n_classes = num_classes
        self.in_channels = in_channels
        self.model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=self.n_classes, in_chans=self.in_channels)
        self.new_classifier = [nn.Dropout(0.5)]
        self.model.classifier = nn.Sequential(*self.new_classifier)

        self.linear_layer = nn.Linear(1280, 640)
        #self.linear_layer_1 = nn.Linear(1280,1280)
        #self.linear_layer = nn.Linear(1280, 1280)
        self.classification_layer = nn.Linear(640, self.n_classes)


    def forward(self, x):

        x = self.model(x)
        x_emb = self.linear_layer(x)
        #x_linear = self.linear_layer_1(x)
        x_logits = self.classification_layer(x_emb)

        x_out = F.log_softmax(x_logits, dim=1)
        return x_emb, x_logits

class EfficientNet(nn.Module):

    def __init__(self, num_classes, in_channels=3):
        super(EfficientNet, self).__init__()

        self.n_classes = num_classes
        self.in_channels = in_channels
        self.model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=self.n_classes, in_chans=self.in_channels)
        self.new_classifier = [nn.Dropout(0.5)]
        self.model.classifier = nn.Sequential(*self.new_classifier)

        self.linear_layer = nn.Linear(1280, 640)
        self.linear_layer_1 = nn.Linear(1280,1280)
        #self.linear_layer = nn.Linear(1280, 1280)
        self.classification_layer = nn.Linear(1280, self.n_classes)


    def forward(self, x):

        x = self.model(x)
        x_emb = self.linear_layer(x)
        x_linear = self.linear_layer_1(x)
        x_logits = self.classification_layer(x_linear)

        x_out = F.log_softmax(x_logits, dim=1)
        return x_emb, x_logits

class ResNet(nn.Module):

    def __init__(self, num_classes, in_channels=3):
        super(ResNet, self).__init__()

        self.n_classes = num_classes
        self.in_channels = in_channels
        self.model = timm.create_model('wide_resnet50_2', pretrained=True, num_classes=self.n_classes, in_chans=self.in_channels)
        self.new_classifier = [nn.Dropout(0.5)]
        self.model.classifier = nn.Sequential(*self.new_classifier)
        self.model.fc = nn.Linear(2048, 1280)
        self.linear_layer = nn.Linear(1280, 640)
        self.classification_layer = nn.Linear(640, self.n_classes)

        #self.linear_layer = nn.Linear(1280, 640)
        #self.linear_layer_1 = nn.Linear(1280,1280)
        #self.linear_layer = nn.Linear(1280, 1280)
        #self.classification_layer = nn.Linear(1280, self.n_classes)


    def forward(self, x):

        x = self.model(x)
        x_emb = self.linear_layer(x)
        x_logits = self.classification_layer(x_emb)

        return x_emb, x_logits

class DogBreedPretrainedWideResnet(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.network = models.wide_resnet50_2(pretrained=True)
        # Replace last layer
        num_ftrs = self.network.fc.in_features
        #self.network.fc = nn.Sequential(
        #    nn.Linear(num_ftrs, 120),
        #    nn.LogSoftmax(dim=1)
        #)
        self.network.fc = nn.Identity()
        self.linear1 = nn.Linear(num_ftrs,640)
        self.linear2 = nn.Linear(640,200)
        #self.act = nn.LogSoftmax(dim=1)

        
    def forward(self, xb):
        xb = self.network(xb)
        x_emb = self.linear1(xb)
        x = self.linear2(x_emb)
        #x = self.act(x)
        return x_emb, x
