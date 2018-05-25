import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Model(nn.Module) :
    def __init__(
            self,
            batch_size
            ) :
        super(Model,self).__init__()
        # hyperparameters settings
        self.batch_size = batch_size

        # layer
        # in convolution layer, layer is composed of
        # 1. Convolution layer
        # 2. ReLU layer
        # 3. Max-pool layer
        self.layer1 = nn.Sequential(
                nn.Conv2d(1,64,10),
                nn.ReLU(),
                nn.MaxPool2d(2,stride=2)
                ) # 1@105*105 > 64@48*48
        self.layer2 = nn.Sequential(
                nn.Conv2d(64,128,7),
                nn.ReLU(),
                nn.MaxPool2d(2,stride=2)
                ) # 64@48*48 > 128@21*21
        self.layer3 = nn.Sequential(
                nn.Conv2d(128,128,4),
                nn.ReLU(),
                nn.MaxPool2d(2,stride=2)
                ) # 128@21*21 > 128@9*9
        self.layer4 = nn.Sequential(
                nn.Conv2d(128,256,4),
                nn.ReLU()
                ) # 128@9*9 > 256@6*6
        # resize 256@6*6 > 9216 > 4096
        self.linear = nn.Sequential(
                nn.Linear(9216,4096),
                nn.Sigmoid()
                )
        self.final = nn.Sequential(
                nn.Linear(4096,1,bias=False),
#                nn.Linear(4096,1),
                nn.Sigmoid()
                )

        self.initHidden()

    def forward(self,image1,image2) :
        #image1, image2 shape : 128*1*105*105
        #image = image.t()
        #image /= 255
        #image_1 = image[:105*105].t()
        #image_2 = image[105*105:].t()
        #image1 = image_1.contiguous().view(self.batch_size,1,105,105)
        #image2 = image_2.contiguous().view(self.batch_size,1,105,105)
        #image1 = image_1.view(self.batch_size,1,105,105)
        #image2 = image_2.view(self.batch_size,1,105,105)

        h1_1 = self.layer1(image1)
        h1_2 = self.layer1(image2)

        h2_1 = self.layer2(h1_1)
        h2_2 = self.layer2(h1_2)

        h3_1 = self.layer3(h2_1)
        h3_2 = self.layer3(h2_2)

        h4_1 = self.layer4(h3_1)
        h4_2 = self.layer4(h3_2)

        h_c1 = h4_1.view(self.batch_size,-1)
        h_c2 = h4_2.view(self.batch_size,-1)
        h1 = self.linear(h_c1)
        h2 = self.linear(h_c2)
        # L1 distance, result=h
        h = F.l1_loss(h1,h2,size_average=False,reduce=False)
        #h = torch.abs(h1 - h2)

        out = self.final(h)
        return out

    def initHidden(self) :
        self.layer1[0].weight.data.normal_(0,0.01)
        self.layer2[0].weight.data.normal_(0,0.01)
        self.layer3[0].weight.data.normal_(0,0.01)
        self.layer4[0].weight.data.normal_(0,0.01)
        self.layer1[0].bias.data.normal_(0.5,0.01)
        self.layer2[0].bias.data.normal_(0.5,0.01)
        self.layer3[0].bias.data.normal_(0.5,0.01)
        self.layer4[0].bias.data.normal_(0.5,0.01)

        self.linear[0].weight.data.normal_(0,0.02)
        self.final[0].weight.data.normal_(0,0.02)
        #self.linear[0].bias.data.normal_(0.5,0.01)
        #self.final[0].bias.data.normal_(0.5,0.01)
