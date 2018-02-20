import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class CNN_Text(nn.Module):
    def __init__(self, args):
        super(CNN_Text, self).__init__()
        self.args = args

        V = args.embed_num
        D = args.embed_dim
        C = args.class_num
        Ci = 1
        Co = args.kernel_num
        Ks = args.kernel_sizes
        L = args.length

        self.embed = nn.Embedding(V, D)

        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D), padding=(K - 1, 0)) for K in Ks])
        # self.len = 1

        senti_dim = args.senti_embed_dim
        self.senti_embed = nn.Embedding(V, senti_dim)
        self.senti_kernel_size = args.senti_kernel_size
        self.senti_conv = nn.Conv2d(Ci, Co, (self.senti_kernel_size, senti_dim))
        # senti_ks = [5, 10]
        # self.senti_conv = nn.ModuleList([nn.Conv2d(Ci, Co, (K, senti_dim)) for K in senti_ks])
        self.dropout = nn.Dropout(args.dropout)

        if self.args.multi_channels:
            self.fc1 = nn.Linear((len(Ks)+1 ) * Co, C)
        else:
            self.fc1 = nn.Linear((len(Ks)) * Co, C)

        #self._initialize_weights()


    def forward(self, x, senti_x):
        x = self.embed(x)  # (N,W,D)

        if self.args.static:
            x = Variable(x)



        x = x.unsqueeze(1)  # (N,Ci,W,D)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N,Co,W), ...]*len(Ks)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N,Co), ...]*len(Ks)


        if self.args.multi_channels:
            senti_x = self.senti_embed(senti_x)
            senti_x = senti_x.unsqueeze(1)
            # self.senti_kernel_size = senti_x.size()[2]/4
            senti_x = F.relu(self.senti_conv(senti_x)).squeeze(3)
            # senti_x = self.senti_conv(senti_x).squeeze(3)
            senti_x = F.avg_pool1d(senti_x, senti_x.size(2)).squeeze(2)

        if self.args.multi_channels:
            x.append(senti_x)

        x = torch.cat(x, 1)

        x = self.dropout(x)  # (N,len(Ks)*Co)

        logit = self.fc1(x)  # (N,C)
        return logit

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(-0.001, 0.001)
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()