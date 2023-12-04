import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import *
from torch.autograd import Variable
from genotypes import PRIMITIVES
import numpy as np
from thop import profile
from matplotlib import pyplot as plt
from thop import profile
from quantize import QConv2d, QLinear
from config_train import config

def loss_kl(outputs, teacher_outputs):
    kl_loss = nn.KLDivLoss(reduction="batchmean")
    return kl_loss


class MixedOp(nn.Module):
    def __init__(self, C_in, C_out, op_idx, layer_id, stride=1, num_bits_list=[32,]):
        super(MixedOp, self).__init__()
        self.layer_id = layer_id
        print(op_idx)
        self._op = OPS[PRIMITIVES[op_idx]](C_in, C_out, layer_id, stride, num_bits_list)

    def forward(self, x, num_bits):
        return self._op(x, num_bits)


    def forward_flops(self, size):
        # int: force #channel; tensor: arch_ratio; float(<=1): force width
        flops, size_out = self._op.forward_flops(size)

        return flops, size_out


class FBNet_Infer(nn.Module):
    def __init__(self, alpha, config):
        super(FBNet_Infer, self).__init__()

        op_idx_list = F.softmax(alpha, dim=-1).argmax(-1)

        self.num_classes = config.num_classes

        self.num_bits_list = config.num_bits_list

        self.num_layer_list = config.num_layer_list
        self.num_channel_list = config.num_channel_list
        self.stride_list = config.stride_list

        self.stem_channel = config.stem_channel
        self.header_channel = config.header_channel

        self.stem = ConvNorm(3, self.stem_channel, kernel_size=3, stride=1, padding=1, bias=False, num_bits_list=[32,])

        self.cells = nn.ModuleList()
        self.feature_fixed = []
        self.connected_feat = [5,9,13,17]
        layer_id = 1

        for stage_id, num_layer in enumerate(self.num_layer_list):
            for i in range(num_layer):
                if i == 0:
                    if stage_id == 0:
                        op = MixedOp(self.stem_channel, self.num_channel_list[stage_id], op_idx_list[layer_id-1], layer_id, stride=self.stride_list[stage_id], num_bits_list=self.num_bits_list)
                    else:
                        op = MixedOp(self.num_channel_list[stage_id-1], self.num_channel_list[stage_id], op_idx_list[layer_id-1], layer_id, stride=self.stride_list[stage_id], num_bits_list=self.num_bits_list)
                else:
                    op = MixedOp(self.num_channel_list[stage_id], self.num_channel_list[stage_id], op_idx_list[layer_id-1], layer_id, stride=1, num_bits_list=self.num_bits_list)
                
                layer_id += 1
                self.cells.append(op)

        self.header = ConvNorm(self.num_channel_list[-1], self.header_channel, kernel_size=1, num_bits_list=[32,])

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = QLinear(self.header_channel, self.num_classes)

        self._criterion = nn.CrossEntropyLoss().cuda()


    def forward(self, input, num_bits=32):
        out = self.stem(input, num_bits=32)

        for i, cell in enumerate(self.cells):
            out = cell(out, num_bits)

        out = self.fc(self.avgpool(self.header(out, num_bits=32)).view(out.size(0), -1), num_bits=32)

        return out
    def get_channel_num(self):
        con_chan = []
        #print(self.cells[self.connected_feat[1]]._op.conv1)#.get_chan)
        for i in self.connected_feat:
            con_chan.append(self.cells[i]._op.conv3.get_chan)
        #print(con_chan)
        return con_chan
    def get_bn_before_relu(self):
        
        #print(self.cells[0]._op)
        '''bn1 = self.cells[0]._op
        bn2 = self.cells[1]._op
        bn3 = self.cells[2]._op
        bn4 = self.cells[3]._op
        '''
        (a,b,c,d) = self.connected_feat
        bn1 = (self.cells[a]._op.bn3 if isinstance(self.cells[a]._op, ConvBlock) else self.cells[a]._op.bn3)
        bn2 = (self.cells[b]._op.bn3 if isinstance(self.cells[b]._op, ConvBlock) else self.cells[b]._op.bn3)
        bn3 = (self.cells[c]._op.bn3 if isinstance(self.cells[c]._op, ConvBlock) else self.cells[c]._op.bn3)
        bn4 = (self.cells[d]._op.bn3 if isinstance(self.cells[d]._op, ConvBlock) else self.cells[d]._op.bn3)
        #print(bn1)
        return [bn1, bn2, bn3, bn4]
    
    def extract_feature(self, input, num_bits):
        out = self.stem(input, num_bits=32)
        klcells = []
        tmp=[]
        feat_num_list = []
        ret = []
        #print(self.connected_feat)
        for i, cell in enumerate(self.cells):
            fpout = cell(out, 32)
            out = cell(out, num_bits)
            klcells.append(out)
            tmp.append(loss_kl(F.log_softmax(out),F.softmax(out)))
        if self.feature_fixed == 0:
            for j, loss in enumerate(sorted(tmp,reverse=True)):
                if j >= 4:
                    break
                for k, feat in enumerate(tmp):
                    if loss == feat:
                        feat_num_list.append(k)
            self.feature_fixed = 1
            self.connected_feat = feat_num_list
        for l in sorted(self.connected_feat):
            ret.append(klcells[l])
        out = self.fc(self.avgpool(self.header(out, num_bits=32)).view(out.size(0), -1), num_bits=32)
        return ret, out

    def forward_flops(self, size):

        flops_total = []

        flops, size = self.stem.forward_flops(size)
        flops_total.append(flops)

        for i, cell in enumerate(self.cells):
            flops, size = cell.forward_flops(size)
            flops_total.append(flops)

        flops, size = self.header.forward_flops(size)
        flops_total.append(flops)

        return sum(flops_total)
        

    def _loss_backward(self, input, target, num_bits_list=None, bit_schedule='joint'):
        if num_bits_list is None:
            num_bits_list = self.num_bits_list

        loss_val = [-1 for _ in num_bits_list]

        if bit_schedule == 'joint':
            for num_bits in num_bits_list:
                logit = self(input, num_bits)
                loss = self._criterion(logit, target)
                loss.backward()

                loss_val[num_bits_list.index(num_bits)] = loss.item()

        elif bit_schedule == 'max_loss':
            loss_list = []

            for i, num_bits in enumerate(num_bits_list):
                logit = self(input, num_bits)
                loss = self._criterion(logit, target)

                loss_list.append(loss.item())

                del logit
                del loss

            num_bits_max = num_bits_list[np.array(loss_list).argmax()]

            logit = self(input, num_bits_max)
            loss = self._criterion(logit, target)

            loss.backward()
            loss_val[num_bits_list.index(num_bits_max)] = loss.item()

        return loss_val
