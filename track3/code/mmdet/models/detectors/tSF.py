from copy import deepcopy
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

# self.query_embed = nn.Embedding(self.num_queries, self.hidden_dim)

## multi-head attention ##
def MLP(channels: list, do_bn=True):
    """ Multi-layer perceptron """
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(
            nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n-1):
            if do_bn:
                layers.append(nn.BatchNorm1d(channels[i]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)

def attention(query, key, value):
    dim = query.shape[1]
    scores = torch.einsum('bdhn,bdhm->bhnm', query, key) / dim**.5
    prob = torch.nn.functional.softmax(scores, dim=-1)
    return torch.einsum('bhnm,bdhm->bdhn', prob, value), prob

class MultiHeadedAttention(nn.Module):
    """ Multi-head attention to increase model expressivitiy """
    def __init__(self, num_heads: int, d_model: int, with_W=True):
        super().__init__()
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads
        self.num_heads = num_heads
        self.with_W = with_W
        self.merge = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])

    def forward(self, query, key, value):
        # B, C, N = query.size()
        batch_dim = query.size(0)    
        if self.with_W:
            # with W
            query, key, value = [l(x).view(batch_dim, self.dim, self.num_heads, -1)
                                for l, x in zip(self.proj, (query, key, value))]
        else:
            # without W
            query = query.view(batch_dim, self.dim, self.num_heads, -1)
            key = key.view(batch_dim, self.dim, self.num_heads, -1)
            value = value.view(batch_dim, self.dim, self.num_heads, -1)

        x, softmax_qk = attention(query, key, value)
        return self.merge(x.contiguous().view(batch_dim, self.dim*self.num_heads, -1)), softmax_qk

##############################################

## single-head attention ##
def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    if activation == "leaky_relu":
        return F.leaky_relu
    raise RuntimeError(F"activation should be relu/gelu/glu/leaky_relu, not {activation}.")

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """
    # usage: self.attention = ScaledDotProductAttention(temperature=np.power(feat_dim, 0.5))
    def __init__(self, temperature, attn_dropout=0.0):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature
        log_attn = F.log_softmax(attn, 2)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        return output, attn, log_attn

class FFN_MLP(nn.Module):
    def __init__(self, feature_dim, d_ffn=1024, dropout=0.1, activation="relu"):
        super(FFN_MLP, self).__init__()
        self.linear1 = nn.Linear(feature_dim, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, feature_dim)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(feature_dim)

    def forward(self, src):
        src2 = self.linear2(self.dropout3(self.activation(self.linear1(src))))
        src = src + self.dropout4(src2)
        src = self.norm3(src)
        return src

class FFN_BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, kernel=3, stride=1, downsample=None, norm_layer=nn.BatchNorm2d):
        super(FFN_BasicBlock, self).__init__()
        self.norm_layer = norm_layer
        if kernel == 1:
            self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        elif kernel == 3:
            self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = self.norm_layer(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = self.norm_layer(planes)
        if kernel == 1:
            self.conv3 = nn.Conv2d(planes, planes, kernel_size=1, bias=False)
        elif kernel == 3:
            self.conv3 = conv3x3(planes, planes)
        self.bn3 = self.norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

##############################################

class tSF(nn.Module):
    def __init__(self, feature_dim, num_queries=100, num_heads=1, FFN_method='MLP'):
        super(tSF, self).__init__()
        self.feature_dim = feature_dim
        self.num_queries = num_queries
        self.num_heads = num_heads
        self.FFN_method = FFN_method
        # query embedding
        self.query_embed = nn.Embedding(self.num_queries, self.feature_dim)
        # attention
        self.with_W = False
        if self.num_heads > 1:
            self.attention = MultiHeadedAttention(self.num_heads, self.feature_dim, with_W=self.with_W) # (B, C, N)
            #self.mlp = MLP([self.feature_dim*2, self.feature_dim*2, self.feature_dim])
            #nn.init.constant_(self.mlp[-1].bias, 0.0)
            self.FFN = FFN_MLP(self.feature_dim, d_ffn=self.feature_dim*2)
        else:
            # with W
            if self.with_W:
                self.merge = nn.Conv1d(self.feature_dim, self.feature_dim, kernel_size=1)
                self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])
            self.attention = ScaledDotProductAttention(temperature=np.power(self.feature_dim, 0.5)) # (B, N, C)
            # FFN
            if self.FFN_method == 'MLP':
                self.FFN = FFN_MLP(self.feature_dim, d_ffn=self.feature_dim*2)
            elif self.FFN_method == 'BasicBlock':
                self.FFN = FFN_BasicBlock(self.feature_dim, self.feature_dim)

    def forward(self, src):
        """
        Shape: 
        - src.size = B, c, h, w. B=batch size
        """
        B, c, h, w = src.size()
        assert c == self.feature_dim
        src = src.contiguous().view(B, c, h*w).transpose(1, 2) # (B, h*w, c)
        # feature interaction
        q = src # (B, h*w, c)
        query_embed_weight = self.query_embed.weight.unsqueeze(0).repeat(B,1,1) # (B, num_queries, c)
        k = query_embed_weight # (B, num_queries, c)
        v = query_embed_weight # (B, num_queries, c)
        if self.num_heads > 1:
            output, softmax_qk = self.attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)) # (B, c, h*w)
            ''' MLP
            output = self.mlp(torch.cat([src.transpose(1, 2), output], dim=1)) # (B, c, h*w)
            output = output.contiguous().view(B, c, h, w) # (B, c, h, w)
            '''
            #''' FFN_MLP
            output = output.transpose(1, 2) + src  # (B, h*w, c)
            output = self.FFN(output)  # (B, h*w, c)
            output = output.transpose(1, 2).contiguous().view(B, c, h, w) # (B, c, h, w)
            #'''   
        else:
            # with W
            if self.with_W:
                q, k, v = [l(x) for l, x in zip(self.proj, (q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)))]
                q = q.transpose(1, 2)
                k = k.transpose(1, 2)
                v = v.transpose(1, 2)

            output, _, softmax_qk = self.attention(q, k, v) # (B, h*w, c)
            # residual {add / product / cat+conv}
            output = output + src  # (B, h*w, c)
            # FFN
            if self.FFN_method == 'MLP':
                output = self.FFN(output)  # (B, h*w, c)
                output = output.transpose(1, 2).contiguous().view(B, c, h, w) # (B, c, h, w)
            elif self.FFN_method == 'BasicBlock':
                output = self.FFN(output.transpose(1, 2).contiguous().view(B, c, h, w)) # (B, c, h, w)
            else:
                output = output.transpose(1, 2).contiguous().view(B, c, h, w) # (B, c, h, w)
        return output, softmax_qk

