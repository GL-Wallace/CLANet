from __future__ import division
from torch.autograd import Variable
from glam.glam import GLAM
import torch
import torchvision.models as models
import torch.nn as nn
from utils import outputActivation
import torch.nn.functional as F
from attentionMap import plotTemporalAttentionMap, plotSpatialAttentionMap


class CLANet(nn.Module):

    # Initialization
    def __init__(self, args):
        super(CLANet, self).__init__()

        # Unpack arguments
        self.args = args

        # Use gpu flag
        self.use_cuda = args['use_cuda']

        # Flag for train mode (True) vs test-mode (False)
        self.train_flag = args['train_flag']

        # Sizes of network layers
        self.encoder_size = args['encoder_size']
        self.decoder_size = args['decoder_size']
        self.in_length = args['in_length']
        self.out_length = args['out_length']
        self.grid_size = args['grid_size']

        self.input_embedding_size = args['input_embedding_size']

        # Define network weights

        # CNN layer for embedding
        self.cnn = VGG16(in_channels=cnn_in_channels, out_nodes=regresor_input_from_cnn)

        # Input embedding layer
        self.ip_emb = torch.nn.Linear(2, self.input_embedding_size)

        # Encoder LSTM
        self.enc_lstm1 = torch.nn.LSTM(self.input_embedding_size, self.encoder_size, 1)

        # Encoder LSTM
        self.enc_lstm2 = torch.nn.LSTM(self.input_embedding_size, self.encoder_size, 1)

        self.spatial_embedding = nn.Linear(5, self.encoder_size)

        self.tanh = nn.Tanh()

        self.pre4att = nn.Sequential(
            nn.Linear(self.encoder_size, 1),
        )

        self.dec_lstm = torch.nn.LSTM(self.encoder_size, self.decoder_size)

        # Output layers:
        self.op = torch.nn.Linear(self.decoder_size, 2)  # 2-dimension (x, y)

        # Activations:
        self.leaky_relu = torch.nn.LeakyReLU(0.1)
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)

    def attention(self, lstm_out_weight, lstm_out):
        alpha = F.softmax(lstm_out_weight, 1)

        lstm_out = lstm_out.permute(0, 2, 1)

        new_hidden_state = torch.bmm(lstm_out, alpha).squeeze(2)
        new_hidden_state = F.relu(new_hidden_state)

        return new_hidden_state, alpha

    # Forward Pass
    def forward(self, hist, nbrs, masks, lat_enc, lon_enc):
        # Forward pass hist:
        lstm_out, (hist_enc, _) = self.enc_lstm1(self.leaky_relu(self.ip_emb(hist)))

        lstm_out = lstm_out.permute(1, 0, 2)
        lstm_weight = self.pre4att(self.tanh(lstm_out))
        new_hidden, soft_attn_weights = self.attention(lstm_weight, lstm_out)
        plotTemporalAttentionMap(soft_attn_weights)
        new_hidden = new_hidden.unsqueeze(2)

        # Forward pass nbrs

        nbrs_out, (nbrs_enc, _) = self.enc_lstm1(self.leaky_relu(self.ip_emb(nbrs)))
        # apply attention mechanism to neighbors
        nbrs_out = nbrs_out.permute(1, 0, 2)

        nbrs_lstm_weight = self.pre4att(self.tanh(nbrs_out))

        new_nbrs_hidden, soft_nbrs_attn_weights = self.attention(nbrs_lstm_weight, nbrs_out)
        nbrs_enc = new_nbrs_hidden

        # Masked scatter
        soc_enc = torch.zeros_like(masks).float()  # mask size: (128, 3, 13, 64)
        soc_enc = soc_enc.masked_scatter_(masks, nbrs_enc)

        # masks_tem = masks.permute(0, 3, 2, 1)

        soc_enc = soc_enc.permute(0, 3, 2, 1)
        soc_enc = soc_enc.contiguous().view(soc_enc.shape[0], soc_enc.shape[1], -1)

        # concatenate hidden states:
        new_hs = torch.cat((soc_enc, new_hidden), 2)
        new_hs_per = new_hs.permute(0, 2, 1)

        # second attention
        weight = self.pre4att(self.tanh(new_hs_per))

        new_hidden_ha, soft_attn_weights_ha = self.attention(weight, new_hs_per)

        # Concatenate encodings:
        enc = new_hidden_ha

        fut_pred = self.decode(enc)

        plotSpatialAttentionMap(soft_attn_weights_ha, masks)
        return fut_pred, soft_attn_weights, soft_nbrs_attn_weights, soft_attn_weights_ha

        # soft_attn_weights and soft_nbrs_attn_weights are the attention weights across time steps (the ego-vehicle and neighbors)
        # soft_attn_weights_ha are the attention weights across vehicles (13 by 3 neighbors, row-wise flattern, [[1, 2, 3], [4, 5, 6], ...[X, X, X]], the 40th is the ego vehicle)

    def decode(self, enc):
        enc = enc.repeat(self.out_length, 1, 1)

        h_dec, _ = self.dec_lstm(enc)
        h_dec = h_dec.permute(1, 0, 2)
        fut_pred = self.op(h_dec)
        fut_pred = fut_pred.permute(1, 0, 2)
        fut_pred = outputActivation(fut_pred)
        return fut_pred

    def decode_by_step(self, enc):
        pre_traj = []

        decoder_input = enc

        for _ in range(self.out_length):
            decoder_input = decoder_input.unsqueeze(0)
            h_dec, _ = self.dec_lstm(decoder_input)
            h_for_pred = h_dec.squeeze()
            fut_pred = self.op(h_for_pred)
            pre_traj.append(fut_pred.view(fut_pred.size()[0], -1))

            embedding_input = fut_pred
            decoder_input = self.spatial_embedding(embedding_input)

        pre_traj = torch.stack(pre_traj, dim=0)
        pre_traj = outputActivation(pre_traj)
        return pre_traj


class VGG16(nn.Module):
    def __init__(self, in_channels=14, out_nodes=1024):
        super().__init__()
        self.vgg16 = models.vgg16(weights=None)
        self.vgg16.features[0] = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1)
        self.relu = nn.LeakyReLU()
        self.vgg16.classifier[6] = nn.Linear(4096, out_nodes)

    def forward(self, x):
        x = self.vgg16(x)
        x = self.relu(x)
        return x


class VGG16GLAM(VGG16):
    def __init__(self, in_channels=14, out_nodes=1024, glam_reduce_channels=32):
        super().__init__(in_channels, out_nodes)
        self.glam = GLAM(in_channels=512, num_reduced_channels=glam_reduce_channels, feature_map_size=8, kernel_size=5)
        self.vgg16.features[22] = nn.Sequential(self.vgg16.features[22], self.glam)

