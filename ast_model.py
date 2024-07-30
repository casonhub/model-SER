# -*- coding: utf-8 -*-

# @Author: Martin Knutelský
# @Contact: xknute00@stud.fit.vutbr.cz
# @Date: 31/1/2023
# @File: ast_model.py
# Copyright (c) 2021, Yuan Gong
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# ORIGINAL SOURCE CODE LINK: https://github.com/YuanGongND/ast/blob/master/src/models/ast_models.py

import torch
import torch.nn as nn
import timm
from timm.models.layers import to_2tuple

# override the timm package to relax the input shape constraint.
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class ASTModel(torch.nn.Module):
    """
    The Audio Spectrogram Transformer model class.
    """
    def __init__(self, label_dim, fstride=16, tstride=16, input_fdim=64, input_tdim=512, model_size='base384', verbose=True):
        """
            Constructor of AST

            Parameters
            ----------
            label_dim : int
                The label dimension, i.e., the number of total classes
            fstride : int
                The stride of patch spliting on the frequency dimension, for 16*16 patchs, fstride=16 means no overlap, fstride=10 means overlap of 6.
            tstride : int
                The stride of patch spliting on the time dimension, for 16*16 patches, tstride=16 means no overlap, tstride=10 means overlap of 6.
            input_fdim : int
                The number of frequency bins of the input spectrogram.
            input_tdim : int
                The number of time frames of the input spectrogram.
            param model_size : str
                The model size of AST, should be in [tiny224, small224, base224, base384], base224 and base 384 are same model, but are trained differently during ImageNet pretraining.

            Returns
            ----------
            None
        """
        super(ASTModel, self).__init__()
        assert timm.__version__ == '0.4.5', 'Please use timm == 0.4.5, the code might not be compatible with newer versions.'            

        # override timm input shape restriction
        timm.models.vision_transformer.PatchEmbed = PatchEmbed

        # if AudioSet pretraining is not used (but ImageNet pretraining may still apply)
        # TODO: REWRITE CONDITION!
        if True:
            if model_size == 'tiny224':
                self.v = timm.create_model('vit_deit_tiny_distilled_patch16_224', pretrained=True)
            elif model_size == 'small224':
                self.v = timm.create_model('vit_deit_small_distilled_patch16_224', pretrained=True)
            elif model_size == 'base224':
                self.v = timm.create_model('vit_deit_base_distilled_patch16_224', pretrained=True)                
            elif model_size == 'base384':
                self.v = timm.create_model('vit_deit_base_distilled_patch16_384', pretrained=True)
            else:
                raise Exception('Model size must be one of tiny224, small224, base224, base384.')
            self.original_num_patches = self.v.patch_embed.num_patches
            self.oringal_hw = int(self.original_num_patches ** 0.5)
            self.original_embedding_dim = self.v.pos_embed.shape[2]
            self.mlp_head = nn.Sequential(nn.LayerNorm(self.original_embedding_dim), nn.Linear(self.original_embedding_dim, label_dim))

            # automatcially get the intermediate shape
            f_dim, t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim)
            num_patches = f_dim * t_dim
            self.v.patch_embed.num_patches = num_patches
            if verbose == True:
                print('---------------AST Model Summary---------------')
                print(f"Model size: {model_size} ,patch size: {tstride}x{fstride}, expected shape of mel spectrogram: {input_tdim}x{input_fdim}")
                print('number of patches={:d}'.format(num_patches))

            # the linear projection layer
            new_proj = torch.nn.Conv2d(1, self.original_embedding_dim, kernel_size=(16, 16), stride=(fstride, tstride))
            new_proj.weight = torch.nn.Parameter(torch.sum(self.v.patch_embed.proj.weight, dim=1).unsqueeze(1))
            new_proj.bias = self.v.patch_embed.proj.bias
            self.v.patch_embed.proj = new_proj

            # the positional embedding
            # if imagenet_pretrain == True:
            new_pos_embed = self.v.pos_embed[:, 2:, :].detach().reshape(1, self.original_num_patches, self.original_embedding_dim).transpose(1, 2).reshape(1, self.original_embedding_dim, self.oringal_hw, self.oringal_hw)
            # cut (from middle) or interpolate the second dimension of the positional embedding
            if t_dim <= self.oringal_hw:
                new_pos_embed = new_pos_embed[:, :, :, int(self.oringal_hw / 2) - int(t_dim / 2): int(self.oringal_hw / 2) - int(t_dim / 2) + t_dim]
            else:
                new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(self.oringal_hw, t_dim), mode='bilinear')
            # cut (from middle) or interpolate the first dimension of the positional embedding
            if f_dim <= self.oringal_hw:
                new_pos_embed = new_pos_embed[:, :, int(self.oringal_hw / 2) - int(f_dim / 2): int(self.oringal_hw / 2) - int(f_dim / 2) + f_dim, :]
            else:
                new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(f_dim, t_dim), mode='bilinear')
            # flatten the positional embedding
            new_pos_embed = new_pos_embed.reshape(1, self.original_embedding_dim, num_patches).transpose(1,2)
            # concatenate the above positional embedding with the cls token and distillation token of the deit model.
            self.v.pos_embed = nn.Parameter(torch.cat([self.v.pos_embed[:, :2, :].detach(), new_pos_embed], dim=1))

    def get_shape(self, fstride, tstride, input_fdim=64, input_tdim=512):
        """
            Obtain number of patches for each dimension.

            Parameters
            ----------
            fstride : int
                Size of stride for frequency axis.
            tstride : int
                Size of stride for time axis.
            input_fdim : int
                Size of frequency axis
            input_tdim : int
                Size of time axis

            Returns
            ----------
            f_dim : int
                Number of patches along frequency axis
            t_dim : int
                Number of patches along time axis
        """
        test_input = torch.randn(1, 1, input_fdim, input_tdim)
        test_proj = nn.Conv2d(1, self.original_embedding_dim, kernel_size=(16, 16), stride=(fstride, tstride))
        test_out = test_proj(test_input)
        f_dim = test_out.shape[2]
        t_dim = test_out.shape[3]
        return f_dim, t_dim

    def forward(self, x):
        """
            Forward propagation method
            Parameters
            ----------
                x : torch.tensor
                    the input spectrogram, expected shape: (batch_size, time_frame_num, frequency_bins), e.g., (12, 1024, 128)
            Parameters
            ----------
                output : torch.tensor
                    Prediction of model
        """
        # expect input x = (batch_size, time_frame_num, frequency_bins), e.g., (12, 1024, 128)
        x = x.unsqueeze(1)
        x = x.transpose(2, 3)

        B = x.shape[0]
        x = self.v.patch_embed(x)
        cls_tokens = self.v.cls_token.expand(B, -1, -1)
        dist_token = self.v.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)
        for blk in self.v.blocks:
            x = blk(x)
        x = self.v.norm(x)
        x = (x[:, 0] + x[:, 1]) / 2

        x = self.mlp_head(x)
        return x


# CUSTOM classifier - FINISH AFTER MAKING SURE THAT NEURAL NETWORK IS LEARNING SMTH


class Classifier(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear6 = nn.Linear(in_features, 1024)
        self.layer_norm6 = nn.LayerNorm(1024)
        self.batch_norm6 = nn.BatchNorm1d(1024)
        self.linear1 = nn.Linear(1024, 512)
        self.layer_norm1 = nn.LayerNorm(512)
        self.batch_norm1 = nn.BatchNorm1d(512)
        self.linear2 = nn.Linear(512, 256)
        self.layer_norm2 = nn.LayerNorm(256)
        self.batch_norm2 = nn.BatchNorm1d(256)
        self.linear3 = nn.Linear(256, 128)
        self.layer_norm3 = nn.LayerNorm(128)
        self.batch_norm3 = nn.BatchNorm1d(128)   
        self.linear4 = nn.Linear(128, 64)
        self.layer_norm4 = nn.LayerNorm(64)
        self.batch_norm4 = nn.BatchNorm1d(64)
        self.linear5 = nn.Linear(64, 32)
        self.layer_norm5 = nn.LayerNorm(32)
        self.batch_norm5 = nn.BatchNorm1d(32)
        self.output = nn.Linear(32, out_features)
        self.dropout = nn.Dropout(0.25)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.dropout(self.relu(self.batch_norm6(self.linear6(x))))
        x = self.dropout(self.relu(self.batch_norm1(self.linear1(x))))
        x = self.dropout(self.relu(self.batch_norm2(self.linear2(x))))
        x = self.dropout(self.relu(self.batch_norm3(self.linear3(x))))
        x = self.dropout(self.relu(self.batch_norm4(self.linear4(x))))
        x = self.dropout(self.relu(self.batch_norm5(self.linear5(x))))

#         x = self.dropout(self.relu(self.linear1(x)))
#         x = self.dropout(self.relu(self.linear2(x)))
#         x = self.dropout(self.relu(self.linear3(x)))
#         x = self.dropout(self.relu(self.linear4(x)))
        return self.output(x)