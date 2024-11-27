"parts of the code were adapted from https://github.com/sj-li/MS-TCN2?utm_source=catalyzex.com"

import torch
import torch.nn as nn
import torch.nn.functional as func
import copy
from torch.nn.utils.rnn import pack_padded_sequence


class MST_TCN2_late(nn.Module):
    def __init__(self, num_layers_PG, num_layers_R, num_R, num_f_maps, dim, num_classes_list,dropout=0.5,window_dim=0, RR_not_BF_mode=False):
        super(MST_TCN2_late, self).__init__()
        self.window_dim = window_dim
        self.RR_not_BF_mode = RR_not_BF_mode
        self.num_R = num_R
        self.PG = MT_Prediction_Generation(num_layers_PG, num_f_maps, dim, num_classes_list,dropout)

        if num_R > 0:
            self.Rs = nn.ModuleList([copy.deepcopy(MT_Refinement(num_layers_R, num_f_maps, sum(num_classes_list), num_classes_list,dropout)) for s in range(num_R)])

    def forward(self, x,*args):
        outputs=[]
        outs, _ = self.PG(x, 0, self.RR_not_BF_mode)
        for out in outs:
            outputs.append(out.unsqueeze(0))
        out = torch.cat(outs,1)

        if self.num_R >0:
            for k,R in enumerate(self.Rs):
                if k != len(self.Rs):
                    outs = R(func.softmax(out, dim=1),0, self.RR_not_BF_mode)
                else:
                    outs = R(func.softmax(out, dim=1),self.window_dim, self.RR_not_BF_mode)


                out = torch.cat(outs, 1)
                for i, output in enumerate(outputs):
                    outputs[i] = torch.cat((output, outs[i].unsqueeze(0)), dim=0)

        return outputs

class MST_TCN2_early(nn.Module):
    def __init__(self, num_layers_PG, num_layers_R, num_R, num_f_maps, dim, num_classes_list,dropout=0.5,window_dim=0, RR_not_BF_mode=False):
        super(MST_TCN2_early, self).__init__()
        self.window_dim = window_dim
        self.RR_not_BF_mode = RR_not_BF_mode
        self.num_R = num_R
        self.PG = MT_Prediction_Generation(num_layers_PG, num_f_maps, dim, num_classes_list,dropout)

        if num_R > 0:
            self.Rs = nn.ModuleList([copy.deepcopy(MT_Refinement(num_layers_R, num_f_maps, sum(num_classes_list), num_classes_list,dropout)) for s in range(num_R)])

    def forward(self, x,*args):
        outputs=[]
        outs, _ = self.PG(x, self.window_dim, self.RR_not_BF_mode)
        for out in outs:
            outputs.append(out.unsqueeze(0))
        out = torch.cat(outs,1)

        if self.num_R >0:
            for R in self.Rs:
                outs = R(func.softmax(out, dim=1),0, self.RR_not_BF_mode)
                out = torch.cat(outs, 1)
                for i, output in enumerate(outputs):
                    outputs[i] = torch.cat((output, outs[i].unsqueeze(0)), dim=0)

        return outputs


class MST_TCN2(nn.Module):
    def __init__(self, num_layers_PG, num_layers_R, num_R, num_f_maps, dim, num_classes_list, dropout=0.5, w_max=0, RR_not_BF_mode=False, use_dynamic_wmax=False):
        super(MST_TCN2, self).__init__()
        self.w_max = w_max
        self.use_dynamic_wmax = use_dynamic_wmax # TODO dynamic w_max
        self.RR_not_BF_mode = RR_not_BF_mode
        self.num_R = num_R
        self.PG = MT_Prediction_Generation(num_layers_PG, num_f_maps, dim, num_classes_list,dropout, use_dynamic_wmax=False)

        if num_R > 0:
            self.Rs = nn.ModuleList([copy.deepcopy(MT_Refinement(num_layers_R, num_f_maps, sum(num_classes_list), num_classes_list,dropout, use_dynamic_wmax=False)) for s in range(num_R)])

    def forward(self, x, *args):
        outputs = []
        outs, _, dynamic_wmax = self.PG(x, self.w_max, self.RR_not_BF_mode) # TODO dynamic w_max
        for out in outs:
            outputs.append(out.unsqueeze(0))
        out = torch.cat(outs, 1)
        
        # TODO dynamic w_max
        layer_cnt           = 0
        total_dynamic_w_max   = 0
        if self.num_R > 0:
            for R in self.Rs:
                # TODO dynamic w_max
                total_dynamic_w_max += int(torch.mean(dynamic_wmax).item() + 0.5) # take mean and convert to integer
                avg_dynamic_w_max = int(total_dynamic_w_max / (layer_cnt + 1))
                layer_cnt += 1
                outs, dynamic_w_max = R(func.softmax(out, dim=1), self.w_max if not self.use_dynamic_wmax else avg_dynamic_w_max, self.RR_not_BF_mode) # TODO dynamic w_max
                out = torch.cat(outs, 1)
                for i, output in enumerate(outputs):
                    outputs[i] = torch.cat((output, outs[i].unsqueeze(0)), dim=0)

        return outputs, int(torch.mean(dynamic_w_max).item() + 0.5) # TODO dynamic w_max


class MT_Prediction_Generation_many_heads(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes_list, dropout=0.5):
        super(MT_Prediction_Generation_many_heads, self).__init__()

        self.layers_heads = self.heads_config(num_layers)
        self.num_layers = num_layers

        self.conv_1x1_in = nn.Conv1d(dim, num_f_maps, 1)

        self.conv_dilated_1 = nn.ModuleList((
            Dilated_conv(num_f_maps, 3, dilation=2**(num_layers-1-i))
            for i in range(num_layers)
        ))

        self.conv_dilated_2 = nn.ModuleList((
            Dilated_conv(num_f_maps, 3, dilation=2**i)
            for i in range(num_layers)
        ))

        self.conv_fusion = nn.ModuleList((
            nn.Conv1d(2*num_f_maps, num_f_maps, 1)
            for i in range(num_layers)

        ))

        self.dropout = nn.Dropout(dropout)

        self.heads = nn.ModuleList([copy.deepcopy(
            nn.ModuleList([copy.deepcopy(
                nn.Conv1d(num_f_maps, num_classes_list[s], 1))
                for s in range(len(num_classes_list))]))
                                 for i in range(len(self.layers_heads))])

    def heads_config(self, num_of_layers):
        heads = []
        for i in ([4, 7, 10]):
            if i < num_of_layers - 1:
                heads.append(i)
        heads.append(num_of_layers-1)
        return heads

    def forward(self, x, w_max, offline_mod):
        optputs = []
        outs = []
        featutes = []

        f = self.conv_1x1_in(x)

        for i in range(self.num_layers):
            f_in = f
            f = self.conv_fusion[i](torch.cat([self.conv_dilated_1[i](f, w_max, offline_mod), self.conv_dilated_2[i](f, w_max, offline_mod)], 1))
            f = func.relu(f)
            f = self.dropout(f)
            f = f + f_in
            if i in self.layers_heads:
                featutes.append(f)
                index = self.layers_heads.index(i)
                for conv_out in self.heads[index]:
                    outs.append(conv_out(f))
                optputs.append(outs)
                outs = []

        return optputs, featutes


class MT_Prediction_Generation(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes_list, dropout, use_dynamic_wmax=False):
        super(MT_Prediction_Generation, self).__init__()

        self.num_layers = num_layers

        self.conv_1x1_in = nn.Conv1d(dim, num_f_maps, 1)

        self.conv_dilated_1 = nn.ModuleList((
            Dilated_conv(num_f_maps, 3, dilation=2**(num_layers-1-i))
            for i in range(num_layers)
        ))

        self.conv_dilated_2 = nn.ModuleList((
            Dilated_conv(num_f_maps, 3, dilation=2**i)
            for i in range(num_layers)
        ))

        self.conv_fusion = nn.ModuleList((
            nn.Conv1d(2*num_f_maps, num_f_maps, 1)
            for i in range(num_layers)

        ))

        self.dropout = nn.Dropout(dropout)

        self.conv_outs = nn.ModuleList([copy.deepcopy(
            nn.Conv1d(num_f_maps, num_classes_list[s], 1))
                                 for s in range(len(num_classes_list))])

        # TODO - dynamic w_max
        self.use_dynamic_wmax = use_dynamic_wmax
        self.fc_wmax = nn.Linear(num_f_maps, 1) # predict dynamic w_max

    def forward(self, x, w_max, RR_not_BF_mode):
        outs = []
        f = self.conv_1x1_in(x)

        # TODO - dynamic w_max
        MAX_WMAX = w_max

        for i in range(self.num_layers):
            f_in = f
            f = self.conv_fusion[i](torch.cat([self.conv_dilated_1[i](f, w_max, RR_not_BF_mode), self.conv_dilated_2[i](f, w_max, RR_not_BF_mode)], 1))
            f = func.relu(f)
            f = self.dropout(f)
            f = f + f_in
        
        # TODO - dynamic w_max
        if not self.use_dynamic_wmax:
            dynamic_w_max =torch.tensor([[MAX_WMAX], [MAX_WMAX]], dtype=float)
        else:
            # Apply global average pooling to reduce `f` to shape (batch_size, num_features)
            f_pooled = torch.mean(f, dim=2)  # Shape: (batch_size, num_features)
            # Pass through the fully connected layer to predict `w_max`
            wmax_logits = self.fc_wmax(f_pooled)  # Predict w_max logits
            dynamic_w_max = torch.sigmoid(wmax_logits) * MAX_WMAX  # Scale to [0, MAX_WMAX]
            # Log batch-wise `w_max` values # TODO log in table
            # print(f"Batch-wise wmax values(PG):     {dynamic_w_max.squeeze().cpu().tolist()}")
            # with open("wmax_log.txt", "a") as log_file:
            #     log_file.write(f"Batch-wise wmax values(PG):    {dynamic_w_max.squeeze().cpu().tolist()}\n")

            # Compute the average across the batch to get a single scalar w_max
            avg_dynamic_w_max = torch.mean(dynamic_w_max)  # Shape: ()
            # print(f"Averaged wmax value(PG):        {avg_dynamic_w_max.item()}")
            # with open("wmax_log.txt", "a") as log_file:
            #     log_file.write(f"Averaged wmax value(PG):       {avg_dynamic_w_max.item()}\n")

        for conv_out in self.conv_outs:
            outs.append(conv_out(f))
        
        return outs, f, dynamic_w_max # TODO - dynamic w_max


class MT_Refinement(nn.Module):  # refinement stage
    def __init__(self, num_layers, num_f_maps, dim, num_classes_list, dropout=0.5, use_dynamic_wmax=False):
        super(MT_Refinement, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList([copy.deepcopy(DilatedResidualLayer(2**i, num_f_maps, num_f_maps,dropout=dropout)) for i in range(num_layers)])
        self.conv_outs = nn.ModuleList([copy.deepcopy(nn.Conv1d(num_f_maps, num_classes_list[s], 1)) for s in range(len(num_classes_list))])
        # TODO - dynamic w_max
        self.use_dynamic_wmax = use_dynamic_wmax
        self.refine_wmax = nn.Linear(num_f_maps, 1) # Refine predicted dynamic w_max
        
    def forward(self, x, w_max, RR_not_BF_mode):
        outs = []
        f = self.conv_1x1(x)

        # TODO - dynamic w_max
        MAX_WMAX = w_max

        for layer in self.layers:
            f = layer(f, w_max, RR_not_BF_mode)
        # TODO - dynamic w_max
        if not self.use_dynamic_wmax:
            dynamic_w_max =torch.tensor([[MAX_WMAX], [MAX_WMAX]], dtype=float)
        else:
            # Apply global average pooling to reduce `f` to shape (batch_size, num_features)
            f_pooled = torch.mean(f, dim=2)  # Shape: (batch_size, num_features)
            # Pass through the fully connected layer to predict `w_max`
            wmax_logits = self.refine_wmax(f_pooled)  # Predict w_max logits
            dynamic_w_max = torch.sigmoid(wmax_logits) * MAX_WMAX  # Scale to [0, MAX_WMAX]
            # Log batch-wise `w_max` values
            # print(f"Batch-wise wmax values(R ):     {dynamic_w_max.squeeze().cpu().tolist()}")
            # with open("wmax_log.txt", "a") as log_file:
            #     log_file.write(f"Batch-wise wmax values(R ):    {dynamic_w_max.squeeze().cpu().tolist()}\n")

            # Compute the average across the batch to get a single scalar w_max
            avg_dynamic_w_max = torch.mean(dynamic_w_max)  # Shape: ()
            # print(f"Averaged wmax value(R ):        {avg_dynamic_w_max.item()}")
            # with open("wmax_log.txt", "a") as log_file:
            #     log_file.write(f"Averaged wmax value(R ):       {avg_dynamic_w_max.item()}\n")
        for conv_out in self.conv_outs:
            outs.append(conv_out(f))
        return outs, dynamic_w_max # TODO - dynamic w_max


class Dilated_conv(nn.Module):
    def __init__(self, num_f_maps, karnel_size, dilation):
        super(Dilated_conv, self).__init__()
        self.dilation = dilation
        self.Dilated_conv = nn.Conv1d(num_f_maps, num_f_maps, karnel_size, dilation=dilation)
        # the dilation seperates the frames

    def forward(self, x, w_max, RR_not_BF_mode):
        if RR_not_BF_mode:
            out = self.Acausal_padding(x, self.dilation)
        else:
            out = self.window_padding(x, self.dilation, w_max)
        out = self.Dilated_conv(out)
        return out

    def Acausal_padding(self, input, padding_dim):
        padding = torch.zeros(input.shape[0],input.shape[1],padding_dim).to(input.device)
        return torch.cat((padding,input,padding),2)

    def window_padding(self, input, padding_dim, w_max=0):
        if w_max > padding_dim:
            w_max = padding_dim
        padding_left = torch.zeros(input.shape[0], input.shape[1], 2*padding_dim - w_max).to(input.device)
        padding_right = torch.zeros(input.shape[0], input.shape[1], w_max).to(input.device)
        return torch.cat((padding_left, input, padding_right), 2)


class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels, dropout=0.5):
        super(DilatedResidualLayer, self).__init__()
        self.dilation = dilation
        self.conv_dilated = nn.Conv1d(in_channels, out_channels, 3, padding=0, dilation=dilation) # In the original code padding=dilation
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, w_max, RR_not_BF_mode):
        if RR_not_BF_mode: # RR - offline mode
            out = self.Acausal_padding(x, self.dilation)
        else: # BF - online mode
            out = self.window_padding(x, self.dilation, w_max)

        out = func.relu(self.conv_dilated(out))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return x + out

    def Acausal_padding(self, input, padding_dim):
        padding = torch.zeros(input.shape[0],input.shape[1],padding_dim).to(input.device)
        return torch.cat((padding,input,padding),2)

    def window_padding(self, input, padding_dim, w_max):
        if w_max > padding_dim:
            w_max = padding_dim
        padding_left = torch.zeros(
            input.shape[0], input.shape[1], 2*padding_dim - w_max).to(input.device)
        padding_right = torch.zeros(
            input.shape[0], input.shape[1], w_max).to(input.device)
        return torch.cat((padding_left, input, padding_right), 2)
