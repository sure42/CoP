import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import OpenAIGPTLMHeadModel
from transformers import AutoTokenizer, OpenAIGPTModel
from conv_tbc import ConvTBC #https://torch.mlverse.org/docs/reference/nnf_conv_tbc.html

        # gpt_loaded = torch.load(gpt_file)# 从文件加载用torch.save()保存的对象。
        # config = gpt_loaded['config']
        # gpt_model = OpenAIGPTLMHeadModel(config).cuda()
        # gpt_model.load_state_dict(gpt_loaded['model'])

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
        self.total_size = len(self.labels)
    
    def __getitem__(self, idx):
        item = {}
        for key in self.encodings:
            item[key] = self.encodings[key][idx]
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    
    def __len__(self):
        return len(self.labels)


class GPTDetector(nn.Module):
    def __init__(
            self, dictionary, embed_dim=384, in_channels = 192, 
            convolutions=[[192, 5]] * 5, dropout=0.1, embed_model_file=None, num_labels = 2
    ):
        super(GPTDetector, self).__init__()

        gpt_loaded = torch.load(embed_model_file)# 从文件加载用torch.save()保存的对象。
        config = gpt_loaded['config']
        gpt_model = OpenAIGPTLMHeadModel(config).cuda()
        gpt_model.load_state_dict(gpt_loaded['model'])
        self.embed_model = gpt_model

        self.dictionary = dictionary
        self.dropout = dropout
        self.num_labels = num_labels

        self.embed_norm = nn.LayerNorm(embed_dim)

        in_channels = convolutions[0][0]
        self.fc1 = linear(embed_dim, in_channels, dropout=dropout)
        self.convolutions = nn.ModuleList()

        layer_in_channels = [in_channels]
        for i, (out_channels, kernel_size) in enumerate(convolutions):
            if kernel_size % 2 == 1:
                padding = kernel_size // 2# “//”在Python中表示整数除法，返回不大于结果的一个最大的整数，即除法结果向下取整
            else:
                padding = 0
            self.convolutions.append(
                nn.Conv1d(in_channels, out_channels * 2, kernel_size, padding=padding) 
            )
            in_channels = out_channels
            layer_in_channels.append(out_channels)

        self.fc2 = linear(in_channels, embed_dim, dropout=dropout)

        self.linear = nn.Sequential(nn.Linear(384, 128),
                                    
                            # nn.Dropout(0.5),# 正则化层

                            nn.Tanh(),
                            nn.Linear(128, self.num_labels))
        
        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, input_ids, src_tokens, attention_mask, labels = None):# src_tokens用于提取input_ids（前文内容+bug）中的bug
        # assert share_embed_model is not None
        if input_ids is not None:
            embed = self.embed_model.transformer(
                input_ids,
                attention_mask=attention_mask,
            )[0]            # B, context_src, H 4*1024*384 batch*长度*编码
 
            bsz = embed.size(0)# 这里应该是用来记录batch
            embed = embed.view(-1, embed.size(-1))      # B x context_src, H
            mask = src_tokens.view(-1)                  # B x context_src
            mask = mask.eq(1)
            # torch.eq(input, other, *, out=None) 对两个张量Tensor进行逐元素的比较，若相同位置的两个元素相同，则返回True
            
            x = embed[mask, :]          # B x src, H
            x = x.view(bsz, -1, x.size(1))      # B, src, H


            # 在这里记录一下之前出的一些问题，
            #   1. 由于是把src和tgt拼到了一起，所以mask两个都包含，但src和tgt不一定一样长，即bug行和修复补丁不一定
            #   一样长，所以会出现无法恢复batch的情况。
            #   2. 再之前出现的问题虽然是跟1同样的表现形式，但出问题的地方不一样（也不一定，感觉可能是）。之前由于是
            #   把所有的数据先做collater，所以，src与prev_content两者都会取最长的，导致会有截断的情况，而截断一般是
            #   右截断，也就是把记录src_tokens的1截去一部分，所以会出现问题。当然也有可能是跟1一样的情况
            # 修改的方法：
            #   对2来说，不用dataloader了，逐条选择，直到够一个batch或prev + src/tgt大于1023--也就是最大长度时，作
            #   为一组数据开始训练。    gpt_detector_trainer.py line:259-264
            #   对1来说，在collater统一长度前，就把src与tgt放到一起，也就是说重新记录一组samples，把里面的src变成对
            #   应的tgt。    gpt_detector_trainer.py line:259-264

            input_ids = input_ids.view(-1)  # B x context_src
            src_tokens = input_ids[mask]      # B x src   这里之后的src_token表示的是bug行，x里是嵌入后的
            src_tokens = src_tokens.view(bsz, -1)               # B, src

        else:
            if src_tokens.is_cuda:
                attention_mask = torch.ones(src_tokens.size()).cuda().masked_fill_(
                    src_tokens == 0, 0).float().cuda()
            else:
                attention_mask = torch.ones(src_tokens.size()).masked_fill_(
                    src_tokens == 0, 0).float()
            x = self.embed_model.transformer(
                input_ids,
                attention_mask=attention_mask,
            )[0]            # B, context_src, H 4*1024*384 batch*长度*编码           

        # normalize the embedding of buggy/context lines
        x = self.embed_norm(x) # 归一化

        x = F.dropout(x, p=self.dropout, training=self.training)
        input_embedding = x

        # project to size of convolution / linear layer
        x = self.fc1(x) # B, context_src, H  H：384->192

        # used to mask padding in input
        encoder_padding_mask = src_tokens.eq(0).t()  # -> T x B 1*240->240*1

        x = x.transpose(1, 2)# B x L x H >> B x H x L L指长度

        #这里才开始本模型的训练
        for conv in self.convolutions:




            # if encoder_padding_mask is not None:
            #     x = x.masked_fill(encoder_padding_mask.unsqueeze(-1), 0)
            #     # masked_fill_(mask, value) 用value填充tensor中与mask中值为1位置相对应的元素


            x = F.dropout(x, p=self.dropout, training=self.training)
            if conv.kernel_size[0] % 2 == 1:
                # padding is implicit（含蓄的，未言明的） in the conv
                x = conv(x) # B x H x L  C:  192->384   [384,len,batch]
            else: 
                padding_l = (conv.kernel_size[0] - 1) // 2
                padding_r = conv.kernel_size[0] // 2
                x = F.pad(x, (0, 0, 0, 0, padding_l, padding_r))
                x = conv(x)
            x = F.glu(x, dim=1) # B x H x L  C: 384->192

        # T x B x C -> B x T x C
        x = x.transpose(1, 2)

        # project back to size of embedding / linear layer
        x = self.fc2(x) 
        # print(x.size()) [1,240,384] [B, L, H]
        x = x.transpose(1, 2)
        pooled_output = torch.mean(x, 2) 
        # print(pooled_output.size()) [1,240] [B, L]

        logits = self.linear(pooled_output)
        if labels is not None:
            loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        else:
            loss = logits

        return loss, logits, pooled_output


def linear(in_features, out_features, dropout=0.):
    """Weight-normalized Linear layer (input: N x T x C)"""
    m = nn.Linear(in_features, out_features)
    m.weight.data.normal_(mean=0, std=math.sqrt((1 - dropout) / in_features))
    m.bias.data.zero_()
    return m

