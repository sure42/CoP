import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from conv_tbc import ConvTBC #https://torch.mlverse.org/docs/reference/nnf_conv_tbc.html


class GPTCoNuTModel(nn.Module):
    def __init__(
            self, dictionary, embed_dim=768, max_positions=1024,
            src_encoder_convolutions=((192, 5),) * 5,
            ctx_encoder_convolutions=((384, 5),) * 7,
            decoder_convolutions=((192, 5),) * 5,
            dropout=0.1, embed_model=None,
    ):
        super(GPTCoNuTModel, self).__init__()
        self.embed_dim = embed_dim
        self.max_positions = max_positions
        self.dictionary = dictionary
        self.src_encoder_convolutions = src_encoder_convolutions
        self.ctx_encoder_convolutions = ctx_encoder_convolutions
        self.decoder_convolutions = decoder_convolutions
        self.embed_model = embed_model

        self.encoder = GPTCoNuTEncoder(
            dictionary, embed_dim, max_positions,
            src_encoder_convolutions, ctx_encoder_convolutions, dropout
        )
        self.decoder = GPTFConvDecoder(# 改成CNN
            dictionary, embed_dim, max_positions,
            decoder_convolutions, dropout,
        )

    def config(self):
        info = dict()
        info['embed_dim'] = self.embed_dim
        info['max_positions'] = self.max_positions
        info['src_encoder_convolutions'] = self.src_encoder_convolutions
        info['ctx_encoder_convolutions'] = self.ctx_encoder_convolutions
        info['decoder_convolutions'] = self.decoder_convolutions
        info['embed_model_config'] = self.embed_model.config
        return info

    def forward(self, src_tokens, src_tokens_with_pre_context, ctx_tokens,
                prev_tokens_index, prev_tokens_with_context=None, labels=None):
        encoder_out = self.encoder(
            src_tokens, src_tokens_with_pre_context,
            ctx_tokens, share_embed_model=self.embed_model
        )
        decoder_out = self.decoder(
            prev_tokens_index, encoder_out,
            prev_tokens_with_context,
            share_embed_model=self.embed_model,
            output_lm_logits=True,
        )

        if labels is not None:
            logits, avg_attn_scores, lm_logits = decoder_out
            loss_fct = nn.NLLLoss()# 损失函数
            
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., :].contiguous()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            shift_lm_logits = lm_logits[..., :-2, :].contiguous()
            shift_labels = prev_tokens_with_context[:, 1:-1].contiguous()
            lm_loss = loss_fct(shift_lm_logits.view(-1, shift_lm_logits.size(-1)), shift_labels.view(-1))

            decoder_out = (logits, avg_attn_scores, loss, lm_loss)

        return decoder_out


class GPTFConvEncoder(nn.Module):# encoder内部的编码
    def __init__(
            self, dictionary, embed_dim=768, max_positions=1024,
            convolutions=((192, 5),) * 5, dropout=0.1,
    ):
        super(GPTFConvEncoder, self).__init__()
        self.dictionary = dictionary
        self.dropout = dropout

        self.embed_norm = nn.LayerNorm(embed_dim)

        convolutions = extend_conv_spec(convolutions)
        in_channels = convolutions[0][0]
        self.fc1 = linear(embed_dim, in_channels, dropout=dropout)
        self.convolutions = nn.ModuleList()
        self.attentions = nn.ModuleList()
        # self.norms = nn.ModuleList()
        # self.residuals = []

        layer_in_channels = [in_channels]
        for i, (out_channels, kernel_size, residual) in enumerate(convolutions):
            if kernel_size % 2 == 1:
                padding = kernel_size // 2# “//”在Python中表示整数除法，返回不大于结果的一个最大的整数，即除法结果向下取整
            else:
                padding = 0
            self.convolutions.append(
                convtbc(in_channels, out_channels * 2, kernel_size,
                        dropout=dropout, padding=padding)
            )
            self.attentions.append(
                AttentionLayer(out_channels, out_channels) if i == len(convolutions) - 1 else None
            )
            # self.norms.append(nn.LayerNorm(out_channels))# 做归一化
            # self.residuals.append(residual)# 残差层
            in_channels = out_channels
            layer_in_channels.append(out_channels)
        self.fc2 = linear(in_channels, embed_dim)

    def forward(self, src_tokens, src_tokens_with_prev_context=None, share_embed_model=None):
        assert share_embed_model is not None
        if src_tokens_with_prev_context is not None:
            if src_tokens.is_cuda:
                attention_mask = torch.ones(src_tokens_with_prev_context.size()).cuda().masked_fill_(
                    src_tokens_with_prev_context == 0, 0).float().cuda() # B,L batch_size和长度
            else:
                attention_mask = torch.ones(src_tokens_with_prev_context.size()).masked_fill_(
                    src_tokens_with_prev_context == 0, 0).float()
            # torch.ones返回一个全为1 的张量，形状由可变参数sizes定义。
            # masked_fill_(mask, value) 用value填充tensor中与mask中值为1位置相对应的元素。mask的形状必须与要填充的tensor形状一致。
            #   这里也就是将所有原值为0的地方用0填充，src_tokens_with_prev_context有一部分是用0填充的
            embed = share_embed_model(
                src_tokens_with_prev_context,
                attention_mask=attention_mask,
            )[0]            # B, context_src, H

            bsz = embed.size(0)# 这里应该是用来记录大小的-B条数据
            embed = embed.view(-1, embed.size(-1))      # B x context_src, H
            mask = src_tokens.view(-1)                  # B x context_src  这里的src_token是区分bug与前文部分
            mask = mask.eq(1)# 取出bug行部分
            # torch.eq(input, other, *, out=None) 对两个张量Tensor进行逐元素的比较，若相同位置的两个元素相同，则返回True
            
            x = embed[mask, :]          # B x src, H
            x = x.view(bsz, -1, x.size(1))      # B, src, H x记录的是

            src_tokens_with_pre_context = src_tokens_with_prev_context.view(-1)  # B x context_src
            src_tokens = src_tokens_with_pre_context[mask]      # B x src
            src_tokens = src_tokens.view(bsz, -1)               # B, src
            # 130行到这里都是调整size
        else:# 如果只有bug行，所以不用再提取了
            if src_tokens.is_cuda:
                attention_mask = torch.ones(src_tokens.size()).cuda().masked_fill_(
                    src_tokens == 0, 0).float().cuda()
            else:
                attention_mask = torch.ones(src_tokens.size()).masked_fill_(
                    src_tokens == 0, 0).float()
            x = share_embed_model(
                src_tokens,
                attention_mask=attention_mask,
            )[0]  # B, context, H

        # normalize the embedding of buggy/context lines
        x = self.embed_norm(x)

        x = F.dropout(x, p=self.dropout, training=self.training)
        input_embedding = x

        # project to size of convolution / linear layer

        x = self.fc1(x)

        # used to mask padding in input
        encoder_padding_mask = src_tokens.eq(0).t()  # -> T x B

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # residuals = [x]
        #这里才开始本模型的训练
        # temporal convolutions / convolutional layer / fconv part 时域卷积/卷积层/ fconv部分
        # for conv, attention, res_layer, norm in zip(self.convolutions, self.attentions,
        #                                             self.residuals, self.norms):
        for conv, attention in zip(self.convolutions, self.attentions):
            # if res_layer > 0:
            #     residual = residuals[-res_layer]
            # else:
            #     residual = None

            if encoder_padding_mask is not None:
                x = x.masked_fill(encoder_padding_mask.unsqueeze(-1), 0)
            target_embedding = x.transpose(0, 1)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if conv.kernel_size[0] % 2 == 1:
                # padding is implicit（含蓄的，未言明的） in the conv
                x = conv(x)
            else:
                padding_l = (conv.kernel_size[0] - 1) // 2
                padding_r = conv.kernel_size[0] // 2
                x = F.pad(x, (0, 0, 0, 0, padding_l, padding_r))
                x = conv(x)
            x = F.glu(x, dim=2)

            # self attention
            if attention is not None:
                x = x.transpose(0, 1)
                x_, _ = attention(x, target_embedding, (x.transpose(1, 2).contiguous(), x), encoder_padding_mask.t())
                x = torch.cat([x, x_], dim=2)
                x = F.glu(x, dim=2)
                x = x.transpose(0, 1)

            # if residual is not None:
            #     x = (x + residual) * math.sqrt(0.5)
            # x = norm(x)
            # residuals.append(x)

        # T x B x C -> B x T x C
        x = x.transpose(1, 0)

        # project back to size of embedding / linear layer
        x = self.fc2(x)

        if encoder_padding_mask is not None:
            encoder_padding_mask = encoder_padding_mask.t()  # -> B x T
            x = x.masked_fill(encoder_padding_mask.unsqueeze(-1), 0)# unsqueeze起升维的作用,参数表示在哪个地方加一个维度,在第一个维度(中括号)的每个元素加中括号，0表示在张量最外层加一个中括号变成第一维

        # add output to input embedding for attention
        y = (x + input_embedding) * math.sqrt(0.5)

        return {
            'src_tokens': src_tokens,
            'encoder_out': (x, y),
            'encoder_padding_mask': encoder_padding_mask,  # B x T
        }


class GPTCoNuTEncoder(nn.Module):# 整体的encoder
    def __init__(
            self, dictionary, embed_dim=768, max_positions=1024,
            src_convolutions=((192, 5),) * 5, ctx_convolutions=((384, 5),) * 7,
            dropout=0.1,
    ):
        super(GPTCoNuTEncoder, self).__init__()
        self.src_encoder = GPTFConvEncoder(
            dictionary, embed_dim, max_positions, src_convolutions, dropout,
        )
        self.context_encoder = GPTFConvEncoder(
            dictionary, embed_dim, max_positions, ctx_convolutions, dropout,
        )

    def forward(self, src_tokens, src_tokens_with_prev_context, ctx_tokens, share_embed_model=None):
        # encode the buggy lines
        src_output = self.src_encoder.forward(
            src_tokens,
            src_tokens_with_prev_context=src_tokens_with_prev_context,
            share_embed_model=share_embed_model,
        )
        # encode the context lines
        ctx_output = self.context_encoder.forward(
            ctx_tokens,
            src_tokens_with_prev_context=None,
            share_embed_model=share_embed_model,
        )
        if src_output['encoder_padding_mask'] is None or ctx_output['encoder_padding_mask'] is None:
            encoder_padding_mask = None
        else:
            encoder_padding_mask = torch.cat(
                [src_output['encoder_padding_mask'],
                 ctx_output['encoder_padding_mask']], 1
            )
        return {
            'src_tokens': torch.cat([src_output['src_tokens'], ctx_output['src_tokens']], 1),
            'encoder_out': (torch.cat([src_output['encoder_out'][0], ctx_output['encoder_out'][0]],1),
                            torch.cat([src_output['encoder_out'][1], ctx_output['encoder_out'][1]],1)),
            'encoder_padding_mask': encoder_padding_mask
        }


class GPTFConvDecoder(nn.Module):
    def __init__(
            self, dictionary, embed_dim=768, max_positions=1024,
            convolutions=((192, 5),) * 5, dropout=0.1,
    ):
        super(GPTFConvDecoder, self).__init__()
        self.dropout = dropout

        self.embed_norm = nn.LayerNorm(embed_dim)

        convolutions = extend_conv_spec(convolutions)
        in_channels = convolutions[0][0]
        self.fc1 = linear(embed_dim, in_channels, dropout=dropout)
        self.convolutions = nn.ModuleList()
        self.attentions = nn.ModuleList()
        # self.norms = nn.ModuleList()
        # self.residuals = []

        layer_in_channels = [in_channels]
        for i, (out_channels, kernel_size, residual) in enumerate(convolutions):
            self.convolutions.append(
                convtbc(in_channels, out_channels * 2, kernel_size,
                        padding=(kernel_size - 1), dropout=dropout, remove_future=True)
            )
            self.attentions.append(AttentionLayer(out_channels, embed_dim))
            # self.norms.append(nn.LayerNorm(out_channels))
            # self.residuals.append(residual)
            in_channels = out_channels
            layer_in_channels.append(out_channels)

        self.fcg = linear(embed_dim + out_channels, 1)
        self.fc2 = linear(in_channels, len(dictionary))

    def forward(self, prev_tokens_index, encoder_out_dict,
                prev_tokens_with_context=None, share_embed_model=None, output_lm_logits=False):
        # prev_tokens_index---tgt_index 标记prev_context和原tgt_tokens
        # encoder_out_dict---encoder_out
        # prev_tokens_with_context---target_with_prev_context

        src_tokens = encoder_out_dict['src_tokens']
        encoder_out = encoder_out_dict['encoder_out']
        encoder_padding_mask = encoder_out_dict['encoder_padding_mask'] # 这个值是为了计算自注意力
        encoder_a, encoder_b = self._split_encoder_out(encoder_out)# x,y

        assert prev_tokens_with_context is not None
        if prev_tokens_index.is_cuda:
            attention_mask = torch.ones(prev_tokens_with_context.size()).cuda().masked_fill_(# 用value填充tensor中与mask中值为1位置相对应的元素
                prev_tokens_with_context == 0, 0).float().cuda()
        else:
            attention_mask = torch.ones(prev_tokens_with_context.size()).masked_fill_(
                prev_tokens_with_context == 0, 0).float()

        # get the embedding of the decoded sequence from gpt
        embed = share_embed_model.transformer(
            prev_tokens_with_context,
            attention_mask=attention_mask,
        )[0]  # B, context_tgt, H
        lm_logits = None
        if output_lm_logits:
            lm_logits = share_embed_model.lm_head(embed)
            # self.lm_head()输出层将GPT2Model(config)计算输出的hidden_states张量的最后一个维度由768维(config.n_embd)
            # 投影为词典大小维度(config.vocab_size)的输出层, 此时hidden_states张量的形状将会由(batch_size, 1, n_embed)投影变为
            # lm_logits张量的(batch_size, 1, vocab_size)
            lm_logits = F.log_softmax(lm_logits, dim=-1)    # B, context_tgt, H
        bsz = embed.size(0)
        mask = prev_tokens_index.view(-1)  # B x context_tgt
        mask = mask.eq(1) # 做遮盖
        embed = embed.view(-1, embed.size(-1))  # B x context_tgt, H

        # take out the target part / exclude context before
        x = embed[mask, :]  # B x tgt, H 取出tgt的对象，去除prev对应的内容

        x = x.view(bsz, -1, x.size(1))  # B, tgt, H

        x = self.embed_norm(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        target_embedding = x

        # project to size of convolution
        x = self.fc1(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # 313到352与114到168的逻辑是一致的，都是将内容输入到GPT中进行提取特征，再调整torch大小

        avg_attn_scores = None
        copy_scores = None
        num_attn_layers = len(self.attentions)
        # residuals = [x]
        # for conv, attention, res_layer, norm in zip(self.convolutions, self.attentions,
        #                                             self.residuals, self.norms):
        for conv, attention in zip(self.convolutions, self.attentions):        
            # if res_layer > 0:
            #     residual = residuals[-res_layer]
            # else:
            #     residual = None

            x = F.dropout(x, p=self.dropout, training=self.training)
            x = conv(x)# 对应184-191
            x = F.glu(x, dim=2)

            # attention     T x B x C -> B x T x C
            x = x.transpose(0, 1)
            x, attn_scores = attention(x, target_embedding, (encoder_a, encoder_b), encoder_padding_mask)
            copy_scores = attn_scores
            attn_scores = attn_scores / num_attn_layers
            if avg_attn_scores is None:
                avg_attn_scores = attn_scores
            else:
                avg_attn_scores.add_(attn_scores)
            x = x.transpose(0, 1)

            # if residual is not None:
            #     x = (x + residual) * math.sqrt(0.5)
            # x = norm(x)
            # residuals.append(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        # B x T x [C + E]
        h = torch.cat([x, target_embedding], dim=-1)
        p_gen = torch.sigmoid(self.fcg(h))

        # project back to size of vocabulary
        # B x T x C
        x = self.fc2(x)

        x = F.softmax(x, dim=-1)

        x = x * p_gen

        x = x.scatter_add(
            2, src_tokens.unsqueeze(1).repeat(1, x.size(1), 1),
            copy_scores * (1 - p_gen)
        )

        x = torch.log(x + 1e-32)
        return x, avg_attn_scores, lm_logits

    def _split_encoder_out(self, encoder_out):
        """Split and transpose encoder outputs."""
        # transpose only once to speed up attention layers
        encoder_a, encoder_b = encoder_out
        #print(encoder_a.size(), encoder_b.size())
        encoder_a = encoder_a.transpose(1, 2).contiguous()
        result = (encoder_a, encoder_b)
        return result

class AttentionLayer(nn.Module):
    def __init__(self, conv_channels, embed_dim):
        super().__init__()
        # projects from output of convolution to embedding dimension
        self.in_projection = linear(conv_channels, embed_dim)
        # projects from embedding dimension to convolution size
        self.out_projection = linear(embed_dim, conv_channels)

        self.bmm = torch.bmm# 计算两个tensor的矩阵乘法

    def forward(self, x, target_embedding, encoder_out, encoder_padding_mask):
        residual = x
        x = (self.in_projection(x) + target_embedding) * math.sqrt(0.5)
        #print(x.size(), encoder_out[0].size(), encoder_out[1].size())
        # B x L x C, B x C x L
        x = self.bmm(x, encoder_out[0])

        x = x.float().masked_fill(
            encoder_padding_mask.unsqueeze(1),
            float('-inf')
        ).type_as(x)

        sz = x.size()
        x = F.softmax(x.view(sz[0] * sz[1], sz[2]), dim=1)
        x = x.view(sz)
        attn_scores = x

        x = self.bmm(x, encoder_out[1])
        s = encoder_out[1].size(1)
        # device = 'cuda' if torch.cuda.is_available() else 'cpu'
        device = x.device
        if s == 0:
            x = x * math.sqrt(s)
        else:
            x = x * (s * math.sqrt(1.0 / s))

        x = (self.out_projection(x) + residual) * math.sqrt(0.5)
        return x, attn_scores


def extend_conv_spec(convolutions):# 把卷积核调整为三维
    """
    Extends convolutional spec that is a list of tuples of 2 or 3 parameters
    (kernel size, dim size and optionally how many layers behind to look for residual)
    to default the residual propagation param if it is not specified
    """
    extended = []
    for spec in convolutions:
        if len(spec) == 3:
            extended.append(spec)
        elif len(spec) == 2:
            extended.append(spec + (1,))# 默认由（192,5）变为（192,5,1）
        else:
            raise Exception('invalid number of parameters in convolution spec ' + str(spec) + '. expected 2 or 3')
    return tuple(extended)


def embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    m.weight.data.normal_(0, 0.1)
    return m


def linear(in_features, out_features, dropout=0.):
    """Weight-normalized Linear layer (input: N x T x C)"""
    m = nn.Linear(in_features, out_features)
    m.weight.data.normal_(mean=0, std=math.sqrt((1 - dropout) / in_features))
    m.bias.data.zero_()
    return m


def convtbc(in_channels, out_channels, kernel_size, dropout=0., **kwargs):
    """Weight-normalized Conv1d layer"""
    m = ConvTBC(in_channels, out_channels, kernel_size, **kwargs)
    std = math.sqrt((4 * (1.0 - dropout)) / (m.kernel_size[0] * in_channels))
    m.weight.data.normal_(mean=0, std=std)
    m.bias.data.zero_()
    return m

