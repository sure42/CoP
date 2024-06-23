import re
import os
import math
import torch
import torch.nn as nn
import sys

# BEAM_SEARCH_DIR = os.path.abspath(__file__)[: os.path.abspath(__file__).rindex('/') + 1] 
BEAM_SEARCH_DIR = os.path.abspath(__file__)[: os.path.abspath(__file__).rindex('\\') + 1]
GPT_CONUT_TRAINER_DIR = os.path.abspath(__file__)[: os.path.abspath(__file__).rindex('\\') + 1]
sys.path.append(GPT_CONUT_TRAINER_DIR + '../models/')
sys.path.append(GPT_CONUT_TRAINER_DIR + '../dataloader/')
# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
device = torch.device('cpu')
# sys.path.append(BEAM_SEARCH_DIR + '../models/')
sys.path.append(BEAM_SEARCH_DIR + '../models/')
from gpt_conut import GPTCoNuTModel
from gpt_fconv import GPTFConvModel
# from gpt_detector import GPTDetector
from dictionary import Dictionary


def get_statement_length(seq):
    s = re.sub('\\s*CaMeL\\s*', 'CaMeL', seq)
    s = re.sub('\\s*_\\s*', '_', s)
    s = re.sub('\\s*\\.\\s*', '.', s)
    s = s.replace('@@ ', '')
    return len(s.strip().split())


def add_token_to_string(string, symbol):
    if symbol in ['CaMeL', '_', '.']:
        return string + symbol
    elif string[-5:] == 'CaMeL' or string[-1:] in ['_', '.'] or string[-2:] == '@@':
        return string + symbol
    else:
        return string + ' ' + symbol


class GPTCoNuTModelCuda(nn.Module):
    def __init__(self, model, beam_size):
        super(GPTCoNuTModelCuda, self).__init__()
        self.model = model
        self.beam_size = beam_size
        self.split_size = beam_size
        self.split_size_list = [self.split_size]
        self.embed_model = model.embed_model

    def forward(self):
        pass

    def encoder_out_to_cuda(self, encoder_out):
        return {
            'src_tokens': encoder_out['src_tokens'].cuda(),
            'encoder_out': (
                encoder_out['encoder_out'][0].cuda(),
                encoder_out['encoder_out'][1].cuda(),
            ),
            'encoder_padding_mask': encoder_out['encoder_padding_mask'].cuda(),
        }

    def encoder_out_to_cpu(self, encoder_out):
        return {
            'src_tokens': encoder_out['src_tokens'].to('cpu'),
            'encoder_out': (
                encoder_out['encoder_out'][0].to('cpu'),
                encoder_out['encoder_out'][1].to('cpu'),
            ),
            'encoder_padding_mask': encoder_out['encoder_padding_mask'].to('cpu'),
        }

    def encode(self, src_tokens, src_with_prev_context, ctx_tokens):
        encoder_out = self.model.encoder(
            src_tokens,
            src_with_prev_context,
            ctx_tokens,
            self.model.embed_model,
        )
        return encoder_out

    def decode(self, prev_tokens_index, encoder_out, prev_tokens):
        step = int(torch.sum(prev_tokens_index[0]))
        ctx_len = prev_tokens.size(1)# prev+生成的
        # if step * ctx_len <= 3000:
        #     self.split_size = min(self.beam_size, 200)
        # elif step * ctx_len <= 5000:
        #     self.split_size = min(self.beam_size, 100)
        # elif step * ctx_len <= 10000:
        #     self.split_size = min(self.beam_size, 50)
        # else:
        #     self.split_size = min(self.beam_size, 20)
        self.split_size = 10
        split_num = self.beam_size // self.split_size
        self.split_size_list = [self.split_size] * split_num # split_size_list指分组
        if self.beam_size % self.split_size != 0:# 有一些剩余的，再分一组
            self.split_size_list += [self.beam_size % self.split_size]
        if prev_tokens_index.size(0) == 1:# 只有一个数据 这个应该是step = 0时的情况
            # return self.model.decoder(
            #     prev_tokens_index.cuda(),
            #     self.encoder_out_to_cuda(encoder_out),
            #     prev_tokens.cuda(),
            #     self.model.embed_model,
            # )[0].to('cpu')# 返回的是参数值x
            return self.model.decoder(
                    prev_tokens_index,
                    encoder_out,
                    prev_tokens,
                    self.model.embed_model,
                )[0].to('cpu')# 返回的是参数值x
        else:
            assert prev_tokens_index.size(0) == sum(self.split_size_list)
            decoder_out = []
            # split_encoder_out = {
            #     'src_tokens': encoder_out['src_tokens'][:self.split_size, ...].cuda(),
            #     'encoder_out': (
            #         encoder_out['encoder_out'][0][:self.split_size, ...].cuda(),
            #         encoder_out['encoder_out'][1][:self.split_size, ...].cuda(),
            #     ),
            #     'encoder_padding_mask': encoder_out['encoder_padding_mask'][:self.split_size, ...].cuda(),
            # }
            split_encoder_out = {
                'src_tokens': encoder_out['src_tokens'][:self.split_size, ...],
                'encoder_out': (
                    encoder_out['encoder_out'][0][:self.split_size, ...],
                    encoder_out['encoder_out'][1][:self.split_size, ...],
                ),
                'encoder_padding_mask': encoder_out['encoder_padding_mask'][:self.split_size, ...],
            }
            for i in range(len(self.split_size_list)):
                if i == len(self.split_size_list) - 1:# 最后一组，可能是多余的数据组成的
                    split_encoder_out = {
                        'src_tokens': split_encoder_out['src_tokens'][:self.split_size_list[-1], ...],
                        'encoder_out': (
                            split_encoder_out['encoder_out'][0][:self.split_size_list[-1], ...],
                            split_encoder_out['encoder_out'][1][:self.split_size_list[-1], ...],
                        ),
                        'encoder_padding_mask': split_encoder_out['encoder_padding_mask'][:self.split_size_list[-1], ...],
                    }
                # logits = self.model.decoder(# 取出一组数据，解码
                #     prev_tokens_index[sum(self.split_size_list[: i]) : sum(self.split_size_list[: i + 1]), ...].cuda(),
                #     split_encoder_out,
                #     prev_tokens[sum(self.split_size_list[: i]) : sum(self.split_size_list[: i + 1]), ...].cuda(),
                #     self.model.embed_model,
                # )[0] # split_size x step(目前生成补丁的长度) x　50061
                logits = self.model.decoder(# 取出一组数据，解码
                    prev_tokens_index[sum(self.split_size_list[: i]) : sum(self.split_size_list[: i + 1]), ...],
                    split_encoder_out,
                    prev_tokens[sum(self.split_size_list[: i]) : sum(self.split_size_list[: i + 1]), ...],
                    self.model.embed_model,
                )[0] # split_size x step(目前生成补丁的长度) x　50061
                logits = logits[:, -1, :]  # beam, L, V -> beam, V 取出最新的一个
                decoder_out.append(logits.to('cpu'))
        logits = torch.cat(decoder_out, dim=0)
        # logits = logits.to('cpu')
        return logits


# class GPTDetectorCuda():
#     def __init__(self, beam_size, embed_model):
#         super(GPTDetectorCuda, self).__init__()
        
#         vocab_file = GPT_CONUT_TRAINER_DIR + '..\..\data\\vocabulary\\vocabulary.txt'
#         dictionary = Dictionary(vocab_file, min_cnt=0)

#         gpt_file = GPT_CONUT_TRAINER_DIR + '..\..\data\models\code_gpt.pt'
#         model_id = 1
#         model_file = '.\data\models\gpt_detector_' + str(model_id) + '.pt'
#         loaded = torch.load(
#             model_file, map_location='cpu'
#         )

#         model = GPTDetector(
#             dictionary, embed_dim=384,embed_model = embed_model,
#         ).to('cpu')
#         model.embed_model.to('cpu')
#         model.load_state_dict(loaded['model'])
#         optimizer = torch.optim.Adam(model.parameters(), lr=6.25e-5)
#         self.model = model

#         self.beam_size = beam_size
#         self.split_size = beam_size
#         self.split_size_list = [self.split_size]

#     def valid(self, tokens, attention_mask, prev_tokens_index):
#         #  prev_tokens_index, encoder_out, prev_tokens
#         step = int(torch.sum(prev_tokens_index[0]))
#         ctx_len = tokens.size(1)
#         if step * ctx_len <= 3000:
#             self.split_size = min(self.beam_size, 200)
#         elif step * ctx_len <= 5000:
#             self.split_size = min(self.beam_size, 100)
#         elif step * ctx_len <= 10000:
#             self.split_size = min(self.beam_size, 50)
#         else:
#             self.split_size = min(self.beam_size, 20)
#         split_num = self.beam_size // self.split_size
#         self.split_size_list = [self.split_size] * split_num
#         if self.beam_size % self.split_size != 0:
#             self.split_size_list += [self.beam_size % self.split_size]
#         if prev_tokens_index.size(0) == 1:
#             # loss, logits, _ = self.model(
#             #     tokens.cuda(),
#             #     attention_mask.cuda(),
#             #     prev_tokens_index.cuda(),
#             # ).to('cpu')
#             loss, logits, _ = self.model(
#                 tokens,
#                 attention_mask,
#                 prev_tokens_index,
#             ).to('cpu')
#             return logits
#         else:
#             assert prev_tokens_index.size(0) == sum(self.split_size_list)
#             decoder_out = []
#             for i in range(len(self.split_size_list)):
#                 # loss, logits, _ = self.model(
#                 #         tokens[sum(self.split_size_list[: i]) : sum(self.split_size_list[: i + 1]), ...].cuda(),
#                 #         prev_tokens_index[sum(self.split_size_list[: i]) : sum(self.split_size_list[: i + 1]), ...].cuda(),
#                 #         attention_mask[sum(self.split_size_list[: i]) : sum(self.split_size_list[: i + 1]), ...].cuda(),                        
#                 #     )
#                 loss, logits, _ = self.model(
#                         tokens[sum(self.split_size_list[: i]) : sum(self.split_size_list[: i + 1]), ...],
#                         prev_tokens_index[sum(self.split_size_list[: i]) : sum(self.split_size_list[: i + 1]), ...],
#                         attention_mask[sum(self.split_size_list[: i]) : sum(self.split_size_list[: i + 1]), ...],                        
#                     )
#                 decoder_out.append(logits.to('cpu'))
#         logits = torch.cat(decoder_out, dim=0)
#         # logits = logits.to('cpu')
#         return logits



class BeamSearch():
    def __init__(self, model, dictionary, beam_size=10):
        self.dictionary = dictionary
        if isinstance(model, GPTCoNuTModel):
            self.model = GPTCoNuTModelCuda(model, beam_size)
        # self.detector = GPTDetectorCuda(beam_size, model.embed_model)
        self.beam_size = beam_size
        self.max_step = 128

    @staticmethod
    def get_prefix(token, dictionary):#把生成的token转换成单词
        prefixs, texts = [], []
        prefix, text = '', ''
        stop = False
        for i in range(len(token) - 1, -1, -1):# 从后向前
            cur = dictionary[token[i]]
            if cur not in ['CaMeL', '_', '0', '1', '$NUMBER$']:
                if not stop:
                    stop = True
                    prefix = cur + prefix
                    text = cur + text
                    prefixs.append(prefix)
                    if text[-2:] != '@@':
                        texts.append(text.replace('@@', ''))
                    else:
                        texts.append(text)
                else:
                    if cur[-2:] == '@@':
                        prefix = cur + prefix
                        text = cur + text
                        prefixs.append(prefix)
                        if text[-2:] != '@@':
                            texts.append(text.replace('@@', ''))
                        else:
                            texts.append(text)
                    else:# 生成结束
                        return prefixs, texts
            else:
                stop = False
                prefix = cur + prefix
                if cur != 'CaMeL':
                    text = cur + text
                prefixs.append(prefix)
                if text[-2:] != '@@':
                    texts.append(text.replace('@@', ''))
                else:
                    texts.append(text)
        prefixs.append(prefix)
        if text[-2:] != '@@':
            texts.append(text.replace('@@', ''))
        else:
            texts.append(text)
        return prefixs, texts

    def generate_gpt_conut_with_detect(self, sample, model_id):
        self.model.eval()
        hypothesis = []
        src_tokens = sample['net_input']['src_tokens']
        ctx_tokens = sample['net_input']['ctx_tokens']
        src_with_prev_context = sample['net_input']['src_with_prev_context']
        identifiers = sample['identifier']

        # model_file = '.\data\models\gpt_detector_' + str(model_id) + '.pt'
        # GPT_CONUT_TRAINER_DIR = os.path.abspath(__file__)[: os.path.abspath(__file__).rindex('\\') + 1]
        # gpt_file = GPT_CONUT_TRAINER_DIR + '..\..\data\models\code_gpt.pt'
        # vocab_file = GPT_CONUT_TRAINER_DIR + '..\..\data\\vocabulary\\vocabulary.txt'
        # dictionary = Dictionary(vocab_file, min_cnt=0)
        # loaded = torch.load(
        #     model_file, map_location='cpu'
        # )
        # # detector_model = GPTDetector(
        # #     dictionary, embed_dim=384, embed_model_file = gpt_file,
        # # ).cuda()
        # detector_model = GPTDetector(
        #     dictionary, embed_dim=384, embed_model_file = gpt_file,
        # )
        # detector_model.load_state_dict(loaded['model'])

        self.max_step = max(100, int(torch.sum(src_tokens[0])))# 最小最大长度  如果原行长度小于100，则生成最长为100，否则，最长不会超过原长？
        # int(torch.sum(src_tokens[0])) 原Bug行的长度--src_tokens标记bug行前和bug行

        prev_tokens_index = sample['target_index']
        prev_tokens_with_context = sample['target_with_prev_context']

        mask = prev_tokens_index.eq(0)# 记录prev_context  实际上也可以拿src来计算
        prev_tokens_index = prev_tokens_index[mask].unsqueeze(0)# prev_tokens_index[mask]只保留true对应的内容，也就是截出prev_content
        prev_tokens_index = torch.cat([prev_tokens_index, torch.ones(1, 1).long()], dim=-1)# torch.long():向下取整 tensor([[1.]])->tensor([[1]]) 接一个1应该是连接符
        prev_tokens_with_context = prev_tokens_with_context[:, : prev_tokens_index.size(1)]# 截取出前文内容
        prev_len = prev_tokens_index.size(1)

        # 这一部分用于初始化
        self.max_step += prev_tokens_index.size(1)# 代码段的总长度
        bsz = src_tokens.size(0)  # bsz = 1  batch
        tokens = torch.zeros(bsz * self.beam_size, self.max_step - prev_len).long()# [bsz * self.beam_size, self.max_step - prev_len]    保存生成的内容 
        scores = torch.zeros(bsz * self.beam_size, self.max_step - prev_len)
        final_scores = torch.zeros(bsz * self.beam_size)# torch.zeros(*size, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) 返回一个形状为为size,类型为torch.dtype，里面的每一个值都是0的tensor
        tokens_string = ['' for _ in range(tokens.size(0))]# token转化为字符串
        prev_tokens_index = prev_tokens_index.repeat(bsz * self.beam_size, 1)# 对张量进行重复扩充，只有两个时：(行的倍数，列的倍数)，此时通道数倍数为1。当参数有三个时：(通道数的倍数，行的倍数，列的倍数)
        prev_tokens_with_context = prev_tokens_with_context.repeat(bsz * self.beam_size, 1)
        tokens = torch.cat([prev_tokens_with_context, tokens], dim=1)# 把Bug行之前的内容接到将生成的部分

        if identifiers is not None:
            for k in identifiers[0]['tokens']:# 这里的作用是什么呢？？？？？
                identifiers[0]['tokens'][k] += [self.dictionary.pad(),
                                                self.dictionary.unk(),
                                                self.dictionary.eos()]#把每个字符的编码接上三个标识符
                identifiers[0]['tokens'][k] = torch.LongTensor(identifiers[0]['tokens'][k])

        length_penalty = {-50: -6.5446, -49: -6.7353, -48: -6.5446, -47: -6.4651, -46: -6.5255,
                          -45: -6.5624, -44: -6.2495, -43: -6.4651, -42: -6.1886, -41: -6.4025,
                          -40: -6.2033, -39: -6.2622, -38: -6.1274, -37: -6.2537, -36: -6.0743,
                          -35: -6.0758, -34: -6.031, -33: -5.927, -32: -5.866, -31: -5.9582,
                          -30: -5.64, -29: -5.7394, -28: -5.6764, -27: -5.5039, -26: -5.3938,
                          -25: -5.4417, -24: -5.3806, -23: -5.299, -22: -5.1481, -21: -5.1686,
                          -20: -5.0302, -19: -4.9543, -18: -4.8488, -17: -4.6396, -16: -4.6334,
                          -15: -4.5676, -14: -4.4454, -13: -4.2981, -12: -4.1142, -11: -4.048,
                          -10: -3.7681, -9: -3.5306, -8: -3.834, -7: -3.1647, -6: -3.011,
                          -5: -2.9796, -4: 0.0, -3: 0.0, -2: 0.0, -1: 0.0, 0: 0.0}# 长度惩罚

        encoder_out = self.model.encode(# 这里encode和下面的decode是beamsearch里的方法
            src_tokens,
            src_with_prev_context,
            ctx_tokens,
        )
        for step in range(0, self.max_step - prev_len):# 进行生成的部分
            if step == 0:# 第一次生成 并给出初始的得分情况
                logits = self.model.decode(# 生成的长度取决于tokens的内容
                    prev_tokens_index[:bsz, :],# 这里是选第一个
                    encoder_out,
                    tokens[:bsz, :step + prev_len],# 这里是pre+已经生成的
                )
                logits = logits[:, -1, :]  # 1, V  这里是每个字符出现的概率
                logits[:, self.dictionary.pad()] = -math.inf
                if identifiers is not None:
                    logits += 100
                    #
                    # 这里为啥要+100？
                    #
                lprobs, indices = logits.topk(k=self.beam_size, dim=1)# 找前1000个
                tokens[:, prev_len + step: prev_len + step + 1] = indices.transpose(0, 1)# 把这1000个生成的单词的编号存到tokens里对应的位置（末尾）上
                prev_tokens_index = torch.cat([prev_tokens_index, torch.ones(self.beam_size, 1).long()], dim=-1)# index里也加上对应的musk
                scores[:, step: step + 1] = lprobs.transpose(0, 1)# 记录第一个生成的得分
                final_scores = scores[:, step]

                for i, string in enumerate(tokens_string):
                    symbol = ''
                    if int(tokens[i, prev_len + step]) != self.dictionary.eos() and \
                            int(tokens[i, prev_len + step]) != self.dictionary.pad():# 对于终止和填充符号，不转化为字符，实际上也没有对应的，当作空（''）来处理
                        symbol = self.dictionary[int(tokens[i, prev_len + step])]
                    tokens_string[i] = add_token_to_string(string, symbol)# 连接
                continue
            if step == 1:
                # 将最开始生成的第一个结果扩展到一组
                # bsz * beam, L
                # 扩展到一组
                split_size = int(self.model.split_size)
                encoder_out['src_tokens'] = encoder_out['src_tokens']. \
                    repeat(split_size, 1)
                encoder_out['encoder_out'] = (
                    encoder_out['encoder_out'][0].repeat(split_size, 1, 1),
                    encoder_out['encoder_out'][1].repeat(split_size, 1, 1),
                )
                encoder_out['encoder_padding_mask'] = encoder_out['encoder_padding_mask']. \
                    repeat(split_size, 1)

            logits = self.model.decode(
                prev_tokens_index,
                encoder_out,
                tokens[:, :prev_len + step],
            )
            logits[:, self.dictionary.pad()] = -math.inf# 使其几乎不可能生成

            if identifiers is not None:
                for i in range(tokens.size(0)):# 遍历这一步生成的所有的字符
                    src_length = int(sample['src_statement_length'][0])# src_statement_length是Bug行以空格切分后的长度
                    cur_length = len(tokens_string[i].strip().split())# 第i个生成的长度
                    if cur_length > src_length:# 判断长度差
                        gap = max(-50, src_length - cur_length)
                    else:
                        gap = max(-50, cur_length - src_length)
                    if cur_length < src_length:# 施加惩罚 更容易停止
                        logits[i, self.dictionary.eos()] += length_penalty[gap]# eos的概率降低
                    else:
                        logits[i, self.dictionary.eos()] -= length_penalty[gap]# eos的概率升高

                    if not identifiers[0]['text']:
                        continue

                    prefixs, texts = BeamSearch.get_prefix(# prefixs指前一步生成的结果，texts目前所有的内容
                        tokens[i][prev_len: prev_len + step],# 选择当前生成的字符
                        self.dictionary
                    )
                    prefixs.reverse()
                    texts.reverse()

                    tmp = None
                    for prefix, text in zip(prefixs, texts):# 这里的作用有点模糊
                        if prefix in identifiers[0]['tokens']:
                            if prefix != '' and prefix[-5:] != 'CaMeL' and prefix[-1] != '_' and \
                                    text in identifiers[0]['text']:
                                tmp = set(identifiers[0]['tokens'][prefix].tolist()) | \
                                      set(identifiers[0]['tokens'][''].tolist())
                                tmp = torch.LongTensor(list(tmp))
                            else:
                                tmp = identifiers[0]['tokens'][prefix]# 找到该预测词的表示
                            break
                        if text in identifiers[0]['text']:
                            break
                    if tmp is None:
                        prefix = ''
                        tmp = identifiers[0]['tokens'][prefix]
                    logits[i, tmp] += 100

            lprobs, indices = logits.topk(k=self.beam_size, dim=1)  # beam, beam 第一维指前面的，第二维是该步生成的结果
            lprobs = lprobs.t().contiguous().view(-1)  # beam x beam 这一轮所有可能生成的结果    contiguous方法改变了多维数组在内存中的存储顺序，以便配合view方法使用
            indices = indices.t().contiguous().view(-1)

            bsz = tokens.size(0)# 这里应该是用来记录batch
            tmp_musk = ~tokens.eq(0)# prev+已生成的  
            tmp_musk = tmp_musk.view(-1)
            tmp_tokens = tokens.view(-1)       
            tmp_tokens = tmp_tokens[tmp_musk]
            tmp_tokens = tmp_tokens.view(bsz, -1)

            attention_mask = torch.ones(tmp_tokens.size()).masked_fill_(
                    tmp_tokens == 0, 0).float()
            # detector_scores = self.detector.valid(tmp_tokens, attention_mask, prev_tokens_index)


            cand_final_scores = lprobs + final_scores.repeat(self.beam_size) # only N-gram
            # cand_final_scores = detector_scores[:, 1].repeat(self.beam_size) # only eva
            # cand_final_scores = lprobs + final_scores.repeat(self.beam_size) + detector_scores[:, 1].repeat(self.beam_size)
            # 论文里提出的新的search方法，向下多考虑一层，lprobs是本层的内容，final_scores是上一步
            # 预测结果 0是有 1是没有 没有的概率越高越好
            cand_final_scores, sort_order = cand_final_scores.sort(descending=True)# sort_order指排序结果 降序
            lprobs = lprobs.index_select(0, sort_order)# index_select(dim,index) dim：表示从第几维挑选数据，类型为int值 index：表示从第一个参数维度中的哪个位置挑选数据，类型为torch.Tensor类的实例
            indices = indices.index_select(0, sort_order)

            # choose finished beam
            eos_mask = indices[: self.beam_size].eq(self.dictionary.eos())
            eos_lprobs = lprobs[: self.beam_size].masked_select(mask=eos_mask)  # N_eos 返回一个根据布尔掩码索引输入张量的 1D 张量 其中布尔掩码和输入张量就是 torch.masked_select(input, mask, out = None) 函数的两个关键参数
            eos_cand_final_scores = cand_final_scores[: self.beam_size].masked_select(mask=eos_mask)  # N_eos
            eos_sort_order = sort_order[: self.beam_size].masked_select(mask=eos_mask)  # N_eos
            if eos_cand_final_scores.size(0) > 0:
                eos_beam_ids = eos_sort_order % self.beam_size  # N_eos
                eos_beam_ids = eos_beam_ids.long()
                eos_beam = tokens[eos_beam_ids, prev_len:]  # N_eos, L
                eos_beam[:, step] = self.dictionary.eos()
                eos_beam_scores = scores[eos_beam_ids, :]
                eos_beam_scores[:, step] = eos_lprobs
                for i in range(eos_beam.size(0)):
                    hypothesis.append(
                        {
                            'hypo': eos_beam[i, : step + 1],
                            'score': eos_beam_scores[i, 1: step + 1],
                            'final_score': float(eos_cand_final_scores[i]) / (1 + step),
                        })
                if len(hypothesis) >= self.beam_size:# 只选择beam_size条
                    hypothesis = hypothesis[: self.beam_size]
                    break

            # choose next beam
            cand_mask = ~indices.eq(self.dictionary.eos())# ~是取反 找出非eos的内容
            cand_final_scores = cand_final_scores.masked_select(mask=cand_mask)[: self.beam_size]# cand_final_scores这些都是排序过的，先去除掉生成eos的，再选出前beam_size条后
            sort_order = sort_order.masked_select(mask=cand_mask)[: self.beam_size]
            lprobs = lprobs.masked_select(mask=cand_mask)[: self.beam_size]
            indices = indices.masked_select(mask=cand_mask)[: self.beam_size]
            cand_beam_ids = sort_order % self.beam_size# 逻辑是什么？
            cand_beam_ids = cand_beam_ids.long()

            new_tokens_string = []
            for i in range(cand_beam_ids.size(0)):
                symbol = ''
                if int(indices[i]) not in [self.dictionary.eos(), self.dictionary.pad()]:
                    symbol = self.dictionary[int(indices[i])]
                new_tokens_string.append(
                    add_token_to_string(tokens_string[int(cand_beam_ids[i])], symbol)
                )
            tokens_string = new_tokens_string

            tokens = tokens[cand_beam_ids, :]
            tokens[:, prev_len + step] = indices# 选定这一步生成的结果了

            scores = scores[cand_beam_ids, :]
            scores[:, step] = lprobs
            final_scores = cand_final_scores# 用于下一步
            prev_tokens_index = torch.cat([prev_tokens_index,# 延长一位
                                           torch.ones(self.beam_size, 1).long()], dim=-1)

        if len(hypothesis) < self.beam_size:# 没有生成足够的 超过了最大长度或生成了eos
            current_num = len(hypothesis)
            for i in range(self.beam_size - current_num):
                tokens[i, -1] = self.dictionary.eos()
                hypothesis.append({
                    'hypo': tokens[i],
                    'score': scores[i, 1:],
                    'final_score': float(final_scores[i]) / (self.max_step - prev_len),
                })

        hypothesis.sort(key=lambda e: e['final_score'], reverse=True)
        return hypothesis
