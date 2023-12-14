import torch
import re


def get_statement_length(seq):
    s = re.sub('\\s*CaMeL\\s*', 'CaMeL', seq)
    s = re.sub('\\s*_\\s*', '_', s)
    s = re.sub('\\s*\\.\\s*', '.', s)
    s = s.replace('@@ ', '')
    return len(s.strip().split())


class GPTCoNuTDataset():
    def __init__(
            self, src, tgt, dictionary, max_source_positions=1024, identifier=None
    ):
        self.src = src
        self.tgt = tgt
        self.dictionary = dictionary
        self.max_source_position = max_source_positions
        self.identifier = identifier

    def __getitem__(self, index):

        def find_sublist(ctx, src):
            start = -1
            for i in range(0, len(ctx) - len(src) + 1):
                if ctx[i: i + len(src)] == src:
                    start = i
                    break
            return start

        src_item, tgt_item = self.src[index], self.tgt[index]
        ctx_index = 0
        for i in range(len(src_item)):
            if src_item[i] == self.dictionary.ctx():
                ctx_index = i
        ctx_item = src_item[ctx_index + 1:]
        src_item = src_item[: ctx_index]

        start = find_sublist(ctx_item, src_item)
        if start <= 0:
            ctx_item = [self.dictionary.eos()] + ctx_item
            start = 1
        assert start > 0
        prev_context = ctx_item[: start]
        behind_context = ctx_item[start + len(src_item): ]
        # print(ctx_item)
        # print(prev_context )
        # print(src_item)
        # print(behind_context)

        return {
            'id': index,
            'source': src_item,
            'source_statement_length': get_statement_length(self.dictionary.string(src_item)),
            'context': ctx_item,
            'target': tgt_item,
            'prev_context': prev_context,
            'behind_context':behind_context,
            'identifier': self.identifier[index] if self.identifier is not None else None
        }

    def __len__(self):
        return len(self.src)

    def merge(self, sources):# 合并 
        max_length = max([len(s) for s in sources])
        merged = []
        for s in sources:
            s_ = s + [self.dictionary.pad()] * (max_length - len(s))# 使所有的长度相同
            s_ = s_[: self.max_source_position]# 再截到模型最大输入长度
            if len(s_) == self.max_source_position and s_[-1] != self.dictionary.pad():
                s_[-1] = self.dictionary.eos()
            merged.append(s_)
        return torch.LongTensor(merged)

    def collater(self, samples):
        id = torch.LongTensor([s['id'] for s in samples])
        
        prev_context = self.merge([s['prev_context'] for s in samples])
        behind_context = self.merge([s['behind_context'] for s in samples])
        src_tokens = self.merge([s['source'] for s in samples])# merge里有处理的最大长度
        src_with_pre_context = self.merge(
            [s['prev_context'] + s['source'] + [0] * (src_tokens.size(1) - len(s['source']))# 0是指填充的部分
             for s in samples]
        )

        max_length = max([len(s['prev_context']) for s in samples])
        a = len(samples)

        # 将程序段分为目标代码前和目标代码行两部分，也就是上面prev_context和source

        src_tokens = self.merge(
            [[0]*len(s['prev_context']) + [1] * src_tokens.size(1)# 标记那一部分是原内容，哪些是填充的
             for s in samples]
        )
        src_statement_length = torch.LongTensor([[s['source_statement_length']] for s in samples])

        ctx_tokens = self.merge([s['context'] for s in samples])
        
        tgt_tokens = self.merge([s['target'] for s in samples])
        tgt_with_prev_context = self.merge(
            [s['prev_context'] + s['target'] + [0] * (tgt_tokens.size(1) - len(s['target']))
             for s in samples]
        )
        tgt_index = self.merge(
            [[0]*(len(s['prev_context']) - 1) + [1] + [1] * tgt_tokens.size(1)# 这里为什么要-1？
             for s in samples]
        )
        identifiers = [s['identifier'] for s in samples]

        return {
            'id': id,
            'net_input': {
                'src_tokens': src_tokens,
                'src_with_prev_context': src_with_pre_context,
                'ctx_tokens': ctx_tokens,
            },
            'src_statement_length': src_statement_length,
            'prev_context': prev_context,
            'behind_context': behind_context,
            'target': tgt_tokens,
            'target_index': tgt_index,
            'target_with_prev_context': tgt_with_prev_context,
            'identifier': identifiers if None not in identifiers else None
        }