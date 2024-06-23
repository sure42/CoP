import codecs # 标准 Python 编解码器
import torch
from gpt_conut_dataset import GPTCoNuTDataset


def find_sublist(ctx, src):
    start = -1
    for i in range(0, len(ctx) - len(src) + 1):
        if ctx[i: i + len(src)] == src:
            start = i
            break
    return start

def read_data(datafile, dictionary):
    fp = codecs.open(datafile, 'r', 'utf-8')
    src_list = []
    tgt_list = []
    with open(datafile, 'r', encoding='utf-8') as fp:
        for line in fp.readlines():
            if line.strip() == '':
                continue
            
            src, tgt = line.split('\t')
            src = src.strip().split() + [dictionary.eos_word]
            tgt = tgt.strip().split() + [dictionary.eos_word]

            
            # 取漏洞行和函数
            for i in range(len(src)):
                if src_item[i] == dictionary.ctx_word:
                    ctx_index = i
            ctx_item = src_item[ctx_index + 1:]
            src_item = src_item[: ctx_index]

            # 按漏洞行将函数进行划分 ctx = prev + src + rear
            start = find_sublist(ctx_item, src_item)
            if start <= 0:
                ctx_item = [dictionary.eos_word] + ctx_item
                start = 1
            assert start > 0
            prev_context = ctx_item[: start]
            rear_context = ctx_item[start + len(src_item):]


            src_list.append(src)
            tgt_list.append(tgt)
            
    return src_list, tgt_list

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, datafile, dictionary, identifier_loader=None):
        self.datafile = datafile
        self.dictionary = dictionary
        self.total_size = 0
        self.get_total_size()

        self.src = []
        self.tgt = []

        self.identifier_loader = identifier_loader
        self.dataset = None

    def get_total_size(self): # 统计数据条数
        fp = codecs.open(self.datafile, 'r', 'utf-8')
        self.total_size = len(fp.readlines())
        fp.close()

    def reinitialize(self):
        self.src = []
        self.tgt = []
        self.dataset = None

    def load_data(self, start, end):
        self.reinitialize()
        fp = codecs.open(self.datafile, 'r', 'utf-8')
        cnt = -1
        while True:
            line = fp.readline()
            if not line:
                break
            if line.strip() == '':
                continue
            cnt += 1
            if cnt < start:
                continue
            if cnt >= end:
                break
            
            src, tgt = line.split('\t')
            src = src.strip().split()
            tgt = tgt.strip().split()

            src_tokens = self.dictionary.index(src)# 切分好的字符表示对应的编号，如'int a=1' src里的token分别是int a = 1 四部分对应的编号
            tgt_tokens = self.dictionary.index(tgt)
            src_tokens = src_tokens + [self.dictionary.eos()]# 序列结束
            tgt_tokens = tgt_tokens + [self.dictionary.eos()]
            self.src.append(src_tokens)
            self.tgt.append(tgt_tokens)
        if self.identifier_loader is not None:# 直接加载部分数据集和标识符
            self.identifier_loader.load_data(start, end)
            self.dataset = GPTCoNuTDataset(self.src, self.tgt, self.dictionary,
                                           identifier=self.identifier_loader.identifier_list)
        else:
            self.dataset = GPTCoNuTDataset(self.src, self.tgt, self.dictionary)

    def __len__(self):
        return len(self.src)
    
    def __getitem__(self, index):

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
        behind_context = ctx_item[start + len(src_item):]

        item = {"id": index}
        item["src"] = src_item
        # item["src_len"] = get_statement_length(len(src_item))
        item["ctx"] = ctx_item
        item["tgt"] = tgt_item
        item["prev"] = prev_context
        item["rear"] = behind_context
        # item["identifier"] = self.identifier[index] if self.identifier is not None else None

        return item

    # def merge(self, sources):# 合并 
    #     max_length = max([len(s) for s in sources])
    #     merged = []
    #     for s in sources:
    #         s_ = s + [self.dictionary.pad()] * (max_length - len(s))# 使所有的长度相同
    #         s_ = s_[: self.max_source_position]# 再截到模型最大输入长度
    #         if len(s_) == self.max_source_position and s_[-1] != self.dictionary.pad():
    #             s_[-1] = self.dictionary.eos()
    #         merged.append(s_)
    #     return torch.LongTensor(merged)

    # def collater(self, samples):
    #     id = torch.LongTensor([s['id'] for s in samples])
        
    #     prev_context = self.merge([s['prev_context'] for s in samples])
    #     behind_context = self.merge([s['behind_context'] for s in samples])
    #     src_tokens = self.merge([s['source'] for s in samples])# merge里有处理的最大长度
    #     src_with_pre_context = self.merge(
    #         [s['prev_context'] + s['source'] + [0] * (src_tokens.size(1) - len(s['source']))# 0是指填充的部分
    #          for s in samples]
    #     )

    #     max_length = max([len(s['prev_context']) for s in samples])
    #     a = len(samples)

    #     # 将程序段分为目标代码前和目标代码行两部分，也就是上面prev_context和source

    #     src_tokens = self.merge(
    #         [[0]*len(s['prev_context']) + [1] * src_tokens.size(1)# 标记那一部分是原内容，哪些是填充的
    #          for s in samples]
    #     )
    #     src_statement_length = torch.LongTensor([[s['source_statement_length']] for s in samples])

    #     ctx_tokens = self.merge([s['context'] for s in samples])
        
    #     tgt_tokens = self.merge([s['target'] for s in samples])
    #     tgt_with_prev_context = self.merge(
    #         [s['prev_context'] + s['target'] + [0] * (tgt_tokens.size(1) - len(s['target']))
    #          for s in samples]
    #     )
    #     tgt_index = self.merge(
    #         [[0]*(len(s['prev_context']) - 1) + [1] + [1] * tgt_tokens.size(1)# 这里为什么要-1？
    #          for s in samples]
    #     )
    #     identifiers = [s['identifier'] for s in samples]

    #     return {
    #         'id': id,
    #         'net_input': {
    #             'src_tokens': src_tokens,
    #             'src_with_prev_context': src_with_pre_context,
    #             'ctx_tokens': ctx_tokens,
    #         },
    #         'src_statement_length': src_statement_length,
    #         'prev_context': prev_context,
    #         'behind_context': behind_context,
    #         'target': tgt_tokens,
    #         'target_index': tgt_index,
    #         'target_with_prev_context': tgt_with_prev_context,
    #         'identifier': identifiers if None not in identifiers else None
    #     }


