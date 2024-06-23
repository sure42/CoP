import codecs
import torch


class Dictionary():
    def __init__(self, vocab_file, min_cnt=0):
        self.vocab_file = vocab_file
        self.pad_word = '<PAD>'# 0
        self.unk_word = '<UNK>'# 1
        self.eos_word = '<EOS>'# 2 序列的结束
        self.ctx_word = '<CTX>'# 3--index
        self.dictionary = {}# {标识符，编号}
        self.symbols = []# 标识符
        self.counts = []# 标识符出现次数
        self.min_cnt = min_cnt

        self.pad_index = self.add_symbol(self.pad_word)
        self.eos_index = self.add_symbol(self.eos_word)
        self.unk_index = self.add_symbol(self.unk_word)
        self.ctx_index = self.add_symbol(self.ctx_word)
        self.read_dictionary()

    def add_symbol(self, symbol, n=1):
        if symbol in self.dictionary:
            return self.dictionary[symbol]
        idx = len(self.dictionary)
        self.dictionary[symbol] = idx
        self.symbols.append(symbol)
        self.counts.append(n)
        return idx

    def read_dictionary(self):
        fp = codecs.open(self.vocab_file, 'r', 'utf-8')
        for l in fp.readlines():
            l = l.strip()
            if len(l.split()) != 2:# 出错
                continue
            symbol, count = l.split()
            if int(count) < self.min_cnt:# 出现次数少，忽略
                continue
            self.add_symbol(symbol, int(count))

    def __getitem__(self, item):
        if type(item) != int:
            return self.symbols[int(item)] if int(item) < len(self.symbols) else self.unk
        return self.symbols[item] if item < len(self.symbols) else self.unk

    def __len__(self):
        return len(self.dictionary)

    def index(self, symbol):
        if type(symbol) == list:
            return [self.index(s) for s in symbol]
        if symbol in self.dictionary:
            return self.dictionary[symbol]
        return self.unk_index


    # 不必要的
    def string(self, tensor, bpe_symbol=None, show_pad=False): # 将Tensor转换为字符串
        if torch.is_tensor(tensor) and tensor.dim() == 2: # 是否为PyTorch张量且是二维的
            return '\n'.join(self.string(t) for t in tensor).split('\n')

        hide = [self.eos(), self.pad()] if not show_pad else [self.eos()]

        sent = ' '.join(self[i] for i in tensor if i not in hide) # 从索引中查找对应的词语添加到sent字符串中，词语之间以空格分隔 
        if bpe_symbol is not None: 
            sent = (sent + ' ').replace(bpe_symbol + ' ', '').rstrip() # 在生成的字符串末尾添加一个空格，然后移除所有"BPE符号 + 空格"的组合，并去除末尾多余的空格
        return sent

    def pad(self):
        return self.pad_index

    def eos(self):
        return self.eos_index

    def unk(self):
        return self.unk_index

    def ctx(self):
        return self.ctx_index


if __name__ == "__main__":
    voc = Dictionary('../../../data/vocabulary/vocabulary.txt')
    print(len(voc))
