import os
import sys
import json
import time
import codecs
import random
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import OpenAIGPTLMHeadModel, T5EncoderModel
from transformers import AutoTokenizer, OpenAIGPTModel

from torch.utils.tensorboard import SummaryWriter

print(os.path.abspath(__file__)+'\n')
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
GPT_CONUT_TRAINER_DIR = os.path.abspath(__file__)[: os.path.abspath(__file__).rindex('\\') + 1]
sys.path.append(GPT_CONUT_TRAINER_DIR + '../models/')
sys.path.append(GPT_CONUT_TRAINER_DIR + '../dataloader/')
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# GPT_CONUT_TRAINER_DIR = os.path.abspath(__file__)[: os.path.abspath(__file__).rindex('/') + 1]
# sys.path.append(GPT_CONUT_TRAINER_DIR + '../models/')
# sys.path.append(GPT_CONUT_TRAINER_DIR + '../dataloader/')

from gpt_detector import GPTDetector, MyDataset
from dictionary import Dictionary
from gpt_conut_data_loader import GPTCoNuTDataLoader

writer = SummaryWriter("logs/004")

def tensor_pad(src, tgt):
    """ 
    src: 格式是二维的tensor
    tgt: 格式是二维的tensor, dim = 0与src相同

    function:
        将src和tgt在dim = 1上扩展到相同长度
    
    return src, tgt
    """ 
    dim_a, len_a = src.size()
    dim_b, len_b =  tgt.size()
    assert dim_a == dim_b

    if len_a > len_b:
        tmp_tgt = F.pad(tgt, (0, len_a - len_b), mode='constant', value = 0)
        dim_b, len_b = tmp_tgt.size()
        tgt = tmp_tgt
    else:
        tmp_src = F.pad(src, (0, len_b - len_a), mode='constant', value = 0)
        dim_a, len_a =  tmp_src.size()
        src = tmp_src
    # torch.nn.functional.pad()会依照从后往前的顺序依照pad的值对input进行padding,pad中两个为一组，第一组x,y为对最后一维左侧填充x，右侧填充y

    assert dim_a == dim_b
    assert len_a == len_b

    return src, tgt

def trans_copy(dataloader, samples):# 输入是从load_data的输出
    # samples:
    #   type:list
    #   item:{  
    #       id,
    #       source, 改成target
    #       source_statement_length, 不用动
    #       context, 
    #       target,
    #       prev_context,
    #       identifier                  
    # }     
    tmp_samples = samples
    for s in tmp_samples:
        s['source'] == s['target']
    labels = [0]*len(samples) + [1]*len(tmp_samples)
    samples = samples + tmp_samples
    dataset = dataloader.dataset.collater(samples)
    tgt_lable, bug_lable = [],[]

    src_with_prev_context = dataset['net_input']['src_with_prev_context']
    src_index = dataset['net_input']['src_tokens']

    target_with_prev_context = dataset['target_with_prev_context']
    target_index = dataset['target_index']

    # 把src_with_prev_context和target_with_prev_context弄成一样长，用0填充
    src_with_prev_context, target_with_prev_context = tensor_pad(src_with_prev_context, target_with_prev_context)
    src_index, target_index = tensor_pad(src_index, target_index)

    tmp_musk_bug = torch.unsqueeze(torch.ones(src_with_prev_context[0].size()).masked_fill_(
                    src_with_prev_context[0] == 0, 0).float(), 0) 
    tmp_musk_tgt = torch.unsqueeze(torch.ones(target_with_prev_context[0].size()).masked_fill_(
                    target_with_prev_context[0] == 0, 0).float(), 0) 
    for i in range(dataset['id'].size(0)):
        bug_lable.append(0)
        tgt_lable.append(1)
        if i != 0: 
            bug_attention_mask = torch.unsqueeze(torch.ones(src_with_prev_context[i].size()).masked_fill_(
                            src_with_prev_context[i] == 0, 0).float(), 0) 
            tgt_attention_mask = torch.unsqueeze(torch.ones(target_with_prev_context[i].size()).masked_fill_(
                            target_with_prev_context[i] == 0, 0).float(), 0) 
            # tmp_musk_bug = torch.cat([tmp_musk_bug, torch.unsqueeze(bug_attention_mask, 0)], 0)
            # tmp_musk_tgt = torch.cat([tmp_musk_tgt, torch.unsqueeze(tgt_attention_mask, 0)], 0)
            tmp_musk_bug = torch.cat([tmp_musk_bug, bug_attention_mask], 0)
            tmp_musk_tgt = torch.cat([tmp_musk_tgt, tgt_attention_mask], 0)
    
    encodings={
        'input_ids': torch.cat([src_with_prev_context, target_with_prev_context], 0),
        'attention_mask': torch.cat([tmp_musk_bug, tmp_musk_tgt], 0),
        'src_tokens': torch.cat([src_index, target_index], 0),
    }
    labels = bug_lable + tgt_lable
    # encodings={
    #     'input_ids': src_with_prev_context,
    #     'attention_mask': tmp_musk_bug,
    #     'src_tokens': src_index,
    # }
    # labels = bug_lable
    return encodings, labels

def trans(dataloader, samples):# 输入是从load_data的输出

    # samples:
    #   type:list
    #   item:{  
    #       id,
    #       source, 改成target
    #       source_statement_length, 不用动
    #       context, 
    #       target,
    #       prev_context,
    #       identifier                  
    # }     
    labels = []
    for sample in samples:
        rand_num = random.randint(0, 100)
        if rand_num > 50:# 取修复后结果   用1记录
            sample['source'] == sample['target']
            labels.append(1)
        else:# 取有漏洞的代码  用0记录
            labels.append(0)



    # tgt_samples = samples
    # src_samples = samples
    # for s in tgt_samples:
    #     s['source'] == s['target']
    # samples = src_samples + tgt_samples
    dataset = dataloader.dataset.collater(samples)

    src_with_prev_context = dataset['net_input']['src_with_prev_context']
    src_index = dataset['net_input']['src_tokens']

    attention_mask = torch.ones(src_with_prev_context.size()).masked_fill_(
                    src_with_prev_context == 0, 0).float() # B,L batch_size和长度
    encodings={
        'input_ids': src_with_prev_context,
        'attention_mask': attention_mask,
        'src_tokens': src_index,
    }
   
    return encodings, labels

class detector_trainer():
    def __init__(self, train_loader, valid_loader, dictionary):
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.dictionary = dictionary
        self.batch_size = 2
        self.load_size = 1000   # load 1200 samples from training data every time

        self.model = None
        self.hyper_parameter = {}
        self.optimizer = None
        self.current_train_step = 0
        self.val_loss = {}

    def shuffle_dataset(self, listlen):
        indices = [i for i in range(listlen)]
        random.seed(42)
        random.shuffle(indices)
        return indices
    
    def train_step(self, batch):
        self.model.train()
        self.current_train_step += 1
        # self.optimizer.zero_grad()

        input_ids = batch['input_ids'].to(device)
        attention_mask=batch['attention_mask'].to(device)
        labels=batch['labels'].to(device)
        src_tokens=batch['src_tokens'].to(device)

        outputs = self.model(
            input_ids=input_ids, src_tokens=src_tokens, attention_mask=attention_mask, labels=labels
        )
        loss, logits = outputs[:2]

        
        # compare_loss = 1

        # loss = apr_loss + 0.3 * compare_loss
        loss.mean().backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 0.5, norm_type=2)# 是对所有的梯度乘以一个clip_coef，且clip_coef一定是小于1的；只解决梯度爆炸问题，不解决梯度消失问题
        self.optimizer.step()
        return loss.mean().item(), logits

    def valid_step(self, batch):
        self.model.eval()
        input_ids = batch['input_ids'].to(device)
        attention_mask=batch['attention_mask'].to(device)
        labels=batch['labels'].to(device)
        src_tokens=batch['src_tokens'].to(device)
        outputs = self.model(
            input_ids=input_ids, src_tokens=src_tokens, attention_mask=attention_mask, labels=labels
        )
        loss, logits = outputs[:2]

        return loss.mean().item(), logits
        # return loss.mean().item(), logits.mean().item()

    def validate_and_save(self, model_id, save_dir):
        with torch.no_grad():
            start_time = time.time()
            oom = 0 
            val_loss, val_fconv_loss, val_lm_loss = [], [], []
            self.valid_loader.load_data(0, self.valid_loader.total_size)
            indices = self.shuffle_dataset(len(self.valid_loader.dataset))# 打乱
            start, end = 0, 0
            cor = 0
            samples = []
            max_src, max_ctx, max_tgt = 0, 0, 0
            valid_labels = []
            while end < len(self.valid_loader.dataset):
                # 取出一条数据
                sample = self.valid_loader.dataset[indices[end]]

                if max_ctx + len(sample['target']) >= 1023 \
                        or max_tgt + len(sample['prev_context']) >= 1023 \
                        or max_ctx + len(sample['source']) >= 1023 \
                        or max_src + len(sample['prev_context']) >= 1023 \
                        or end - start == self.batch_size:# 一个batch
                                        
                    valid_encodings, valid_labels = trans(self.valid_loader, samples)
                    valid_loss = []
                    valid_set = MyDataset(valid_encodings, valid_labels)
                    valid_loader = DataLoader(valid_set, batch_size=len(samples), shuffle=True)
                    for batch in valid_loader:# 里面包含两个batch，因为扩展了正负类
                        # try:
                            loss, logits = self.valid_step(batch)
                            pred_probs = torch.softmax(logits.squeeze(1), dim=1)
                            pred_labels = torch.argmax(pred_probs, dim=1).tolist()

                            valid_loss.append(loss)
                            valid_labels.extend(pred_labels)

                            print(pred_probs)

                            for (pred, ori) in zip(pred_labels, batch['labels']):
                                print("pred, ori:", pred, ori)
                                if(pred == ori.item()):
                                    cor += 1                    
                        # except Exception as e:
                        #     print(e)
                        #     oom += 1
                    if end  % 20 == 0:
                        info = 'modle-{}-valid  load data:{}, valid data:{}/{}, loss:{}, time:{}s, oom:{}'.\
                            format(model_id, self.valid_loader.total_size, 
                                end, self.valid_loader.total_size,
                                round(float(np.mean(loss)), 6),
                                int(time.time() - start_time), oom
                                )
                        start_time = time.time()
                        print(info)
                    start = end
                    max_src, max_ctx, max_tgt = 0, 0, 0
                    samples = []
                    continue
                max_src = max(max_src, len(sample['source']))
                max_ctx = max(max_ctx, len(sample['prev_context']))
                max_tgt = max(max_tgt, len(sample['target']))
                end += 1# 下一轮
                samples.append(sample)
            if len(samples) > 0:# 剩余的数据
                valid_encodings, valid_labels = trans(self.valid_loader, samples)
                valid_loss = []
                valid_set = MyDataset(valid_encodings, valid_labels)
                valid_loader = DataLoader(valid_set, batch_size=len(samples), shuffle=True)
                for batch in valid_loader:# 里面包含两个batch，因为扩展了正负类
                    # try:
                        loss, logits = self.valid_step(batch)
                        pred_probs = torch.softmax(logits.squeeze(1), dim=1)
                        pred_labels = torch.argmax(pred_probs, dim=1).tolist()

                        valid_loss.append(loss)
                        valid_labels.extend(pred_labels)

                        for (ori, pred) in zip(pred_labels, batch['labels']):
                            # print(ori, pred.item())
                            if(ori == pred.item()):
                                cor += 1

                    # except Exception as e:
                    #     print(e)
                    #     oom += 1

                if end  % 20 == 0:
                    info = 'modle-{}-valid  load data:{}, valid data:{}/{}, loss:{}, time:{}s, oom:{}'.\
                        format(model_id, self.valid_loader.total_size, 
                            end, self.valid_loader.total_size,
                            round(float(np.mean(loss)), 6),
                            int(time.time() - start_time), oom
                            )
                    start_time = time.time()
                    print(info)
        checkpoint = {
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'current_step': self.current_train_step,
            }

        torch.save(checkpoint, save_dir + 'gpt_detector_' + str(model_id) + '.pt')
        print(cor/(len(self.valid_loader.dataset)*2))
        print(cor)
        print('*'*30)
        return val_loss

    def train(self, model_id, epochs, save_dir, gpt_file, dropout=0.1, lr=6.25e-6):
        start_time_all = time.time()        
        # 模型读取        
        gpt_model = T5EncoderModel.from_pretrained(gpt_file)
        self.model = GPTDetector(
            self.dictionary, embed_dim=384, dropout=dropout,embed_model = gpt_model,
        ).cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=6.25e-6)

        # for param in self.model.embed_norm.parameters():
        #     param.requires_grad = False

        # self.model = nn.DataParallel(self.model, device_ids=device_ids)# 当迭代次数或者epoch足够大的时候，我们通常会使用nn.DataParallel函数来用多个GPU来加速训练
        # device_ids是所有可使用的GPU
        da = 1
        for epoch in range(epochs):
            start_time = time.time()
            for i in range(0, self.train_loader.total_size, self.load_size):# 训练数据不是一次加载完毕的，每次加载load_size条
                oom = 0
                self.train_loader.load_data(i, i + self.load_size)
                indices = self.shuffle_dataset(len(self.train_loader.dataset))# 打乱
                start, end = 0, 0
                samples = []
                max_src, max_ctx, max_tgt = 0, 0, 0
                # indices = [215, 260, 428, 730]   215-prev_context=876  260-source=197 428-source=198 730-source=240
                while end < len(self.train_loader.dataset):
                    sample = self.train_loader.dataset[indices[end]]# 取出一条训练数据
                    if max_ctx + len(sample['target']) >= 1023 \
                            or max_tgt + len(sample['prev_context']) >= 1023 \
                            or max_ctx + len(sample['source']) >= 1023 \
                            or max_src + len(sample['prev_context']) >= 1023 \
                            or end - start == self.batch_size:# 一个batch
                        # max_ctx记录之前最长的prev_context长度，max_tgt--target，max_src--source
                        # 之所以记录最长的加上这一批target和source，是为了避免train_step/collater对src_with_prev_context做merge时超过长度
                        # 虽说根据本项目里的数据集出现这种情况的可能性很小，但仍有可能出现，所以加这一句来保证鲁棒性                         
                        train_encodings, train_labels = trans(self.train_loader, samples)
                        train_loss, train_apr_loss, train_lm_loss = [], [], []
                        train_set = MyDataset(train_encodings, train_labels)
                        train_loader = DataLoader(train_set, batch_size=len(samples), shuffle=True)
                        # 试一下dataloader可以不
                        # iterator
                        for batch in train_loader:# 里面包含两个batch，因为扩展了正负类
                            try:
                                loss, logits = self.train_step(batch)# 内部需要改
                                # pred_probs = torch.softmax(outputs[1].squeeze(1), dim=1)
                                # 找到
                                pred_probs = torch.softmax(logits.squeeze(1), dim=1)
                                writer.add_scalar("loss-5-10-0.1-6.25e-5", loss, global_step = da)
                                da += 1
                                # print(da)
                                writer.close()
                                train_loss.append(loss)
                            except Exception as e:
                                print(e)
                                oom += 1
                        if end  % 20 == 0:# 打印训练过程 10 1210 ... 
                            info = 'modle-{}  epoch:{}, load data:{}-{}, train data:{}/{}, loss:{}, time:{}s, oom:{}'.\
                                format(model_id, epoch + 1, i , i + self.load_size,
                                    end, self.load_size,
                                    round(float(np.mean(loss)), 6),
                                    int(time.time() - start_time), oom
                                    )
                            start_time = time.time()
                            print(info)
                        start = end
                        max_src, max_ctx, max_tgt = 0, 0, 0
                        samples = []
                        continue
                    max_src = max(max_src, len(sample['source']))
                    max_ctx = max(max_ctx, len(sample['prev_context']))
                    max_tgt = max(max_tgt, len(sample['target']))
                    end += 1# 下一轮
                    samples.append(sample)
                if len(samples) > 0:# 剩余的数据
                    train_encodings, train_labels = trans(self.train_loader, samples)
                    train_loss, train_apr_loss, train_lm_loss = [], [], []
                    train_set = MyDataset(train_encodings, train_labels)
                    train_loader = DataLoader(train_set, batch_size=len(samples), shuffle=True)

                    for batch in train_loader:# 里面包含两个batch，因为扩展了正负类
                        try:
                            loss, logits = self.train_step(batch)# 内部需要改
                            # pred_probs = torch.softmax(outputs[1].squeeze(1), dim=1)
                            # 找到
                            pred_probs = torch.softmax(logits.squeeze(1), dim=1)
                            train_loss.append(loss)
                        except Exception as e:
                            print(e)
                            oom += 1
                    if end  % 20 == 0:# 打印训练过程 10 1210 ... 
                        info = 'modle-{}  epoch:{}, load data:{}-{}, train data:{}/{}, loss:{}, time:{}s, oom:{}'.\
                            format(model_id, epoch + 1, i , i + self.load_size,
                                end, self.load_size,
                                round(float(np.mean(loss)), 6),
                                int(time.time() - start_time), oom
                                )
                        start_time = time.time()
                        print(info)
        end_time_all = time.time()
        print('model train time:',end_time_all - start_time_all)
        train_time = end_time_all - start_time_all
        time_file = save_dir + 'time.txt'
        with open(time_file, 'a+') as f:
            info = 'detect train : epoch:{}, all data:{}, time:{}s, oom:{}, dir:{}\n'.\
                format(epochs, len(self.train_loader.dataset),
                        int(train_time), oom, save_dir + 'gpt_conut_' + str(model_id) + '.pt'
                )
            f.write(info)

        start_time_all = time.time()
        self.validate_and_save(model_id, save_dir)
        end_time_all = time.time()
        valid_time = end_time_all - start_time_all
        print('detector test time:',end_time_all - start_time_all)
        with open(time_file, 'a+') as f:
            info = 'detect valid : epoch:{}, all data:{}, time:{}s, oom:{}, dir:{}\n'.\
                format(epochs, len(self.train_loader.dataset),
                        int(valid_time), oom, save_dir + 'gpt_conut_' + str(model_id) + '.pt'
                )
            f.write(info)    

def test():
    # device_ids = [0]
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"   
    vocab_file = GPT_CONUT_TRAINER_DIR + '..\..\data\\vocabulary\\vocabulary.txt'
    train_file = GPT_CONUT_TRAINER_DIR + '..\..\data\data\\training_bpe.txt'
    valid_file = GPT_CONUT_TRAINER_DIR + '..\..\data\data\\training_bpe.txt'
    # valid_file = GPT_CONUT_TRAINER_DIR + '..\..\data\data\\validation_bpe.txt'
    gpt_file = GPT_CONUT_TRAINER_DIR + '..\..\data\models\code_gpt.pt'
    
    dictionary = Dictionary(vocab_file, min_cnt=0)
    print('dictionary initialized, vocab size:{}'.format(len(dictionary)))

    train_loader = GPTCoNuTDataLoader(train_file, dictionary)
    valid_loader = GPTCoNuTDataLoader(valid_file, dictionary)
    print('data loader initialized, train size:{}, validate size:{}'.
          format(train_loader.total_size, valid_loader.total_size))

    # 模型结构读取
    trainer = detector_trainer(train_loader, valid_loader, dictionary)
    
    model_id = 1
    
    model_file = '.\data\models\gpt_detector_' + str(model_id) + '.pt'
    loaded = torch.load(
        model_file, map_location='cpu'
    )

    trainer.model = GPTDetector(
        trainer.dictionary, embed_dim=384,dropout=0.5,embed_model = gpt_file,
    ).cuda()
    trainer.model.load_state_dict(loaded['model'])
    trainer.optimizer = torch.optim.Adam(trainer.model.parameters(), lr=6.25e-5)

    trainer.validate_and_save(model_id, save_dir = '.\data\models\\')

def read_jsonl(src_file, dictionary):
    import jsonlines
    import re
    
    labels = []
    texts = []
    with open(src_file, 'r+') as src_data:
        for item in jsonlines.Reader(src_data):
            labels.append(int(item['target']))
            tmp_text = item['func']
            tmp_text = tmp_text.replace('\n', '')
            tmp_text = tmp_text.replace('\t', ' ')
            strinfo = re.compile('/\*.*\*/')
            tmp_text = strinfo.sub('', tmp_text)
            tmp_text = tmp_text.strip(' ','').split()
            texts.append(tmp_text)
    text_tokens = []
    for text in texts:
        text = dictionary.index(text)
        text_tokens.append(text)
    texts = text_tokens
    return texts, labels

def test_src():
    device_ids = [0]
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"   
    vocab_file = GPT_CONUT_TRAINER_DIR + '..\..\data\\vocabulary\\vocabulary.txt'
    train_file = GPT_CONUT_TRAINER_DIR + '..\..\data\data\\training_bpe.txt'
    valid_file = GPT_CONUT_TRAINER_DIR + '..\..\data\data\\validation_bpe.txt'
    src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    work_dir = os.path.dirname(src_dir)
    gpt_file = os.path.join(work_dir, 'data/models', 'codet5p')
    # gpt_file = GPT_CONUT_TRAINER_DIR + '..\..\data\models\code_gpt.pt'
    
    dictionary = Dictionary(vocab_file, min_cnt=0)
    print('dictionary initialized, vocab size:{}'.format(len(dictionary)))

    train_loader = GPTCoNuTDataLoader(train_file, dictionary)
    valid_loader = GPTCoNuTDataLoader(valid_file, dictionary)
    print('data loader initialized, train size:{}, validate size:{}'.
          format(train_loader.total_size, valid_loader.total_size))

    # 模型结构读取
    trainer = detector_trainer(train_loader, valid_loader, dictionary)    
    model_id = 2
    epochs = 1
    model_file = '.\data\models\gpt_conut_test_' + str(model_id) + '.pt'
    loaded = torch.load(
        model_file, map_location='cpu'
    )
    gpt_model = T5EncoderModel.from_pretrained(gpt_file)
    trainer.model = GPTDetector(
        trainer.dictionary, embed_dim=384, max_positions=1024, embed_model = gpt_model,
    ).cuda()
    trainer.model.load_state_dict(loaded['model'])
    trainer.optimizer = torch.optim.Adam(trainer.model.parameters(), lr=6.25e-5)

    
    # 加载数据
    src_file = GPT_CONUT_TRAINER_DIR + '..\..\data\data\\test.jsonl'
    texts, labels = read_jsonl(src_file, dictionary)
    # trainer.validate_and_save(model_id, save_dir = '.\data\models\\')
    with torch.no_grad():
        start_time = time.time()
        oom = 0 
        val_loss, val_fconv_loss, val_lm_loss = [], [], []


        # 取batch

        # 输入到模型预测

    print(cor/len(trainer.valid_loader.dataset))
    print(cor)
    print('*'*30)
    return val_loss    

def train(model_id, epochs = 10, dropout=0.5, lr=6.25e-6):
    # device_ids = [0]
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"   
    vocab_file = GPT_CONUT_TRAINER_DIR + '..\..\data\\vocabulary\\vocabulary.txt'
    train_file = GPT_CONUT_TRAINER_DIR + '..\..\data\data\\training_bpe.txt'
    valid_file = GPT_CONUT_TRAINER_DIR + '..\..\data\data\\validation_bpe.txt'
    src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    work_dir = os.path.dirname(src_dir)
    gpt_file = os.path.join(work_dir, 'data/models', 'codet5p')
    # gpt_file = GPT_CONUT_TRAINER_DIR + '..\..\data\models\code_gpt.pt'
    
    dictionary = Dictionary(vocab_file, min_cnt=0)
    print('dictionary initialized, vocab size:{}'.format(len(dictionary)))

    train_loader = GPTCoNuTDataLoader(train_file, dictionary)
    valid_loader = GPTCoNuTDataLoader(valid_file, dictionary)
    print('data loader initialized, train size:{}, validate size:{}'.
          format(train_loader.total_size, valid_loader.total_size))

    # 模型结构读取
    trainer = detector_trainer(train_loader, valid_loader, dictionary)


    trainer.train(model_id, epochs, save_dir = '.\data\models\\', gpt_file=gpt_file, dropout=dropout, lr = lr )# 之前是‘\data\models\\’的时候，会报错，应该是格式的问题；此外，绝对路径可以用，但是不能包含中文，应该会变成乱码无法识别

if __name__ == '__main__':
    # device_ids = [0, 1, 2, 3]
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    
    model_id = 2
    epochs = 10
    train(1, 10, 0.5, 6.25e-6)
    # test()
    # test_src()





