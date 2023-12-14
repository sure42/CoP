import os
import sys
import json
import time
import codecs
import random
import numpy as np
import torch
import torch.nn as nn
from transformers import OpenAIGPTLMHeadModel
from transformers import AutoTokenizer, OpenAIGPTModel
print(os.path.abspath(__file__)+'\n')
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
GPT_CONUT_TRAINER_DIR = os.path.abspath(__file__)[: os.path.abspath(__file__).rindex('\\') + 1]
sys.path.append(GPT_CONUT_TRAINER_DIR + '../models/')
sys.path.append(GPT_CONUT_TRAINER_DIR + '../dataloader/')

# GPT_CONUT_TRAINER_DIR = os.path.abspath(__file__)[: os.path.abspath(__file__).rindex('/') + 1]
# sys.path.append(GPT_CONUT_TRAINER_DIR + '../models/')
# sys.path.append(GPT_CONUT_TRAINER_DIR + '../dataloader/')

from gpt_conut import GPTCoNuTModel
from dictionary import Dictionary
from gpt_conut_data_loader import GPTCoNuTDataLoader


class GPTCoNuTTrainer():
    def __init__(self, train_loader, valid_loader, dictionary, gpt_file):
        gpt_loaded = torch.load(gpt_file)# 从文件加载用torch.save()保存的对象。
        config = gpt_loaded['config']
        gpt_model = OpenAIGPTLMHeadModel(config).cuda()
        gpt_model.load_state_dict(gpt_loaded['model'])
        self.model = gpt_model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.dictionary = dictionary

        self.batch_size = 2
        self.load_size = 1000   # load 1200 samples from training data every time

        self.gpt_model = gpt_model
        self.model = None
        self.hyper_parameter = {}
        self.optimizer = None
        self.current_train_step = 0
        self.val_loss = {}

    def shuffle_dataset(self):
        indices = [i for i in range(len(self.train_loader.dataset))]
        random.seed(42)
        random.shuffle(indices)
        return indices

    def train_step(self, samples):
        self.model.train()
        self.current_train_step += 1
        self.optimizer.zero_grad()

        batch = self.train_loader.dataset.collater(samples) # 调整数据格式
        if torch.cuda.is_available():
            outputs = self.model(
                batch['net_input']['src_tokens'].cuda(),
                batch['net_input']['src_with_prev_context'].cuda(),
                batch['net_input']['ctx_tokens'].cuda(),
                prev_tokens_index=batch['target_index'].cuda(),
                prev_tokens_with_context=batch['target_with_prev_context'].cuda(),
                labels=batch['target'].cuda(),# 正确的补丁
            )
        else:
            outputs = self.model(
                batch['net_input']['src_tokens'],
                batch['net_input']['src_with_prev_context'],
                batch['net_input']['ctx_tokens'],
                prev_tokens_index=batch['target_index'],
                prev_tokens_with_context=batch['target_with_prev_context'],
                labels=batch['target'],
            )
        logits, avg_attn_scores, apr_loss, lm_loss = outputs[:4]
        loss = apr_loss + 0.3 * lm_loss
        loss.mean().backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 0.5, norm_type=2)# 是对所有的梯度乘以一个clip_coef，且clip_coef一定是小于1的；只解决梯度爆炸问题，不解决梯度消失问题
        self.optimizer.step()
        return loss.mean().item(), apr_loss.mean().item(), lm_loss.mean().item()

    def valid_step(self, samples):
        self.model.eval()
        batch = self.valid_loader.dataset.collater(samples)
        outputs = self.model(
            batch['net_input']['src_tokens'].cuda(),
            batch['net_input']['src_with_prev_context'].cuda(),
            batch['net_input']['ctx_tokens'].cuda(),
            prev_tokens_index=batch['target_index'].cuda(),
            prev_tokens_with_context=batch['target_with_prev_context'].cuda(),
            labels=batch['target'].cuda(),
        )
        logits, avg_attn_scores, apr_loss, lm_loss = outputs[:4]
        loss = apr_loss + 0.3 * lm_loss
        return loss.mean().item(), apr_loss.mean().item(), lm_loss.mean().item(), logits

    def validate_and_save(self, model_id, save_dir):
        oom = 0
        with torch.no_grad():
            val_loss, val_fconv_loss, val_lm_loss = [], [], []
            for i in range(0, self.valid_loader.total_size, self.batch_size):
                samples = [self.valid_loader.dataset[j]
                           for j in range(i, min(len(self.valid_loader.dataset), i + self.batch_size))]
                try:
                    loss, fconv_loss, lm_loss, logits = self.valid_step(samples)
                    val_loss.append(float(loss))
                    val_fconv_loss.append(float(fconv_loss))
                    val_lm_loss.append(float(lm_loss))
                except Exception as e:
                    oom += 1

            info = 'val loss:{}, val apr_loss:{}, val lm_loss:{}, val ppl:{}, oom:{}'.format(
                round(float(np.mean(val_loss)), 6),
                round(float(np.mean(val_fconv_loss)), 6),
                round(float(np.mean(val_lm_loss)), 6),
                round(float(np.exp(np.mean(val_loss))), 6),
                oom
            )
            print(info)

            val_loss = np.mean(val_fconv_loss)
            checkpoint = {
                'model': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'current_step': self.current_train_step,
                'config': self.model.module.config(),
                'val_loss': val_loss,
            }

            # save_dir = 'D:\\test\\'
            # # assert 1==2    
            # os.makedirs(save_dir+'models', exist_ok = True) 
            # torch.save(checkpoint , os.path.join(save_dir+'models','model.pth'))
            # assert 1==2

            torch.save(checkpoint, save_dir + 'gpt_conut_' + str(model_id) + '.pt')
            self.val_loss[model_id] = {
                'val_loss': val_loss,
                'hyper-parameter': str(self.hyper_parameter),
            }

        return val_loss

    def train(self, model_id, epochs, hyper_parameter, save_dir):
        start_time_all = time.time()
        self.hyper_parameter = hyper_parameter
        self.model = GPTCoNuTModel(
            self.dictionary, embed_dim=384, max_positions=1024,
            src_encoder_convolutions=self.hyper_parameter['src_encoder_convolutions'],
            ctx_encoder_convolutions=self.hyper_parameter['ctx_encoder_convolutions'],
            decoder_convolutions=self.hyper_parameter['decoder_convolutions'],
            dropout=self.hyper_parameter['dropout'], embed_model=self.gpt_model,
        ).cuda()
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=6.25e-5)
        self.model = nn.DataParallel(self.model, device_ids=device_ids)# 当迭代次数或者epoch足够大的时候，我们通常会使用nn.DataParallel函数来用多个GPU来加速训练
        # device_ids是所有可使用的GPU信号
        
        self.valid_loader.load_data(0, self.valid_loader.total_size)#加载所有的标识符和数据集

        # print(self.valid_loader.dataset[0]) # 这里实际上是调用__getitem__函数
        # assert 1==2
        count = 0

        for epoch in range(epochs):
            start_time = time.time()
            for i in range(0, self.train_loader.total_size, self.load_size):# 训练数据不是一次加载完毕的，每次加载load_size--1200条
                oom = 0
                self.train_loader.load_data(i, i + self.load_size)
                indices = self.shuffle_dataset()# 打乱
                train_loss, train_apr_loss, train_lm_loss = [], [], []

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
                        try:
                            loss, apr_loss, lm_loss = self.train_step(samples)
                            train_loss.append(loss)
                            train_apr_loss.append(apr_loss)
                            train_lm_loss.append(lm_loss)
                        except Exception as e:
                            # print(e)
                            # print(i, start, end)
                            oom += 1
                            # if(oom==2): 
                            #     assert 1==2

                        start = end
                        max_src, max_ctx, max_tgt = 0, 0, 0
                        samples = []
                        continue
                    max_src = max(max_src, len(sample['source']))
                    max_ctx = max(max_ctx, len(sample['prev_context']))
                    max_tgt = max(max_tgt, len(sample['target']))
                    end += 1# 下一轮
                    samples.append(sample)
                    count = count + 1
                if len(samples) > 0:# 剩余的数据
                    try:
                        loss, apr_loss, lm_loss = self.train_step(samples)
                        train_loss.append(loss)
                        train_apr_loss.append(apr_loss)
                        train_lm_loss.append(lm_loss)
                    except Exception as e:
                        oom += 1

                if (i // self.load_size) % 10 == 0:# 打印训练过程 10 1210 ... 
                    info = 'epoch:{}, load data:{}, lr:{}, loss:{}, apr_loss:{}, lm_loss:{}, time:{}s, oom:{}'.\
                        format(epoch + 1, i + self.load_size,
                               round(self.optimizer.param_groups[0]['lr'], 10),# round用于数字的四舍五入
                               round(float(np.mean(train_loss)), 6),
                               round(float(np.mean(train_apr_loss)), 6),
                               round(float(np.mean(train_lm_loss)), 6),
                               int(time.time() - start_time), oom
                               )
                    start_time = time.time()
                    print(str(model_id) + ' ' + info)

                if (i // self.load_size) % 100 == 0:# 100 1200000 1200000000 ...
                    self.validate_and_save(model_id, save_dir)
        print(count)
        print("*"*20)
        end_time_all = time.time()
        print('model train time:',end_time_all - start_time_all)
        train_time = end_time_all - start_time_all
        time_file = save_dir + 'time.txt'
        with open(time_file, 'a+') as f:
            info = 'train : epoch:{}, all data:{}, time:{}s, oom:{}, dir:{}\n'.\
                format(epochs, len(self.train_loader.dataset),
                        int(train_time), oom, save_dir + 'gpt_conut_' + str(model_id) + '.pt'
                )
            f.write(info)

        start_time_all = time.time()
        self.validate_and_save(model_id, save_dir)
        end_time_all = time.time()
        valid_time = end_time_all - start_time_all
        print('model test time:',end_time_all - start_time_all)
        with open(time_file, 'a+') as f:
            info = 'valid : epoch:{}, all data:{}, time:{}s, oom:{}, dir:{}\n'.\
                format(epochs, len(self.train_loader.dataset),
                        int(valid_time), oom, save_dir + 'gpt_conut_' + str(model_id) + '.pt'
                )
            f.write(info)        



if __name__ == '__main__':
    # device_ids = [0, 1, 2, 3]
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    device_ids = [0]
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"   
    vocab_file = GPT_CONUT_TRAINER_DIR + '../../data/vocabulary/vocabulary.txt'
    train_file = GPT_CONUT_TRAINER_DIR + '../../data/data/training_bpe.txt'
    valid_file = GPT_CONUT_TRAINER_DIR + '../../data/data/validation_bpe.txt'
    gpt_file = GPT_CONUT_TRAINER_DIR + '../../data/models/code_gpt.pt'
    
    dictionary = Dictionary(vocab_file, min_cnt=0)
    print('dictionary initialized, vocab size:{}'.format(len(dictionary)))

    train_loader = GPTCoNuTDataLoader(train_file, dictionary)
    valid_loader = GPTCoNuTDataLoader(valid_file, dictionary)
    print('data loader initialized, train size:{}, validate size:{}'.
          format(train_loader.total_size, valid_loader.total_size))

    trainer = GPTCoNuTTrainer(train_loader, valid_loader, dictionary, gpt_file)

    hyper_parameter = {
        'src_encoder_convolutions': ((192, 5),) * 1,
        'ctx_encoder_convolutions': ((384, 5),) * 1,
        'decoder_convolutions': ((192, 5),) * 1,
        'dropout': 0.01,
    }
    model_id = 4
    epochs = 5


    # trainer.train(model_id, epochs, hyper_parameter, save_dir=GPT_CONUT_TRAINER_DIR + '..\..\data\models\\')
    trainer.train(model_id, epochs, hyper_parameter, save_dir = './data/models/')# 之前是‘\data\models\\’的时候，会报错，应该是格式的问题；此外，绝对路径可以用，但是不能包含中文，应该会变成乱码无法识别