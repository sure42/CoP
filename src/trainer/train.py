import sys
import os
import argparse
import torch
import codecs


# os.path.dirname 返回给定文件路径的目录部分，也就是去掉最后一级文件名或者目录名之后的路径
# os.path.sep 替换直接写死的 '\\'
# os.getcwd() 工作目录    os.path.abspath(__file__)脚本的绝对路径
src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
work_dir = os.path.dirname(src_dir)

model_path = os.path.join(src_dir, "models")
sys.path.append(model_path) 
dataloader_path = os.path.join(src_dir, "dataloader")
sys.path.append(dataloader_path)

# print(src_dir) # CoP\src
# print(work_dir) # CoP\src
# print(model_path)
# print(dataloader_path)

from dictionary import Dictionary
from MyDataloader import MyDataset

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
 

def get_args():
    argparser = argparse.ArgumentParser(sys.argv[0], conflict_handler='resolve')
    argparser.add_argument("--dataset",  type=str, default='mix')
    argparser.add_argument("--train", action="store_true", help="use the training model (default: False)")
    argparser.add_argument("--test", action="store_true", help="use the testing model (default: False)")
    argparser.add_argument("--batch_size", "--batch", type=int, default=4)
    argparser.add_argument("--lr", type=float, default=5e-5)
    argparser.add_argument("--epoch", type=float, default=3)
    # argparser.add_argument("--epoch", type=float, default=3)
    argparser.add_argument("--dropout", type=float, default=0.01)
    args = argparser.parse_args()
    return args

def main():
    args = get_args()


    # 一些配置
    device_ids = [0]
    os.environ["CUDA_VISIBLE_DEVICES"] = "0" 

    # 读取数据集

    os.path.join(work_dir, "documents/myfile.txt") # os.path.join('dir1', 'dir2', 'file.txt')
    
    vocab_file = os.path.join(work_dir, 'data', 'vocabulary/vocabulary.txt')
    dictionary = Dictionary(vocab_file)

    if args.train:
        train_file = os.path.join(work_dir, 'data', 'data/training_bpe.txt')
        train_loader = MyDataset(train_file, dictionary)
    if args.test:
        valid_file = os.path.join(work_dir, 'data', 'data/validation_bpe.txt')
        valid_loader = MyDataset(valid_file, dictionary)
    
    i = 0

    # with open(vocab_file, 'r', encoding='utf-8') as fp:
    #     for line in fp.readlines():
    #         i = i+1
    #         print(line)
    
    print(i)
    # 模型加载

    # code - tokenizer - tensor - dataset - dataloader(自动有) - input/mask - model - output









if __name__ == "__main__":
    main()