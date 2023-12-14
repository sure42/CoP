import codecs
import torch
import sys
import os
import javalang
from transformers import OpenAIGPTLMHeadModel
import time
GENERATOR_DIR = os.path.abspath(__file__)[: os.path.abspath(__file__).rindex('\\') + 1]
# sys.path.append(GENERATOR_DIR + '../models/')
# sys.path.append(GENERATOR_DIR + '../dataloader/')
sys.path.append(GENERATOR_DIR + '../models/')
sys.path.append(GENERATOR_DIR + '../dataloader/')
from gpt_conut_data_loader import GPTCoNuTDataLoader
from gpt_fconv_data_loader import GPTFConvDataLoader
from identifier_data_loader import IdentifierDataLoader
from dictionary import Dictionary
from gpt_conut import GPTCoNuTModel
from gpt_fconv import GPTFConvModel
from beamsearch import BeamSearch
import tokenization

def check_parse(tmp_str):
    flag_src = True
    if '/ /' in tmp_str or '\\' in tmp_str:
        flag_src = False
        return flag_src
    tmp_str = tokenization.token2statement(tmp_str.split(' '), ['3'], ['tmp_str'])
    tmp_str  = tmp_str[0].replace('@@ ', '')
    try:
        tree = javalang.parse.parse(tmp_str)# 前+h+后  javalangparser. Parsebxception
    except javalang.parser.JavaSyntaxError as e:#  javalang.parser.ParseException
        flag_src = False
    return flag_src

class Generator():
    def __init__(self, model, dictionary, data_loader, beam_size=10):
        self.model = model
        self.dictionary = dictionary
        self.data_loader = data_loader
        self.beam_size = beam_size
        self.beamsearch = BeamSearch(model, dictionary, beam_size)
        # print(self.model, beam_size)

    def generate(self, output_path, output_file_parse):

        wp = codecs.open(output_path, 'w', 'utf-8')
        wp_parse = codecs.open(output_file_parse, 'w', 'utf-8')
        self.data_loader.load_data(0, self.data_loader.total_size)
        for i in range(self.data_loader.total_size):
            start_time_ge = time.time()
            print(i, '/', self.data_loader.total_size)
            data = self.data_loader.dataset[i]
            # try:
            self.beamsearch.beam_size = self.beam_size
            sample = self.data_loader.dataset.collater([data])
            with torch.no_grad():
                # if isinstance(self.model, GPTCoNuTModel):
                #     hypothesis = self.beamsearch.generate_gpt_conut(sample)
                # elif isinstance(self.model, GPTFConvModel):
                #     hypothesis = self.beamsearch.generate_gpt_fconv(sample)
                if isinstance(self.model, GPTCoNuTModel):
                    hypothesis = self.beamsearch.generate_gpt_conut_with_detect(sample, 1)
                    # hypothesis = self.beamsearch.generate_gpt_conut(sample)

            # except Exception as e:# 这个格式可以参考
            #    print(e)
            #    continue

            hypothesis_parse = []
            # 判断源程序是否可编译
            src_str = self.dictionary.string(sample['prev_context'].view(-1)) + ' ' + \
                            self.dictionary.string(sample['net_input']['src_tokens'].view(-1)) + ' ' + \
                            self.dictionary.string(sample['behind_context'].view(-1))
            skip_flag = True
            if check_parse(src_str):
                skip_flag = False
                for h in hypothesis:
                    hyp_str = self.dictionary.string(sample['prev_context'].view(-1)) + ' ' + \
                                self.dictionary.string(h['hypo']) + ' ' + \
                                self.dictionary.string(sample['behind_context'].view(-1))

                    if check_parse(hyp_str):
                        hypothesis_parse.append(h)
            else:
                hypothesis_parse = hypothesis

            id = str(sample['id'].item())
            wp.write('S-{}\t'.format(id))# 源代码
            wp.write(self.dictionary.string(data['source']) + '\n')
            wp.write('T-{}\t'.format(id))# 目标补丁
            wp.write(self.dictionary.string(data['target']) + '\n')
            wp.write('skip_flag-{}\t'.format(id))
            wp.write(str(skip_flag) + '\n')
            for h in hypothesis:# 生成的候选
                wp.write('H-{}\t{}\t'.format(id, str(h['final_score'])))
                wp.write(self.dictionary.string(h['hypo']) + '\n')
                wp.write('P-{}\t'.format(id))
                wp.write(' '.join(str(round(s.item(), 4)) for s in h['score']) + '\n')# round小数位数

            wp_parse.write('S-{}\t'.format(id))# 源代码
            wp_parse.write(self.dictionary.string(data['source']) + '\n')
            wp_parse.write('T-{}\t'.format(id))# 目标补丁
            wp_parse.write(self.dictionary.string(data['target']) + '\n')
            wp_parse.write('skip_flag-{}\t'.format(id))
            wp_parse.write(str(skip_flag) + '\n')
            for h in hypothesis_parse:# 生成的候选
                wp_parse.write('H-{}\t{}\t'.format(id, str(h['final_score'])))
                wp_parse.write(self.dictionary.string(h['hypo']) + '\n')
                wp_parse.write('P-{}\t'.format(id))
                wp_parse.write(' '.join(str(round(s.item(), 4)) for s in h['score']) + '\n')# round小数位数
            
            end_time_ge = time.time()
            generate_time = end_time_ge - start_time_ge
            print('generate test time:',generate_time)
            time_file = './data/models/time_ge.txt'
            with open(time_file, 'a+') as f:
                info = 'generate : gpt:{}, detector:{}, id:{}, time:{}s, dir:{}\n'.\
                    format('1', '1', generate_time, id, output_file )
                f.write(info)
            wp_parse.write('time-{}\t'.format(id))
            wp_parse.write(str(generate_time) + '\n')
            wp.write('time-{}\t'.format(id))
            wp.write(str(generate_time) + '\n')
        with open(time_file, 'a+') as f:
            info = '****************************************\n'
            f.write(info)                          
        wp.close()
        wp_parse.close()





def generate_gpt_conut(vocab_file, model_file, input_file, identifier_txt_file, identifier_token_file, output_file, output_file_parse,  beam_size):

    dictionary = Dictionary(vocab_file, min_cnt=0)
    print(len(dictionary))
    loaded = torch.load(
        model_file, map_location='cpu'
    )
    config = loaded['config']
    gpt_config = config['embed_model_config']
    gpt_config.attn_pdrop = 0
    gpt_config.embd_pdrop = 0
    gpt_config.resid_pdrop = 0
    gpt_model = OpenAIGPTLMHeadModel(gpt_config)# 加载GPT模型配置
    model = GPTCoNuTModel(
        dictionary=dictionary, embed_dim=config['embed_dim'],
        max_positions=config['max_positions'],
        src_encoder_convolutions=config['src_encoder_convolutions'],
        ctx_encoder_convolutions=config['ctx_encoder_convolutions'],
        decoder_convolutions=config['decoder_convolutions'],
        dropout=0, embed_model=gpt_model,
    )

    model.load_state_dict(loaded['model'])
    identifier_loader = IdentifierDataLoader(
        dictionary, identifier_token_file, identifier_txt_file
    )
    data_loader = GPTCoNuTDataLoader(
        input_file, dictionary,
        identifier_loader=identifier_loader
    )
    generator = Generator(model, dictionary, data_loader, beam_size=beam_size)
    print('start generate')
    generator.generate(output_file, output_file_parse)




def generate_gpt_fconv(vocab_file, model_file, input_file, identifier_txt_file, identifier_token_file, output_file, output_file_parse,  beam_size):
    dictionary = Dictionary(vocab_file, min_cnt=0)
    print(len(dictionary))
    loaded = torch.load(
        model_file, map_location='cpu'
    )
    config = loaded['config']
    gpt_config = config['embed_model_config']
    gpt_config.attn_pdrop = 0
    gpt_config.embd_pdrop = 0
    gpt_config.resid_pdrop = 0
    gpt_model = OpenAIGPTLMHeadModel(gpt_config)
    model = GPTFConvModel(
        dictionary=dictionary, embed_dim=config['embed_dim'],
        max_positions=config['max_positions'],
        encoder_convolutions=config['encoder_convolutions'],
        decoder_convolutions=config['decoder_convolutions'],
        dropout=0, embed_model=gpt_model,
    )
    model.load_state_dict(loaded['model'])
    identifier_loader = IdentifierDataLoader(
        dictionary, identifier_token_file, identifier_txt_file
    )
    data_loader = GPTFConvDataLoader(
        input_file, dictionary,
        identifier_loader=identifier_loader
    )
    generator = Generator(model, dictionary, data_loader, beam_size=beam_size)
    print('start generate')
    generator.generate(output_file)


if __name__ == "__main__":
    vocab_file = GENERATOR_DIR + '../../data/vocabulary/vocabulary.txt'
    input_file = GENERATOR_DIR + '../../candidate_patches/QuixBugs/quixbugs_bpe.txt'
    identifier_txt_file = GENERATOR_DIR + '../../candidate_patches/QuixBugs/identifier.txt'
    identifier_token_file = GENERATOR_DIR + '../../candidate_patches/QuixBugs/identifier.tokens'
    beam_size = 1000
    # beam_size = 53*
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"

    model_file = GENERATOR_DIR + '../../data/models/gpt_conut_1.pt'
    output_file = GENERATOR_DIR + '../../data/patches/gpt_conut_1_quixbugs.txt'
    output_file_parse = GENERATOR_DIR + '../../data/patches/gpt_conut_parse_1_quixbugs.txt'

    generate_gpt_conut(vocab_file, model_file, input_file, identifier_txt_file, identifier_token_file, output_file, output_file_parse,  beam_size)
 
    # model_file = GENERATOR_DIR + '..\..\data\models\gpt_fconv_1.pt'
    # output_file = GENERATOR_DIR + '..\..\data\patches\gpt_fconv_2.txt'
    # generate_gpt_fconv(vocab_file, model_file, input_file, identifier_txt_file, identifier_token_file, output_file, beam_size)
