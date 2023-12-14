import codecs
import json
import os

# RERANK_DIR = os.path.abspath(__file__)[: os.path.abspath(__file__).rindex('/') + 1]

RERANK_DIR = os.path.abspath(__file__)[: os.path.abspath(__file__).rindex('\\') + 1]

def read_defects4j_meta(meta_path):
    fp = codecs.open(meta_path, 'r', 'utf-8')
    meta = []
    for l in fp.readlines():
        proj, bug_id, path, start_loc, end_loc = l.strip().split('\t')
        meta.append([
            proj, bug_id, path, start_loc, end_loc + 1
        ])
    return meta


def read_quixbugs_meta(meta_path):
    fp = codecs.open(meta_path, 'r', 'utf-8')
    meta = []
    for l in fp.readlines():
        proj, loc = l.strip().split('\t')# proj-项目名，loc-location，bug的定位或者说序号，有些只有一个，有些是多个，用'-'连接
        if '-' in loc:
            start_loc, end_loc = loc.split('-')
            start_loc = int(start_loc)
            end_loc = int(end_loc)
        else:
            start_loc, end_loc = int(loc), int(loc) + 1
        meta.append([
            proj, str(start_loc), str(end_loc)
        ])
    return meta


def read_hypo(hypo_path):
    fp = codecs.open(hypo_path, 'r', 'utf-8')
    hypo = {}
    for l in fp.readlines():
        l = l.strip().split()# S-x XX XX XX XX XXX XXXX
        if l[0][:2] == 'S-':
            id = int(l[0][2:])# 上面有划分，此外，加冒号是为了两位数之上的情况
            src = ' '.join(l[1:]).strip()
            src = src.replace('@@ ', '')
            hypo[id] = {'src': src, 'patches': []}
        if l[0][:2] == 'H-':
            id = int(l[0][2:])
            patch = ' '.join(l[2:]).strip()
            patch = patch.replace('@@ ', '')
            score = float(l[1])
            hypo[id]['patches'].append([patch, score])
    return hypo


def cure_rerank(meta, hypo_path_list, output_path):
    # the patch with same rank from different models are grouped together
    group_by_rank = {}
    for hypo_path in hypo_path_list:# 两个模型补丁的地址
        hypo = read_hypo(hypo_path)
        print('finish loading', hypo_path)
        # hypo:
        # XX:           这个就是id
        # {
        #   'src':xxxxxxx;
        #   'patches':a         a是一个列表，a[i]同样也是一个列表，a[i][0]是补丁内容，a[i][1]是得分
        # }
        for id in hypo:
            if id not in group_by_rank:
                group_by_rank[id] = {'src': hypo[id]['src'], 'patches': []}
            for rank, (patch, score) in enumerate(hypo[id]['patches']):
                if rank >= len(group_by_rank[id]['patches']):# 判断字典的大小
                    group_by_rank[id]['patches'].append([])
                group_by_rank[id]['patches'][rank].append([patch, score])
    
    # group_by_rank将两个模型生成的补丁连接到一起
    # the patch with same rank are ranked by scores
    reranked_hypo = {}
    print('start ranking')
    for id in group_by_rank:
        key = '-'.join(meta[id])
        reranked_hypo[key] = {'src': group_by_rank[id]['src'], 'patches': []}# 把项目对应的补丁放在一起

        added_patches = set()
        for patches_same_rank in group_by_rank[id]['patches']:# 单个的补丁
            ranked_by_score = sorted(patches_same_rank, key=lambda e: e[1], reverse=True)# 这里一般会有两个，分别是两个模型生成的结果
            for patch, score in ranked_by_score:
                if patch not in added_patches:
                    added_patches.add(patch)
                    reranked_hypo[key]['patches'].append({'patch': patch, 'score': score})
            if len(added_patches) >= 5000:# 只选择5000条补丁
                break

    print('dumping result in json file')
    json.dump(reranked_hypo, open(output_path, 'w'), indent=2)

import time

if __name__ == '__main__':
    meta_path = RERANK_DIR + '../../candidate_patches/QuixBugs/meta.txt'
    quixbugs_meta = read_quixbugs_meta(meta_path)
    # hypo_path_list = [RERANK_DIR + '../../data/patches/gpt_conut_3.txt'] + \
    #                  [RERANK_DIR + '../../data/patches/gpt_fconv_1.txt']
    # hypo_path_list = [RERANK_DIR + '../../data/patches/gpt_conut_2_quixbugs_only_eva.txt']
    hypo_path_list = [RERANK_DIR + '../../data/patches/gpt_conut_15_quixbugs.txt']
    # gpt_conut_parse_4
    # output_path = RERANK_DIR + '../../data/patches/reranked_patches_2_quixbugs_only_eva.json'
    output_path = RERANK_DIR + '../../data/patches/reranked_patches_1_quixbugs.json'
    start_time_all = time.time()
    cure_rerank(quixbugs_meta, hypo_path_list, output_path)
    end_time_all = time.time()
    generate_time = end_time_all - start_time_all
    print('rerank time:',end_time_all - start_time_all)
    time_file = './data/models/time.txt'
    with open(time_file, 'a+') as f:
        info = 'rerank : hypo:{}, out:{}, time:{}s, dir:{}\n'.\
            format('gpt_conut_parse_2_quixbugs', 'reranked_patches_parse_2_quixbugs', generate_time, output_path )
        f.write(info)   
