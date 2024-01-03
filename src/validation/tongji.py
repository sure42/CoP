import json
import os
import shutil
import time
import subprocess
import sys


VALIDATE_QUIXBUGS_DIR = os.path.abspath(__file__)[: os.path.abspath(__file__).rindex('\\') + 1]
# VALIDATE_QUIXBUGS_DIR = os.path.abspath(__file__)[: os.path.abspath(__file__).rindex('/') + 1]
sys.path.append(VALIDATE_QUIXBUGS_DIR + '../dataloader/')


def jisuan(com_list, all_list):

    a_sum = 0
    b_sum = 0
    for a, b in zip(com_list, all_list):
        if a != 0:
            a_sum += a
            b_sum += b
    print(a_sum, b_sum)
    return a_sum/b_sum

# 可编译率


# 补丁情况
def reapir(reranked_result):
    k = 0
    for key in reranked_result:
        i = 0
        patches = reranked_result[key]['patches']
        flag = True
        for item in patches:
            i += 1
   
            if item['correctness'] == 'plausible' and flag :
                flag = False
                print(key)
                k+=1
            if item['correctness'] == 'plausible':
                print("----"+item['patch'], i)
    print(k)


def top(reranked_result):
    k = 0
    pos = 0
    for key in reranked_result:
        i = 0
        patches = reranked_result[key]['patches']
        flag = True
        for item in patches:
            i += 1

            if i > 1000:
                break
            if item['correctness'] == 'plausible' and flag :
                flag = False
                k+=1
                pos += i
                print(key)
            if item['correctness'] == 'plausible':
                print("----"+item['patch'], i)
    print(k)
    print(pos)
result_path = VALIDATE_QUIXBUGS_DIR + '..\\..\\data\\patches\\defects4jv2.0_validated_patches_1_parse.json'
# result_path = VALIDATE_QUIXBUGS_DIR + '..\\..\\data\\patches\\quixbugs_validated_patches_parse_2.json'
reranked_result = json.load(open(result_path, 'r'))
all_list = []
# com_list = []
top(reranked_result)

# k =0
# com_30 = 0
# com_100 = 0
# com_500 = 0
# com_1000 = 0
# for key in reranked_result:
#     count_all = 0
#     count_com = 0
#     item_lits = {
#         'length':0,
#         '30':0,
#         '100':0,
#         '500':0,
#         '1000':0,
#     }   
#     i = 0
#     patches = reranked_result[key]['patches']
#     item_lits['length'] = len(patches)
#     flag = True
#     for item in patches:
#         i += 1
#         if item['correctness'] != 'uncompilable':
#             count_com += 1
#         if i == 30:
#             item_lits['30'] = count_com
#         if i == 100:
#             item_lits['100'] = count_com
#         if i == 500:
#             item_lits['500'] = count_com
#         if i == 1000:
#             item_lits['1000'] = count_com     
#         if item['correctness'] == 'plausible' and flag :
#             flag = False
#             print(key)
#             k+=1
#         if item['correctness'] == 'plausible':
#             print("----"+item['patch'], i)
# print(k)
