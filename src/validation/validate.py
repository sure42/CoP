import json
import os
import logging
import shutil
import subprocess
import time
import threading
import concurrent.futures
import sys
import random
from validate_quixbugs import validate_quixbugs
from validate_defects4j import validate_defects4j
from count_module import count, count_lock
VALIDATE_DIR = os.path.abspath(__file__)[: os.path.abspath(__file__).rindex('\\') + 1]
# VALIDATE_QUIXBUGS_DIR = os.path.abspath(__file__)[: os.path.abspath(__file__).rindex('/') + 1]
sys.path.append(VALIDATE_DIR + '../dataloader/')


def patches_read(reranked_result_path):
    
    pass

def thread_test(id):
    for i in range(5):
        print(id, random.randint(1, 10))



def log_config():

    logger = logging.getLogger('ValidationLogger')
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler('ValidationLogger.log')
    fh.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger

def average_split(lst, n):
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]

def validate(reranked_result_path, output_dir, tmp_dir, threads_num = 5):

    global count
    mylogger  = log_config()
    mylogger.info("*"*20)
    mylogger.info("start validation!")

    current_path = os.path.abspath(os.path.dirname(__file__))
    mylogger.info("current path:" + current_path)
    mylogger.info("threads number:" + str(threads_num))
    mylogger.info("output path:" + output_dir)
    # 读取数据
    reranked_result = json.load(open(reranked_result_path, 'r'))

    # 验证结果
    validated_results = {}
    # 线程返回值
    
    times = []
    
    cnt, right = 0, 0

    # 给出临时文件夹的位置
    tmp_dir_threads = []
    for i in range(threads_num):
        tmp_dir_threads.append(tmp_dir + '/Thread_{}'.format(i))
        # output_dirs = output_dir + 'Tmp_patches_Thread_{}.json'.format(i)
    
    # print(tmp_dir_threads)

    for key in reranked_result:
        cnt += 1
        # if cnt < 33:
        #     continue
        print("-"*20)
        print("Start validate No.{}".format(cnt))
        
        if 'quixbug' in reranked_result_path:
            proj, start_loc, end_loc = key.split('-')
            print(right, '/', cnt, proj)
        elif 'defects' in reranked_result_path:
            proj, bug_id, path, start_loc, end_loc = key.split('-')
            print(right, '/', cnt, proj)

        validated_results[key] = {'src': reranked_result[key]['src'], 'timecost': 0, 'flag': False, 'patches': []}
    
        start_time = time.time()
        
        # 分割验证数据
        tokenized_patchs = reranked_result[key]['patches']
        # for i in range(threads_num):
        thread_pathes = average_split(tokenized_patchs, threads_num)

        results = []
        # 多线程 
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # 将数据块提交给线程池处理
            if 'quixbug' in reranked_result_path:
                futures = [executor.submit(validate_quixbugs, i, thread_pathes[i], tmp_dir_threads[i], proj, start_loc, end_loc, cnt, right) for i in range(threads_num)]
            elif 'defects' in reranked_result_path:
                futures = [executor.submit(validate_defects4j, i, thread_pathes[i], tmp_dir_threads[i], proj, bug_id, path, start_loc, end_loc, cnt, right) for i in range(threads_num)]
            
            # 等待所有任务完成
            # concurrent.futures.wait(futures)
            flag = False
            for future in concurrent.futures.as_completed(futures):
                # 在这里接受返回值，即验证的补丁，可能为空
                correct_repair, validated_result = future.result()
                # results.extend(future.result() )
                results.extend(validated_result)
                if correct_repair == True:
                    flag = True
                    
            if flag:
                right += 1

        # 把结果存到validated_results里
        # 记录验证单个Bug的时间
        end_time = time.time()
        times.append(end_time - start_time)
        Str = "Finish validate No.{}\nTake time: {}".format(cnt, times[-1])
        print("-"*20)

        validated_results[key]['patches']= results
        validated_results[key]['timecost']= times[-1]
        validated_results[key]['flag']= flag

        # 将结果写入文件

        # 这里还是有一点问题，理想是先写入，然后每次都续写，现在应该是覆盖
        # 这里没有其他办法，只能是每次都覆盖，否则的话代码太复杂
        json.dump(validated_results, open(output_dir, 'w'), indent=2)  


    print("-"*20)
    Str = "Finish validation!\n"
    Str += "The validated projects nums: {}\n".format(cnt)
    Str += "The fixed projects nums: {}\n".format(right)
    Str += "The average time per project: {}\n".format(sum(times)/len(times))
    mylogger.info(Str)


if __name__ == '__main__':
    
    model_ids = 2
    
    reranked_result_path = VALIDATE_DIR + '../../data/patches/quixbugs_reranked_patches_parse_{}.json'.format(model_ids)
    output_dir = VALIDATE_DIR + '../../data/patches/quixbugs_validated_patches_parse_{}_time.json'.format(model_ids)
    tmp_dir = VALIDATE_DIR + 'tmp/validate_quixbugs'
    
    # reranked_result_path = VALIDATE_DIR + '../../data/patches/reranked_patches_{}_defects4jv1.2_parse.json'.format(model_ids)
    # output_path = VALIDATE_DIR + '../../data/patches/validated_patches_{}_defects4jv1.2_parse.json'.format(model_ids)
    # tmp_dir = VALIDATE_DIR + 'tmp/validate_d4j'

    Threads_num = 5
    validate(reranked_result_path, output_dir, tmp_dir, Threads_num)