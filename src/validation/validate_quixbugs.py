import json
import os
import shutil
import time
import subprocess
import sys
# RERANK_DIR = os.path.abspath(__file__)[: os.path.abspath(__file__).rindex('\\') + 1]
VALIDATE_QUIXBUGS_DIR = os.path.abspath(__file__)[: os.path.abspath(__file__).rindex('\\') + 1]
# VALIDATE_QUIXBUGS_DIR = os.path.abspath(__file__)[: os.path.abspath(__file__).rindex('/') + 1]
sys.path.append(VALIDATE_QUIXBUGS_DIR + '../dataloader/')

import tokenization
import chardet

from count_module import count, count_lock

def command_with_timeout(cmd, timeout=5):
    # cmd = ' '.join(cmd)
    cmd = ['cmd.exe', '/c'] + cmd
    # cmd_A.extend(cmd)

    p = subprocess.Popen(cmd, shell=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE, universal_newlines=False)
    t_beginning = time.time()
    while True:
        if p.poll() is not None:
            break
        seconds_passed = time.time() - t_beginning
        if timeout and seconds_passed > timeout:
            p.terminate()
            return 'TIMEOUT', 'TIMEOUT'
        time.sleep(1)
    # 输出编码
    # encoding = sys.stdout.encoding or 'utf-8'

    out, err = p.communicate()
    # out = out.decode(chardet.detect(out)['encoding'])
    # err = err.decode(chardet.detect(err)['encoding']) 
    # # 将字节流解码为字符串
    # if out == b'':
    #     result = chardet.detect(err)
    # if err == b'':
    #     result = chardet.detect(out)
    # # encoding = result['encoding']
    if out != b'':
        out = out.decode(chardet.detect(out)['encoding'])
    if err != b'':
        err = err.decode(chardet.detect(err)['encoding'])  
    return out, err


def compile_fix(filename, tmp_dir):
    FNULL = open(os.devnull, 'w')
    p = subprocess.call(['cmd.exe', '/c', "javac",
                         tmp_dir + "Node.java",
                         tmp_dir + "WeightedEdge.java",
                         filename], stderr=FNULL)
    return False if p else True


def quixbugs_test_suite(algo, quixbugs_dir):
    QUIXBUGS_MAIN_DIR = quixbugs_dir
    CUR_DIR = os.getcwd()
    FNULL = open(os.devnull, 'w')
    try:
        os.chdir(QUIXBUGS_MAIN_DIR)# os.chdir()方法用于改变当前工作目录到指定的路径
        # p1 = subprocess.Popen(['cmd.exe', '/c', "javac", "-cp", "lib/junit-4.13.2.jar;lib/hamcrest-core-1.3.jar;", 
        #                         "java_testcases/junit/" + algo.upper() + "_TEST.java"],
        #                         stdout=subprocess.PIPE, stderr=FNULL, universal_newlines=True)# 编译
        out, err = command_with_timeout(
            ["java", "-cp", "lib/junit-4.13.2.jar;lib/hamcrest-core-1.3.jar;",
             "org.junit.runner.JUnitCore", "java_testcases.junit." + algo.upper() + "_TEST"], timeout=5
        )# 执行

        os.chdir(CUR_DIR)
        if "FAILURES" in str(out) or "FAILURES" in str(err):
            return 'wrong'
        elif "TIMEOUT" in str(out) or "TIMEOUT" in str(err):
            return 'timeout'
        else:
            print(str(out))
            print(str(err))
            # 需要检查一下输出，不应该这么多plausible
            return 'plausible'
    except Exception as e:
        print(e)
        os.chdir(CUR_DIR)
        return 'uncompilable'


def insert_fix_quixbugs(file_path, start_loc, end_loc, patch):
    shutil.copyfile(file_path, file_path + '.bak')

    with open(file_path, 'r') as file:
        data = file.readlines()

    patched = False
    with open(file_path, 'w') as file:
        for idx, line in enumerate(data):
            if start_loc - 1 <= idx < end_loc - 1:
                if not patched:
                    file.write(patch)
                    patched = True
            else:
                file.write(line)

    return file_path + '.bak'


def get_strings_numbers(file_path, loc):
    numbers_set = {}
    strings_set = {}
    with open(file_path, 'r') as file:
        data = file.readlines()
        for idx, line in enumerate(data):
            dist = loc - idx - 1
            strings, numbers = tokenization.get_strings_numbers(line)# 取出数字
            for num in numbers:
                if num != '0' and num != '1':
                    if num in numbers_set:
                        numbers_set[num] = min(dist, numbers_set[num])
                    else:
                        numbers_set[num] = dist
            for str in strings:
                if str in strings_set:
                    strings_set[str] = min(dist, strings_set[str])
                else:
                    strings_set[str] = dist
    final_strings = []
    final_numbers = []
    for k, v in numbers_set.items():
        final_numbers.append([k, v])
    for k, v in strings_set.items():
        final_strings.append([k, v])
    final_numbers.sort(key=lambda x: x[1])
    final_strings.sort(key=lambda x: x[1])
    return final_strings, final_numbers


def validate_quixbugs(id, tokenized_patchs, tmp_dir, proj, start_loc, end_loc, cnt, right):

    global count, count_lock

    count = False

    if not os.path.exists(tmp_dir):
        command_with_timeout(['mkdir', tmp_dir])

    command_with_timeout(['rd', '/s', '/q', tmp_dir + '/java_programs/'])
    command_with_timeout(['mkdir', tmp_dir + '/java_programs'])
    
    shutil.copyfile(tmp_dir + "/java_programs_bak/" + proj + '.java',
                    tmp_dir + "/java_programs/" + proj + '.java')
    shutil.copyfile(tmp_dir + "/java_programs_bak/Node.java", tmp_dir + "/java_programs/Node.java")
    shutil.copyfile(tmp_dir + "/java_programs_bak/WeightedEdge.java", tmp_dir + "/java_programs/WeightedEdge.java")
    # 记录结果
    validated_result = []

    start_time = time.time()
    correct_repair = False
    for tokenized_patch in tokenized_patchs:
        # validate 5 hours for each bug at most
        if time.time() - start_time > 5 * 3600:
            break
        # validate 5000 patches for each bug at most
        if len(tokenized_patchs) >= 5000:
            break
        filename = tmp_dir + "/java_programs/" + proj + '.java'

        score = tokenized_patch['score']
        tokenized_patch = tokenized_patch['patch']

        strings, numbers = get_strings_numbers(filename, (int(start_loc) + int(end_loc)) // 2)
        strings = [item[0] for item in strings][:5]
        numbers = [item[0] for item in numbers][:5]
        # one tokenized patch may be reconstructed to multiple source-code patches
        reconstructed_patches = tokenization.token2statement(tokenized_patch.split(' '), numbers, strings)
        # validate most 5 source-code patches come from the same tokenized patch
        for patch in reconstructed_patches[:5]:
            patch = patch.strip()
            insert_fix_quixbugs(filename, int(start_loc), int(end_loc), patch)
            compile = compile_fix(filename, tmp_dir + "/java_programs/")
            correctness = 'uncompilable'
            if compile:
                correctness = quixbugs_test_suite(proj, quixbugs_dir=tmp_dir)
                if correctness == 'plausible':
                    if not correct_repair:
                        right += 1
                        correct_repair = True

                        # 通知其他线程完成修复
                        with count_lock:
                            count = True
                        validated_result.append({
                            'patch': patch, 'correctness': correctness,
                        })

                    print("Thread {}: {}/{} Plausible patch:{}".format(id, right, cnt, patch))
                    break
                elif correctness == 'wrong':
                    print("Thread-{}: {}/{} Wrong patch:{}".format(id, right, cnt, patch))
                elif correctness == 'timeout':
                    print("Thread-{}: {}/{} Timeout patch:{}".format(id, right, cnt, patch))
            else:
                print("Thread-{}: {}/{} Uncompilable patch:{}".format(id, right, cnt, patch))
            validated_result.append({
                'patch': patch, 'correctness': correctness,
            })
            shutil.copyfile(tmp_dir + "/java_programs_bak/" + proj + '.java',
                            tmp_dir + "/java_programs/" + proj + '.java')# 覆盖修改过的

        with count_lock:
            if count == True:
                return correct_repair, validated_result
    return correct_repair, validated_result


    


    


if __name__ == '__main__':
    reranked_result_path = VALIDATE_QUIXBUGS_DIR + '..\\..\\data\\patches\\reranked_patches_2_quixbugs_only_eva.json'
    output_path = VALIDATE_QUIXBUGS_DIR + '..\\..\\data\\patches\\validated_patches_2_eva.json'
    tmp_dir = VALIDATE_QUIXBUGS_DIR + 'tmp\\validate_quixbugs'
    validate_quixbugs(reranked_result_path, output_path, tmp_dir)
