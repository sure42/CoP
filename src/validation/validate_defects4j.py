import json
import os
import shutil
import subprocess
import time
import sys


# VALIDATE_DEFECTS4J_DIR = os.path.abspath(__file__)[: os.path.abspath(__file__).rindex('/') + 1] # linux
VALIDATE_DEFECTS4J_DIR = os.path.abspath(__file__)[: os.path.abspath(__file__).rindex('\\') + 1] # win
sys.path.append(VALIDATE_DEFECTS4J_DIR + '../dataloader/')
git_bash_path = "C:\Program Files\Git\git-bash.exe"
import tokenization
import chardet
import send2trash

from count_module import count, count_lock

def clean_tmp_folder(tmp_dir):# 清空tmp/defect4j内的内容
    if os.path.isdir(tmp_dir):
        for files in os.listdir(tmp_dir):
            file_p = os.path.join(tmp_dir, files)
            try:
                if os.path.isfile(file_p):
                    os.unlink(file_p)# 用于删除文件，如果文件是一个目录则返回一个错误
                elif os.path.isdir(file_p):
                    shutil.rmtree(file_p)# 递归地删除文件
                    # send2trash.send2trash(file_p)
            except Exception as e:
                print(e)
    else:
        os.makedirs(tmp_dir)


def checkout_defects4j_project(project, bug_id, tmp_dir):
    FNULL = open(os.devnull, 'w')
    # tmp_dir = tmp_dir.replace('\\','/')
    command = "defects4j checkout " + " -p " + project + " -v " + bug_id + " -w " + tmp_dir
    p = subprocess.Popen([command], shell=True, stdout=FNULL, stderr=FNULL)
    p.wait()
    # out, err = p.communicate()
    # 将字节流解码为字符串
    # if out != b'':
    #     result = chardet.detect(err)
    #     out = out.decode(chardet.detect(out)['encoding'])
    # if err != b'':
    #     err = err.decode(chardet.detect(err)['encoding'])  
    # return out, err


def compile_fix(project_dir):
    os.chdir(project_dir)
    p = subprocess.Popen(["defects4j", "compile"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)# 可编译？
    out, err = p.communicate()
    if "FAIL" in str(err) or "FAIL" in str(out):
        return False
    return True


def command_with_timeout(cmd, timeout=30000):
    # 来产生子进程,并连接到子进程的标准输入/输出/错误中去
    # stdin stdout stderr分别表示程序的标准输入、输出、错误句柄,他们可以设置成文件对象，文件描述符或者PIPE和None。None--没有任何重定向，继承父进程；PIPE--创建管道
    # universal_newlines：windows下文本换行符用’\r\n’，而Linux下用 ‘\n’。若设置为True，Python统一把这些换行符当作’\n’来处理。
    # cmd = ['cmd.exe', '/c'] + cmd
    # cmd_A.extend(cmd)
    # cmd = [git_bash_path] + ['-c'] + cmd
    p = subprocess.Popen(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE, universal_newlines=True)
    # p = subprocess.Popen(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE, universal_newlines=True)
    t_beginning = time.time()
    while True:
        if p.poll() is not None:# 用于检查子进程是否已经结束，返回returncode。
            break
        seconds_passed = time.time() - t_beginning
        if timeout and seconds_passed > timeout:
            p.terminate()# 停止(stop)子进程。在windows平台下，该方法将调用Windows API TerminateProcess（）来结束子进程
            return 'TIMEOUT', 'TIMEOUT'
        time.sleep(1)
    # 输出编码
    out, err = p.communicate()
    # # 将字节流解码为字符串
    # if out == b'':
    #     result = chardet.detect(err)
    # if err == b'':
    #     result = chardet.detect(out)
    # encoding = result['encoding']
    # if out != b'':
    #     out = out.decode(encoding)
    # if err != b'':
    #     err = err.decode(encoding)  
    return out, err


def defects4j_test_suite(project_dir, timeout=300):
    os.chdir(project_dir)
    out, err = command_with_timeout(["defects4j", "test", "-r"], timeout)
    return out, err


def defects4j_trigger(project_dir, timeout=300):
    os.chdir(project_dir)
    out, err = command_with_timeout(["defects4j", "export", "-p", "tests.trigger"], timeout)
    return out, err


def defects4j_relevant(project_dir, timeout=300):
    os.chdir(project_dir)
    out, err = command_with_timeout(["defects4j", "export", "-p", "tests.relevant"], timeout)
    return out, err


def defects4j_test_one(project_dir, test_case, timeout=300):
    os.chdir(project_dir)
    out, err = command_with_timeout(["defects4j", "test", "-t", test_case], timeout)
    return out, err


def insert_fix_defects4j(file_path, start_loc, end_loc, patch, project_dir):
    file_path = project_dir + '/' + file_path
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
            strings, numbers = tokenization.get_strings_numbers(line)
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


def validate_defects4j(id, tokenized_patchs, tmp_dir, proj, bug_id, path, start_loc, end_loc, cnt, right):

                    # (id, tokenized_patchs, tmp_dir, proj, start_loc, end_loc, cnt, right):

    # cnt, right
    # 要验证的补丁
    # 现在的漏洞 current_bug = proj + '_' + bug_id
    # key划分后的结果 

    global count, count_lock

    if not os.path.exists(tmp_dir):
        command_with_timeout(['mkdir', tmp_dir])
        # os.makedirs(tmp_dir, exist_ok=True)

    validated_result = []
    last_bug = None

    current_bug = proj + '_' + bug_id
    current_is_correct = False
    if last_bug != current_bug:
        last_bug = current_bug
        # checkout project
        clean_tmp_folder(tmp_dir)
        # init_defects4j_project(tmp_dir)
        checkout_defects4j_project(proj, bug_id + 'b', tmp_dir)# checkout--下载项目 签出buggy或固定项目版本
        if proj == "Mockito":
            print("Thread {}: Mockito needs separate compilation".format(id))
            compile_fix(tmp_dir)# ["defects4j", "compile"]

        # check standard test time
        start_time = time.time()
        init_out, init_err = defects4j_test_suite(tmp_dir)# ["defects4j", "test", "-r"] # test--在buggy或固定项目版本上运行单个测试方法或测试套件
        standard_time = int(time.time() - start_time)

        # check failed test cases
        failed_test_cases = str(init_out).split(' - ')[1:]

        for i, failed_test_case in enumerate(failed_test_cases):
            failed_test_cases[i] = failed_test_case.strip()# 删除前导和尾随空格的字符串副本
        init_fail_num = len(failed_test_cases)
        print("Thread {}: init fail number:{} {}s".format(id, init_fail_num, str(standard_time)))

        trigger, err = defects4j_trigger(tmp_dir)# ["defects4j", "export", "-p", "tests.trigger"] # 导出version-specific属性，例如类路径、目录或测试列表
        triggers = trigger.strip().split('\n')
        for i, trigger in enumerate(triggers):
            triggers[i] = trigger.strip()
        print("Thread {}: trigger number:{}".format(id, triggers))

        relevant, err = defects4j_relevant(tmp_dir)# ["defects4j", "export", "-p", "tests.relevant"]
        relevants = relevant.strip().split('\n')
        for i, relevant in enumerate(relevants):
            relevants[i] = relevant.strip()
        print("Thread {}: relevant number:{}".format(id, relevant))

    bug_start_time = time.time()
    for tokenized_patch in tokenized_patchs:
        # validate 5 hours for each bug at most 每次最多五小时
        if time.time() - bug_start_time > 5 * 3600:
            break
        # validate 5000 patches for each bug at most 每次最多5000条
        if len(validated_result) >= 5000:
            break

        score = tokenized_patch['score']
        tokenized_patch = tokenized_patch['patch']

        strings, numbers = get_strings_numbers(tmp_dir + '/'+ path, (int(start_loc) + int(end_loc)) // 2)
        strings = [item[0] for item in strings][:5]
        numbers = [item[0] for item in numbers][:5]
        # one tokenized patch may be reconstructed to multiple source-code patches
        reconstructed_patches = tokenization.token2statement(tokenized_patch.split(' '), numbers, strings)# 看下效果
        # validate most 5 source-code patches come from the same tokenized patch
        for patch in reconstructed_patches[:5]:
            patch = patch.strip()

            patched_file = insert_fix_defects4j(path, int(start_loc), int(end_loc), patch, tmp_dir)# 这里是把补丁进行替换
            if proj == 'Mockito':
                # Mockito needs seperate compile
                compile_fix(tmp_dir)

            # trigger cases is few and total time is long, we test trigger cases first.
            outs = []
            correctness = None
            start_time = time.time()
            if standard_time >= 10 and len(triggers) <= 5:
                for trigger in triggers:
                    out, err = defects4j_test_one(tmp_dir, trigger)
                    if 'TIMEOUT' in str(err) or 'TIMEOUT' in str(out):
                        print("Thread {}: {}/{} {} Time out for patch:{} {}s".format(id, right, cnt, current_bug,  patch, str(int(time.time() - start_time))))
                        correctness = 'timeout'
                        break
                    elif 'FAIL' in str(err) or 'FAIL' in str(out):
                        print("Thread {}: {}/{} {} Uncompilable patch:{} {}s".format(id, right, cnt, current_bug,  patch, str(int(time.time() - start_time))))
                        correctness = 'uncompilable'
                        break
                    elif "Failing tests: 0" in str(out):
                        continue
                    else:
                        outs += str(out).split(' - ')[1:]
            if len(set(outs)) >= len(triggers):# 这里指的是添加补丁后的跟之前比，有一些测试用例无法通过
                # does not pass any one more
                print("Thread {}: {}/{} {} Wrong patch:{} {}s".format(id, right, cnt, current_bug,  patch, str(int(time.time() - start_time))))
                correctness = 'wrong'

            # tonguo le daduoshu qie haoshiduan
            if correctness is None:
                # pass at least one more trigger case
                # have to pass all non-trigger
                out, err = defects4j_test_suite(tmp_dir)

                if 'TIMEOUT' in str(err) or 'TIMEOUT' in str(out):
                    print("Thread {}: {}/{} {} Time out for patch:{} {}s".format(id, right, cnt, current_bug,  patch, str(int(time.time() - start_time))))
                    correctness = 'timeout'
                elif 'FAIL' in str(err) or 'FAIL' in str(out):
                    print("Thread {}: {}/{} {} Uncompilable patch:{} {}s".format(id, right, cnt, current_bug,  patch, str(int(time.time() - start_time))))
                    correctness = 'uncompilable'
                elif "Failing tests: 0" in str(out):
                    if not current_is_correct:
                        current_is_correct = True
                        right += 1
                    print("Thread {}: {}/{} {} Plausible patch:{} {}s".format(id, right, cnt, current_bug,  patch, str(int(time.time() - start_time))))
                    correctness = 'plausible'
                elif len(str(out).split(' - ')[1:]) < init_fail_num:
                    # fail less, could be correct
                    current_failed_test_cases = str(out).split(' - ')[1:]
                    no_new_fail = True
                    for current_failed_test_case in current_failed_test_cases:
                        if current_failed_test_case.strip() not in failed_test_cases:# 发生了新的错误
                            no_new_fail = False
                            break
                    if no_new_fail:
                        # fail less and no new fail cases, could be plausible
                        if not current_is_correct:
                            current_is_correct = True
                            right += 1
                        print("Thread {}: {}/{} {} Plausible patch:{} {}s".format(id, right, cnt, current_bug,  patch, str(int(time.time() - start_time))))
                        correctness = 'plausible'
                    else:
                        print("Thread {}: {}/{} {} Wrong patch:{} {}s".format(id, right, cnt, current_bug,  patch, str(int(time.time() - start_time))))
                        correctness = 'wrong'
                else:
                    print("Thread {}: {}/{} {} Wrong patch:{} {}s".format(id, right, cnt, current_bug,  patch, str(int(time.time() - start_time))))
                    correctness = 'wrong'

            validated_result.append({
                            'patch': patch, 'score': score, 'correctness': correctness,
                        })
            
            if current_is_correct:
                # 通知其他线程完成修复
                with count_lock:
                    count = True
                break
            shutil.copyfile(patched_file, patched_file.replace('.bak', ''))


        with count_lock:
            if count == True:
                return current_is_correct, validated_result

    return current_is_correct, validated_result


if __name__ == '__main__':

    model_ids = 1
    reranked_result_path = VALIDATE_DEFECTS4J_DIR + '../../data/patches/reranked_patches_{}_defects4jv1.2_parse.json'.format(model_ids)
    output_path = VALIDATE_DEFECTS4J_DIR + '../../data/patches/validated_patches_{}_defects4jv1.2_parse.json'.format(model_ids)
    # tmp_dir = '\\tmp\\validate_d4j'
    tmp_dir = VALIDATE_DEFECTS4J_DIR + 'tmp/validate_d4j/'
    validate_defects4j(reranked_result_path, output_path, tmp_dir)
