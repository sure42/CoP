import threading

# 定义线程锁和成功处理的数据计数器
count_lock = threading.Lock()
count = 0