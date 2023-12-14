from collections.abc import Callable, Iterable, Mapping
import time 
import threading
from typing import Any

class SonThread(threading.Thread):
    def __init__(self, n, parent):
        self.n = n
        self.parent = parent
        super().__init__()
    
    def run(self) -> None:
        print("Sonthread", self.n, " parent::", self.parent)

class MyThread(threading.Thread):
    def __init__(self, n ,size):
        self.n = n
        self.size = size
        super().__init__()
    
    def run(self) -> None:
        for i in range(self.size):
            t = SonThread(i, self.n)
            t.start()           
            print("No", self.n, " :", i)


for i in range(3):
    t = MyThread(i, i+5)
    t.start()