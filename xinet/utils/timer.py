import time

from .array import NumPyArrayLike

class Timer(NumPyArrayLike):
    """记录多次运行时间
    """

    def __init__(self, data: list = None, dtype=None):
        super().__init__(data, dtype)
        self.start()

    def start(self):
        """启动计时器。"""
        self.tik = time.time()

    def stop(self):
        """停止计时器并将时间记录在列表中"""
        t = time.time() - self.tik
        self.append(t)
        return t

    def avg(self):
        """返回平均时间"""
        # sum(self): 返回时间总和
        return sum(self) / len(self)


