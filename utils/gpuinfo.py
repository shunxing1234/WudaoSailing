import os
import pynvml
import torch
pynvml.nvmlInit()



class GpuInfo(object):

    def __init__(self):
        self.gpu_available = torch.cuda.is_available()
        self.gpu_num=0
        self.used_info=[]
        if self.gpu_available:
            self.gpu_num=torch.cuda.device_count()
            self.used_info=self.usegpu(self.gpu_num)
            print(self.used_info)



    def usegpu(self,need_gpu_count=1):
        used_info=[]
        for index in range(min(need_gpu_count,pynvml.nvmlDeviceGetCount())):
            # 这里的index 是GPU id
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
            used= meminfo.used/meminfo.total
            used_info.append([meminfo.used,meminfo.total,used])
        return used_info