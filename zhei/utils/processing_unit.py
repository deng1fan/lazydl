from nvitop import select_devices, Device
import sys
from zhei.utils.redis import RedisClient
from zhei.utils.notice import notice
from zhei.utils.log import Logger
import time
import os
import torch
from typing import Union
from omegaconf import DictConfig

log = Logger(__name__) 

def set_processing_units(config: Union[DictConfig, dict] = {}):
    """设置处理器，支持自动选择和手动选择、排队和不排队

    Args:
        config (dict, optional): Defaults to {}.
            processing_unit: 处理器类型，可选值为：cpu、mps, 1, ^2 等，其中 ^2 表示自动选择两块 GPU，默认为 cpu
            processing_unit_type: 选择处理器，可选值为：cpu、mps、gpu-a-s、gpu-a-m、gpu-m-s、gpu-m-m
            processing_unit_min_free_memory: 最小空闲内存，单位为 GiB，默认为 10
            processing_unit_min_free_memory_ratio: 最小空闲内存比例，默认为 0.5
            queuing: 是否排队，依赖 Redis，默认为 False
            visible_devices: 可见的 GPU 序号，用逗号分隔，默认为 None

    Returns:
        Union[DictConfig, dict]: config
    """
    processing_unit = config.get("processing_unit", "cpu")
    processing_unit_type = config.get("processing_unit_type", "cpu")
    if processing_unit in ["cpu", "mps"]:
        # CPU 或 MPS 模式，直接返回
        return config
    
    # ---------------------------------------------------------------------------- #
    #                         获取需要的处理器数量                                     
    # ---------------------------------------------------------------------------- #
    min_count = 1
    if processing_unit_type in ["gpu-a-s", "gpu-m-s"]:
        min_count = 1
    elif processing_unit_type == "gpu-a-m":
        min_count = processing_unit
    else:
        min_count = len(processing_unit)
    
    queuing = config.get("queuing", False)
    if not queuing and processing_unit_type in ["gpu-m-s", "gpu-m-m"]:
        # 不排队，且手动选择 GPU，直接返回，无需更新
        log.warning("Using manual GPU selection, no need to update processing unit.")
        log.warning("To avoid this warning, set queuing (requirements redis) to True.")
        return config
    
    # ---------------------------------------------------------------------------- #
    #                         获取当前符合条件的所有处理器                                     
    # ---------------------------------------------------------------------------- #
    
    devices = Device.all()
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", None) if config.get("visible_devices", None) is None else config.get("visible_devices")
    if visible_devices is not None:
        devices = [Device(index=device_id) for device_id in visible_devices.split(",")]
    
    processing_units = select_devices(
            devices=devices,
            format="index",
            min_count=min_count,
            min_free_memory=config.get("processing_unit_min_free_memory", "10") +  + "GiB",
            max_memory_utilization=config.get("processing_unit_min_free_memory_ratio", "0.5"),
    )
    
    task_id, wait_num = None, None
    redis_client = RedisClient()
    # ---------------------------------------------------------------------------- #
    #                         如果需要排队就送入队列                                     
    # ---------------------------------------------------------------------------- #
    if not is_processing_units_ready(processing_unit_type, processing_unit, processing_units, min_count):
        if queuing:
            task_id, wait_num = redis_client.join_wait_queue(processing_unit_type, processing_unit, units_count=min_count)
        else:
            log.error("No enough processing units are available, please wait for a moment or turn on queuing mode.")
            notice("No enough processing units are available, please wait for a moment or turn on queuing mode.")
            raise Exception("No enough processing units are available, please wait for a moment or turn on queuing mode.")
    else:
        config.processing_unit = update_processing_unit(processing_unit_type, processing_unit, processing_units, min_count)
        log.info(f"Updated processing unit: {config.get('processing_unit')}")
        return config
        
    # ---------------------------------------------------------------------------- #
    #                         排队模式，等待处理器                                     
    # ---------------------------------------------------------------------------- #  
    while not redis_client.is_my_turn(task_id) or not is_processing_units_ready(processing_unit_type, processing_unit, processing_units, min_count):
        sys.sleep(30)
        curr_time = str(time.strftime('%m月%d日 %H:%M:%S', time.localtime()))
        if redis_client.is_my_turn(task_id):
            # 更新队列
            redis_client.update_queue(task_id)
            
        wait_num = len(redis_client.client.lrange("wait_queue", 0, -1)) - 1
        print(f"\rcurr_time: {curr_time} | wait_num: {wait_num} | processing_unit_type: {processing_unit_type} | processing_unit: {processing_unit}", end='',  flush=True)
            
        processing_units = select_devices(
            format="index",
            min_count=min_count,
            min_free_memory=config.get("processing_unit_min_free_memory", "10") + "GiB",
            max_memory_utilization=config.get("processing_unit_min_free_memory_ratio", "0.5"),
        )
    
    # ---------------------------------------------------------------------------- #
    #                         从队列中弹出并注册处理器和进程                              
    # ---------------------------------------------------------------------------- #
    redis_client.pop_wait_queue(task_id)
    redis_client.register_gpus(task_id, processing_unit_type, processing_unit, units_count=min_count)
    redis_client.register_process(task_id, processing_unit_type, processing_unit, units_count=min_count)
    
    # ---------------------------------------------------------------------------- #
    #                         更新可用处理器                                     
    # ---------------------------------------------------------------------------- #
    config.processing_unit = update_processing_unit(processing_unit_type, processing_unit, processing_units, min_count)
    log.info(f"Updated processing unit: {config.get('processing_unit')}")
        
    return config


def is_processing_units_ready(processing_unit_type, processing_unit, processing_units, min_count):
    if processing_unit_type in ["gpu-m-s", "gpu-m-m"]:
        if len(set(processing_unit) - set(processing_units)) > 0:
            # 有处理器不在 processing_units 中
            return False
        else:
            return True 
    else:
        if len(processing_units) < min_count:
            # 没有符合条件的处理器
            return False
        else:
            return True
        
        
def update_processing_unit(processing_unit_type, processing_unit, processing_units, min_count):
    if processing_unit_type in ["gpu-m-s", "gpu-m-m"]:
        return processing_unit
    else:
        return processing_units[:min_count]
    
    
def gpu_ready():
    """判断 GPU 是否可用

    Returns:
        bool: True 表示 GPU 可用，False 表示 GPU 不可用
    """
    return torch.cuda.is_available() and torch.cuda.device_count() > 0
    
