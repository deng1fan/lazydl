from zhei.utils.log import Logger
import os
import pickle
import json
import jsonlines
import torch
import pandas as pd

log = Logger(__name__) 


def load_in(path, data_name=""):
    """读取文件，根据文件后缀名自动选择读取方法
        目前支持保存类型有：‘pkl’、‘txt’、‘pt’、‘json’, 'jsonl'、'csv'

    Args:
        data_name: str, 打印提示时需要，便于控制台查看保存的文件是什么文件, 默认为空

    Returns:
        data：Object
    """
    if not os.path.exists(path):
        log.info(f"文件路径似乎并不存在....")
        raise FileNotFoundError(path)
    log.info(f"正在加载文件 {data_name} from {path}")
    if ".pkl" in path:
        with open(path, "rb") as f:
            data = pickle.load(f)
    elif ".json" in path:
        with open(path, "r") as f:
            # 读取json数据
            data = json.load(f)
    elif ".jsonl" in path:
        data = []
        with open(path, "rb") as f:
            for item in jsonlines.Reader(f):  # 每一行读取后都是一个json，可以按照key去取对应的值
                data.append(item)
    elif ".pt" in path:
        data = torch.load(path)
    elif ".txt" in path:
        data = []
        for item in open(path):
            item = item.replace("\n", "").strip()
            if item == "":
                continue
            data.append(item)
    elif ".csv" in path:
        data = pd.read_csv(path)
    log.info(f"成功加载 {data_name}!")
    return data


def save_as(data, save_path, file_format="pt", data_name="", protocol=4):
    """将参数中的文件对象保存为指定格式格式文件
        目前支持保存类型有：‘pkl’、‘txt’、‘pt’、‘json’, 'jsonl'
        默认为‘pt’

    Args:
        data: obj, 要保存的文件对象
        save_path: str, 文件的保存路径，应当包含文件名和后缀名
        data_name: str, 打印提示时需要，便于控制台查看保存的文件是什么文件, 默认为空
        protocol: int, 当文件特别大的时候，需要将此参数调到4以上, 默认为4
        file_format: str, 要保存的文件类型，支持‘pkl’、‘txt’、‘pt’、‘json’、‘jsonl’

    Returns:
        None
    """
    parent_path = "/".join(save_path.split("/")[:-1])
    if not os.path.exists(parent_path):
        log.info(f"保存路径的父文件夹（{parent_path}）不存在，将自动创建....")
        os.makedirs(parent_path)
    save_path = save_path + f".{file_format}"
    log.info(f"正在保存文件 {data_name} 到 {save_path}")
    if file_format == "pkl":
        with open(save_path, "wb") as f:
            pickle.dump(data, f, protocol=protocol)
    elif file_format == "txt":
        if not isinstance(data, list):
            data = [data]
        with open(save_path, "w") as f:
            for line in data:
                f.write(str(line) + "\n")
    elif file_format == "json":
        with open(save_path, "w") as f:
            json.dump(data, f)
    elif file_format == "jsonl":
        with jsonlines.open(save_path, mode="w") as writer:
            writer.write_all(data)
    elif file_format == "pt":
        torch.save(data, save_path)
    else:
        raise Exception(f"请添加针对{file_format}类型文件的保存方法！")
    log.info(f"保存 {data_name} 成功!")
    return None
