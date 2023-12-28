import comet_ml
from omegaconf import DictConfig, OmegaConf
import os
import json
import sys
from lazydl.utils.log import Logger
from typing import Union
import datetime
import numpy

log = Logger(__name__) 


def init_env(config: Union[DictConfig, dict], current_dir: str = "./") -> None:
    """初始化环境，包括随机种子、可见GPU、Comet.ml环境变量、进程名

    Args:
        config (DictConfig, dict): 配置文件
            seed (str, optional): 种子. Defaults to '3407'.
            use_deterministic (bool, optional): 是否使用确定性算法，使用了训练会变慢. Defaults to False.
            visibale_cuda (str, optional): 设置可见 GPU，如果不使用设置为 ''. Defaults to 'all'.
            proctitle (str, optional): 进程名. Defaults to 'python'.
            proctitle_prefix_id (bool, optional): 是否在进程名前边添加进程 ID. Defaults to True.
    """
    if isinstance(config, dict):
        config = OmegaConf.create(config)
    OmegaConf.set_struct(config, False)
    updated_env = {}
    
    config.root_dir = current_dir
    if config.get("stage", "debug") == "debug":
        log.info("Debug mode will set fast_dev_run to True")
        config.fast_dev_run = True
    
    # ---------------------------------------------------------------------------- #
    #                         生成 Task ID                                     
    # ---------------------------------------------------------------------------- #
    task_id_prefix = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    config.task_id = "{}{}{}".format(task_id_prefix, ''.join([str(num) for num in numpy.random.randint(0, 9, size=18)]), os.getpid())
    
    
    # ---------------------------------------------------------------------------- #
    #                         设置可见 GPU                                                          
    # ---------------------------------------------------------------------------- #
    if config.get("env.visibale_cuda", "all") != "all":
        os.environ['CUDA_VISIBLE_DEVICES'] = config.get("visibale_cuda")
        updated_env['CUDA_VISIBLE_DEVICES'] = os.environ['CUDA_VISIBLE_DEVICES']
    
    # ---------------------------------------------------------------------------- #
    #                         设置随机种子                                           
    # ---------------------------------------------------------------------------- #
    seed = config.get("seed", 3407)
    use_deterministic = config.get("use_deterministic", False)
    import random
    import numpy as np
    import torch
    random.seed(seed)   # Python的随机性
    os.environ['PYTHONHASHSEED'] = str(seed)    # 设置Python哈希种子，为了禁止hash随机化，使得实验可复现
    updated_env['PYTHONHASHSEED'] = os.environ['PYTHONHASHSEED']
    np.random.seed(seed)   # numpy的随机性
    torch.manual_seed(seed)   # torch的CPU随机性，为CPU设置随机种子
    torch.cuda.manual_seed(seed)   # torch的GPU随机性，为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.   torch的GPU随机性，为所有GPU设置随机种子
    torch.backends.cudnn.benchmark = not use_deterministic # if benchmark=True, deterministic will be False
    torch.backends.cudnn.deterministic = use_deterministic   # 选择确定性算法
    
    # ---------------------------------------------------------------------------- #
    #                         设置 Comet.ml 环境变量                                     
    # ---------------------------------------------------------------------------- #
    os.environ["COMET_API_KEY"] = "" if config.get("comet_api_key") is None else config.get("comet_api_key")
    
    updated_env['COMET_API_KEY'] = os.environ['COMET_API_KEY']
    
    # ---------------------------------------------------------------------------- #
    #                         设置 Comet Project                                     
    # ---------------------------------------------------------------------------- #
    experiment = None
    if config.get("start_comet_log", "False"):
        api = comet_ml.api.API()
        experiment = api.get_experiment_by_key(config.get("task_id"))
        
        if experiment is None:
            experiment = comet_ml.Experiment(project_name=config.get("comet_project_name", "no-comet-project"),
                                            experiment_key=config.get("task_id"),)
            experiment.set_name(config.get("comet_exp_name", "NULL") + "_" + config.get("stage", "debug"))
            
            experiment_config = sys.argv[-1].replace("+experiments=", "")

        
        tmux_session = "/"
        for arg in sys.argv:
            if "tmux_session" in arg:
                tmux_session = arg.replace("+tmux_session=", "")
                
        experiment.log_other("tmux_session", tmux_session)
        experiment.log_other("进程ID", str(os.getpid()))
        experiment.log_other("实验配置文件", experiment_config)
        for key, value in config.items():
            if isinstance(value, dict):
                value = json.dumps(value)
            experiment.log_other(key, str(value))
        for tag in config.get("comet_tags", []):
            experiment.add_tag(tag)
        log.info("Comet 实验记录已启动")
    
    # ---------------------------------------------------------------------------- #
    #                         设置钉钉 Token                                     
    # ---------------------------------------------------------------------------- #
    os.environ["DINGDING_ACCESS_TOKEN"] = config.get("dingding_access_token") if config.get("dingding_access_token") else "NONE_DINGDING_ACCESS_TOKEN"
    os.environ["DINGDING_SECRET"] = config.get("dingding_secret") if config.get("dingding_secret") else "NONE_DINGDING_SECRET"
    
    updated_env['DINGDING_ACCESS_TOKEN'] = os.environ['DINGDING_ACCESS_TOKEN']
    updated_env['DINGDING_SECRET'] = os.environ['DINGDING_SECRET']
    
        
    # ---------------------------------------------------------------------------- #
    #                         设置其他环境变量                                     
    # ---------------------------------------------------------------------------- #
    os.environ["TOKENIZERS_PARALLELISM"] = config.get("tokenizers_parallelism", "False")
    updated_env['TOKENIZERS_PARALLELISM'] = os.environ['TOKENIZERS_PARALLELISM']
    
    
    # ---------------------------------------------------------------------------- #
    #                         设置进程名                                     
    # ---------------------------------------------------------------------------- #
    import setproctitle
    if config.get("proctitle_prefix_id", True):
        setproctitle.setproctitle(str(os.getpid()) + "| " + config.get("proctitle", "python"))
    else:
        setproctitle.setproctitle(config.get("proctitle", "python"))
        
    # ---------------------------------------------------------------------------- #
    #                        打印环境变量                                      
    # ---------------------------------------------------------------------------- #
    log.info(json.dumps(updated_env, indent=4, ensure_ascii=False).replace("{", "Environment Variables\n").replace("}", ""))
    
    # ---------------------------------------------------------------------------- #
    #                         设置处理器类型                                     
    # ---------------------------------------------------------------------------- #
    processing_unit = config.get("processing_unit", "cpu")
    if processing_unit == "cpu":
        config.processing_unit_type = "cpu"
    elif processing_unit == "mps":
        config.processing_unit_type = "mps"
    else:
        if not torch.cuda.is_available():
            log.error("CUDA is not available, please check your CUDA installation.")
            raise Exception("CUDA is not available, please check your CUDA installation.")
        if "^" in processing_unit:
            # 自动选择 GPU
            processing_unit = int(processing_unit.replace("^", ""))
            if processing_unit > torch.cuda.device_count():
                log.error("The number of GPUs you specified is greater than the number of GPUs available.")
                raise Exception("The number of GPUs you specified is greater than the number of GPUs available.")
            if processing_unit > 1:
                config.processing_unit_type = "gpu-a-m"
            else:
                config.processing_unit_type = "gpu-a-s"
        else:
            processing_unit = processing_unit.split(",")
            if len(processing_unit) > torch.cuda.device_count():
                log.error("The number of GPUs you specified is greater than the number of GPUs available.")
                raise Exception("The number of GPUs you specified is greater than the number of GPUs available.")
            if len(processing_unit) > 1:
                config.processing_unit_type = "gpu-m-m"
            else:
                config.processing_unit_type = "gpu-m-s"
        config.processing_unit = processing_unit
        
    log.info(f"Processing unit type: {config.get('processing_unit_type')}")
    log.info(f"Processing unit num: {config.get('processing_unit')}\n")
    
    return config, experiment

