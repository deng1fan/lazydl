from omegaconf import DictConfig, OmegaConf
import os
import json
from zhei.utils.log import Logger
from typing import Union

log = Logger(__name__) 


def init_env(config: Union[DictConfig, dict]) -> None:
    """初始化环境，包括随机种子、可见GPU、Comet.ml环境变量、进程名

    Args:
        config (DictConfig): 配置文件
            seed (str, optional): 种子. Defaults to '3407'.
            use_deterministic (bool, optional): 是否使用确定性算法，使用了训练会变慢. Defaults to False.
            visibale_cuda (str, optional): 设置可见 GPU，如果不使用设置为 ''. Defaults to 'all'.
            comet_exp (dict, optional): Comet.ml相关环境变量设置. Defaults to {}.
            proctitle (str, optional): 进程名. Defaults to 'python'.
            proctitle_prefix_id (bool, optional): 是否在进程名前边添加进程 ID. Defaults to True.
    """
    if isinstance(config, dict):
        config = OmegaConf.create(config)
    OmegaConf.set_struct(config, False)
    env = config.get("env", {})
    updated_env = {}
    # ---------------------------------------------------------------------------- #
    #                         设置可见 GPU                                                          
    # ---------------------------------------------------------------------------- #
    if env.get("env.visibale_cuda", "all") != "all":
        os.environ['CUDA_VISIBLE_DEVICES'] = env.get("visibale_cuda")
        updated_env['CUDA_VISIBLE_DEVICES'] = os.environ['CUDA_VISIBLE_DEVICES']
    
    # ---------------------------------------------------------------------------- #
    #                         设置随机种子                                           
    # ---------------------------------------------------------------------------- #
    seed = env.get("seed", 3407)
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
    os.environ["COMET_API_KEY"] = env.get("comet_api_key", "")
    os.environ["COMET_PROJECT_NAME"] = env.get("comet_project_name", "Default Project")
    os.environ["COMET_AUTO_LOG_ENV_CPU"] = env.get("comet_auto_log_env_cpu", "False")
    os.environ["COMET_AUTO_LOG_ENV_GPU"] = env.get("comet_auto_log_env_gpu", "False")
    os.environ["COMET_AUTO_LOG_ENV_DETAILS"] = env.get("comet_auto_log_env_details", "False")
    os.environ["COMET_AUTO_LOG_CO2"] = env.get("comet_auto_log_co2", "False")
    os.environ["COMET_AUTO_LOG_GIT_METADATA"] = env.get("comet_auto_log_git_metadata", "False")
    os.environ["COMET_AUTO_LOG_GIT_PATCH"] = env.get("comet_auto_log_git_patch", "False")
    
    updated_env['COMET_API_KEY'] = os.environ['COMET_API_KEY']
    updated_env['COMET_PROJECT_NAME'] = os.environ['COMET_PROJECT_NAME']
    updated_env['COMET_AUTO_LOG_ENV_CPU'] = os.environ['COMET_AUTO_LOG_ENV_CPU']
    updated_env['COMET_AUTO_LOG_ENV_GPU'] = os.environ['COMET_AUTO_LOG_ENV_GPU']
    updated_env['COMET_AUTO_LOG_ENV_DETAILS'] = os.environ['COMET_AUTO_LOG_ENV_DETAILS']
    updated_env['COMET_AUTO_LOG_CO2'] = os.environ['COMET_AUTO_LOG_CO2']
    updated_env['COMET_AUTO_LOG_GIT_METADATA'] = os.environ['COMET_AUTO_LOG_GIT_METADATA']
    updated_env['COMET_AUTO_LOG_GIT_PATCH'] = os.environ['COMET_AUTO_LOG_GIT_PATCH']
    
    
    # ---------------------------------------------------------------------------- #
    #                         设置钉钉 Token                                     
    # ---------------------------------------------------------------------------- #
    os.environ["DINGDING_ACCESS_TOKEN"] = env.get("dingding_access_token", "")
    os.environ["DINGDING_SECRET"] = env.get("dingding_secret", "")
    
    updated_env['DINGDING_ACCESS_TOKEN'] = os.environ['DINGDING_ACCESS_TOKEN']
    updated_env['DINGDING_SECRET'] = os.environ['DINGDING_SECRET']
    
    
    # ---------------------------------------------------------------------------- #
    #                         设置任务相关环境变量                                     
    # ---------------------------------------------------------------------------- #
    os.environ["TASK_DESC"] = config.get("task_desc", "No task description.")
    os.environ["TASK_ID"] = config.get("task_id", "No task identifier.")
    updated_env['TASK_DESC'] = os.environ['TASK_DESC']
    updated_env['TASK_ID'] = os.environ['TASK_ID']
        
    # ---------------------------------------------------------------------------- #
    #                         设置其他环境变量                                     
    # ---------------------------------------------------------------------------- #
    os.environ["TOKENIZERS_PARALLELISM"] = env.get("tokenizers_parallelism", "False")
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
            if processing_unit >= torch.cuda.device_count():
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
    log.info(f"Processing unit: {config.get('processing_unit')}")
    
    return config