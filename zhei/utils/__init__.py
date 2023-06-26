def init_env(seed: str='3407', use_deterministic: bool=False,
             visibale_cuda: str='all', 
             comet_exp: dict={},
             proctitle: str='python',
             proctitle_prefix_id: bool=True,
             ) -> None:
    """初始化环境，包括随机种子、可见GPU、Comet.ml环境变量、进程名

    Args:
        seed (str, optional): 种子. Defaults to '3407'.
        use_deterministic (bool, optional): 是否使用确定性算法，使用了训练会变慢. Defaults to False.
        visibale_cuda (str, optional): 设置可见 GPU，如果不使用设置为 ''. Defaults to 'all'.
        comet_exp (dict, optional): Comet.ml相关环境变量设置. Defaults to {}.
        proctitle (str, optional): 进程名. Defaults to 'python'.
        proctitle_prefix_id (bool, optional): 是否在进程名前边添加进程 ID. Defaults to True.
    """
    # ---------------------------------------------------------------------------- #
    #                         设置可见 GPU                                                          
    # ---------------------------------------------------------------------------- #
    import os
    if visibale_cuda != 'all':
        os.environ['CUDA_VISIBLE_DEVICES'] = visibale_cuda
    
    # ---------------------------------------------------------------------------- #
    #                         设置随机种子                                           
    # ---------------------------------------------------------------------------- #
    import random
    import numpy as np
    import torch
    random.seed(seed)   # Python的随机性
    os.environ['PYTHONHASHSEED'] = str(seed)    # 设置Python哈希种子，为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)   # numpy的随机性
    torch.manual_seed(seed)   # torch的CPU随机性，为CPU设置随机种子
    torch.cuda.manual_seed(seed)   # torch的GPU随机性，为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.   torch的GPU随机性，为所有GPU设置随机种子
    torch.backends.cudnn.benchmark = not use_deterministic # if benchmark=True, deterministic will be False
    torch.backends.cudnn.deterministic = use_deterministic   # 选择确定性算法
    
    # ---------------------------------------------------------------------------- #
    #                         设置 Comet.ml 环境变量                                     
    # ---------------------------------------------------------------------------- #
    if comet_exp != {}:
        os.environ["COMET_API_KEY"] = comet_exp.get("comet_api_key", "")
        os.environ["COMET_PROJECT_NAME"] = comet_exp.get("comet_project_name", "收集箱")
        os.environ["COMET_AUTO_LOG_ENV_CPU"] = comet_exp.get("comet_auto_log_env_cpu", "False")
        os.environ["COMET_AUTO_LOG_ENV_GPU"] = comet_exp.get("comet_auto_log_env_gpu", "False")
        os.environ["COMET_AUTO_LOG_ENV_DETAILS"] = comet_exp.get("comet_auto_log_env_details", "False")
        os.environ["COMET_AUTO_LOG_CO2"] = comet_exp.get("comet_auto_log_co2", "False")
        os.environ["COMET_AUTO_LOG_GIT_METADATA"] = comet_exp.get("comet_auto_log_git_metadata", "False")
        os.environ["COMET_AUTO_LOG_GIT_PATCH"] = comet_exp.get("comet_auto_log_git_patch", "False")
        
    # ---------------------------------------------------------------------------- #
    #                         设置其他环境变量                                     
    # ---------------------------------------------------------------------------- #
    os.environ["TOKENIZERS_PARALLELISM"] = "False"
    
    # ---------------------------------------------------------------------------- #
    #                         设置进程名                                     
    # ---------------------------------------------------------------------------- #
    import setproctitle
    if proctitle_prefix_id:
        setproctitle.setproctitle(str(os.getpid()) + "| " + proctitle)
    else:
        setproctitle.setproctitle(proctitle)