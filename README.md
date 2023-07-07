# 安装
```bash
pip install zhei
```


# 使用
```python
import zhei as j

j.hi()
```


# 功能支持

### 帮助信息
- hi()：打印 Banner
- help()：打印帮助信息，待完善
- print_error_info(e: Exception)：打印错误信息


### 通知
- notice(msg, warning=False, access_token="", secret="")：钉钉通知，默认读取 init_env() 设置的环境变量中的凭证
    ```
    Args:
        msg (str): 通知内容
        warning (bool, optional): 是否为警告. Defaults to False.
        access_token (str, optional): 钉钉机器人 access_token. Defaults to "".
        secret (str, optional): 钉钉机器人 secret. Defaults to "".
    ```

### 配置
- init_env(config: Union[DictConfig, dict])：初始化环境
    ```
    包括随机种子、可见GPU、Comet.ml环境变量、进程名

    Args:
        config (DictConfig): 配置文件
            seed (str, optional): 种子. Defaults to '3407'.
            use_deterministic (bool, optional): 是否使用确定性算法，使用了训练会变慢. Defaults to False.
            visibale_cuda (str, optional): 设置可见 GPU，如果不使用设置为 ''. Defaults to 'all'.
            comet_exp (dict, optional): Comet.ml相关环境变量设置. Defaults to {}.
            proctitle (str, optional): 进程名. Defaults to 'python'.
            proctitle_prefix_id (bool, optional): 是否在进程名前边添加进程 ID. Defaults to True.
    ```

### 处理器
- gpu_ready())：检查 GPU 是否可用
- set_processing_units(config: Union[DictConfig, dict] = {})：设置处理器
    ```
    支持自动选择和手动选择、排队和不排队

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
    ```

### 包装类
- Result：包装类，用于作为模块间通信的媒介，可使用字典进行初始化


### 文件读取和保存
- load_in(path, data_name="")：读取文件
    ```
    根据文件后缀名自动选择读取方法
        目前支持保存类型有：‘pkl’、‘txt’、‘pt’、‘json’, 'jsonl'、'csv'

    Args:
        data_name: str, 打印提示时需要，便于控制台查看保存的文件是什么文件, 默认为空

    Returns:
        data：Object
    ```

- save_as(data, save_path, file_format="pt", data_name="", protocol=4)：保存文件
    ```
    将参数中的文件对象保存为指定格式格式文件
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
    ```

