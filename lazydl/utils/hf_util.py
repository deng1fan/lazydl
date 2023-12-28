from huggingface_hub import snapshot_download
from peft import PeftModel, LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, BitsAndBytesConfig
import torch
import bitsandbytes as bnb
from collections import defaultdict
from modelscope import snapshot_download as snapshot_download_ms
from lazydl.utils.log import Logger
from lazydl.utils.result import Result
from typing import Union

logger = Logger(__name__)


def download_model(model_name, cache_dir=None, endpoint='https://hf-mirror.com', use_modelscope=False):
    """下载 HuggingFace 模型权重，支持断点续传，适用于无法连接 HuggingFace 服务器的情况
 
    Args:
        model_name (_type_): 模型名称
        cache_dir (_type_, optional): 缓存地址，默认为 ~/.cache/huggingface. Defaults to None.
        endpoint (str, optional): 下载节点地址. Defaults to 'https://hf-mirror.com'.
        use_modelscope: 是否使用 ModelScope 下载

    Returns:
        _type_: _description_
    """
    if use_modelscope:
        return snapshot_download_ms(model_id=model_name,
                            ignore_file_pattern=["*.msgpack", "*.h5", "*.ot"],
                            cache_dir=cache_dir)
        
    return snapshot_download(repo_id=model_name,
                            repo_type='model',
                            ignore_patterns=["*.msgpack", "*.h5", "*.ot"],
                            local_dir=cache_dir,
                            local_dir_use_symlinks=False,  # 不转为缓存乱码的形式, auto, Small files (<5MB) are duplicated in `local_dir` while a symlink is created for bigger files.
                            endpoint=endpoint,
                            etag_timeout=60,
                            resume_download=True)
    
    
    
    
def merge_lora_to_base_model(model_name_or_path, adapter_name_or_path, save_path):
    """使用该脚本，将lora的权重合并大base model中

    Args:
        model_name_or_path (_type_): base model 的名称或路径
        adapter_name_or_path (_type_): lora 的名称或路径
        save_path (_type_): 合并后的模型保存路径
    """
    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        # llama不支持fast
        use_fast=False if config.model_type == 'llama' else True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        device_map='auto',
        # device_map={'': 'cpu'}
    )
    model = PeftModel.from_pretrained(model, adapter_name_or_path, device_map={'': 'cpu'})
    model = model.merge_and_unload()

    tokenizer.save_pretrained(save_path)
    model.save_pretrained(save_path)
    logger.info(f"\nMerged model and tokenizer have saved to {save_path}")
    return model
    
    
def load_model_and_tokenizer(model_name_or_path, load_in_4bit=False, 
                             adapter_name_or_path=None, use_qlora=False,
                             gradient_checkpointing=True, lora_rank=64, lora_alpha=16, 
                             lora_dropout=0.05, device_map='auto'):
    """加载模型，支持4bit

    Args:
        model_name_or_path (_type_): 模型名称或路径
        load_in_4bit (bool, optional): 是否使用 4bit 量化进行推理. Defaults to False.
        adapter_name_or_path (_type_, optional):  Lora 权重路径. Defaults to None.
        lora_rank：qlora矩阵的秩。一般设置为8、16、32、64等，在qlora论文中作者设为64。越大则参与训练的参数量越大，一般来说效果会更好，但需要更多显存，。
        lora_alpha: qlora中的缩放参数。一般设为16、32即可。
        lora_dropout: lora权重的dropout rate。
        learning_rate：qlora中的学习率设置更大一些，一般为1e-4、2e-4。
        gradient_checkpointing：如果显存捉襟见肘，可以开启。以时间换空间，模型不缓存激活状态，会进行两次forward计算，以节省显存。

    Returns:
        Model: 模型
        Tokenizer: 分词器
    """
    logger.info("加载模型....")
    if load_in_4bit or use_qlora:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
        )
    else:
        quantization_config = None

    # 加载base model
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        load_in_4bit=load_in_4bit,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        device_map=device_map,
        quantization_config=quantization_config
    )

    # 加载adapter
    if not use_qlora and adapter_name_or_path is not None:
        model = PeftModel.from_pretrained(model, adapter_name_or_path)
        
    if use_qlora:
        # casts all the non int8 modules to full precision (fp32) for stability
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=gradient_checkpointing)
        print(f'memory footprint of model: {model.get_memory_footprint()/(1024*1024*1024)} GB')
        # 找到所有需要插入adapter的全连接层
        target_modules = find_all_linear_names(model)
        # 初始化lora配置
        config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, config)
        # model.print_trainable_parameters()
        model.config.torch_dtype = torch.float32
        
    # 加载tokenzier
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        # llama不支持fast
        use_fast=False if model.config.model_type == 'llama' else True
    )
    # QWenTokenizer比较特殊，pad_token_id、bos_token_id、eos_token_id均为None。eod_id对应的token为<|endoftext|>
    if tokenizer.__class__.__name__ == 'QWenTokenizer':
        tokenizer.pad_token_id = tokenizer.eod_id
        tokenizer.bos_token_id = tokenizer.eod_id
        tokenizer.eos_token_id = tokenizer.eod_id
    elif tokenizer.__class__.__name__ == 'LlamaTokenizer':
        assert tokenizer.eos_token_id is not None
        assert tokenizer.bos_token_id is not None
        tokenizer.pad_token_id = tokenizer.unk_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id
    # ChatGLMTokenizer不需要设置，仅设置其他tokenizer
    elif tokenizer.__class__.__name__ != 'ChatGLMTokenizer':
        assert tokenizer.eos_token_id is not None
        assert tokenizer.bos_token_id is not None
        tokenizer.pad_token_id = tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id
        
    # # 部分tokenizer没有pad_token_id
    # if tokenizer.pad_token_id is None:
    #     tokenizer.pad_token_id = tokenizer.unk_token_id
    # # 部分tokenizer的pad_token_id与eos_token_id相同，如InternLM，会导致无法计算eos_token_id的loss。将pad_token_id设为unk_token_id
    # if tokenizer.pad_token_id == tokenizer.eos_token_id and tokenizer.unk_token_id is not None:
    #     tokenizer.pad_token_id = tokenizer.unk_token_id
    # # 如果两者相同，模型训练时不会计算eos_token_id的loss
    # if tokenizer.pad_token_id == tokenizer.eos_token_id:
    #     raise Exception('pad_token_id should not be equal to eos_token_id')

    logger.info("模型准备就绪")
    return model, tokenizer


def verify_model_dtype(model):
    """
    查看模型种各种类型的参数的情况
    """
    dtype2param_num = defaultdict(int)  # 每种数据类型的参数量
    dtype2param_name = defaultdict(list)  # 每种数据类型的参数名称
    dtype2trainable_param_num = defaultdict(int)  # 每种数据类型参与训练的参数量
    dtype2trainable_param_name = defaultdict(list)  # 每种数据类型参与训练的参数名称
    for name, p in model.named_parameters():
        dtype = p.dtype
        dtype2param_num[dtype] += p.numel()
        dtype2param_name[dtype].append(name)
        if p.requires_grad:
            dtype2trainable_param_num[dtype] += p.numel()
            dtype2trainable_param_name[dtype].append(name)
    # 统计全部参数中，各种类型参数分布
    total = 0
    print('verify all params of the model')
    for k, v in dtype2param_num.items():
        total += v
    for k, v in dtype2param_num.items():
        print(k, v, v / total)
    for k, v in dtype2trainable_param_name.items():
        print(k, v)

    print()
    # 统计可训练参数中，各种类型参数分布
    print('verify trainable params the model')
    total_trainable = 0
    for k, v in dtype2trainable_param_num.items():
        total_trainable += v
    for k, v in dtype2trainable_param_num.items():
        print(k, v, v / total_trainable)
    for k, v in dtype2trainable_param_num.items():
        print(k, v)
        

def find_all_linear_names(model):
    """
    找出所有全连接层，为所有全连接添加adapter
    """
    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)



class Pipeline:
    def __init__(self, 
                 model: object = None,
                 tokenizer: object = None,
                 model_name_or_path: str = None,
                 use_qlora: bool = True,
                 load_in_4bit: bool = False,
                 device_map: str = "auto",
                 adapter_name_or_path: str = None,
                 gradient_checkpointing: bool = True, 
                 lora_rank: int = 64, 
                 lora_alpha: int = 16, 
                 lora_dropout: int = 0.05, 
                 *args, **kargs):
        if model is not None and tokenizer is not None:
            self.model, self.tokenizer = model, tokenizer
        else:
            self.model, self.tokenizer = load_model_and_tokenizer(model_name_or_path, 
                                                                    use_qlora=use_qlora, load_in_4bit=load_in_4bit, device_map=device_map,
                                                                    gradient_checkpointing=gradient_checkpointing,
                                                                    adapter_name_or_path=adapter_name_or_path,
                                                                    lora_rank=lora_rank, 
                                                                    lora_alpha=lora_alpha, 
                                                                    lora_dropout=lora_dropout)
        self.model = self.model.eval()
        
        
    def generate(self, 
                 user_inputs: Union[str, list] = None, 
                 input_ids: list[list] = None,
                 max_length: int =512, 
                 top_k: int = 10, 
                 top_p: int = 0.9, 
                 temperature: int = 0.7, 
                 num_beams: int = 1,
                 num_return_sequences: int = 1,
                 *args, **kargs):
        """
        Generate text using the model
        """
        if input_ids is None:
            user_inputs_type = type(user_inputs)
            if user_inputs_type == str:
                user_inputs = [user_inputs]
            input_ids = self.tokenizer.batch_encode_plus(user_inputs, max_length=max_length, truncation=True, add_special_tokens=True)["input_ids"]

        else:
            user_inputs_type = type(input_ids[0])
            
            
        # 找出batch中的最大长度
        lengths = [len(x) for x in input_ids]
        # 取出batch中的最大长度
        batch_max_len = min(max(lengths), max_length)


        for input_index, ids in enumerate(input_ids):
            input_id_pad_len = batch_max_len - len(ids)
            input_ids[input_index] = [self.tokenizer.pad_token_id] * input_id_pad_len + ids
        
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        input_ids = input_ids.to(self.model.device)
        
        outputs = self.model.generate(input_ids,
                                        max_new_tokens=max_length,
                                        do_sample=True,
                                        top_k=top_k,
                                        top_p=top_p,
                                        temperature=temperature,
                                        num_beams=num_beams,
                                        num_return_sequences=num_return_sequences,
        )
        model_input_ids_len = input_ids.size(1)
        response_ids = outputs[:, model_input_ids_len:]
        model_responses = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
        if user_inputs_type == str:
            model_responses = model_responses[0]
        return model_responses
    