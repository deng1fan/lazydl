from lazydl.utils.log import Logger
from datasets import Dataset
from torch.utils.data import DataLoader
import lazydl as l
import pandas as pd
import torch

logger = Logger(__name__)



class BaseProcessor:
    def __init__(self, data_file, tokenizer, max_uer_input_length, max_target_length, used_for_eval=False):
        self.tokenizer = tokenizer
        self.max_uer_input_length = max_uer_input_length
        self.max_target_length = max_target_length
        self.max_seq_length = max_uer_input_length + max_target_length
        self.used_for_eval = used_for_eval
        logger.info('Loading data: {}'.format(data_file))
        self.data_list = self.preprocess_dataset(data_file)
        logger.info("there are {} data in dataset".format(len(self.data_list)))
        self.data = reverse_dict_list(self.data_list)
        self.dataset = Dataset.from_dict(self.data)
        self.dataset = self.dataset.map(
            lambda batch: self.encode_data(batch),
            batched=True,
            desc="编码数据集...",
        )
        
    def encode_data(self, batch):
        raise NotImplementedError()



def iterate_data(data, step=1, desc=""):
    """迭代数据

    Args:
        data (_type_): 可遍历的数据
        step (int, optional): 遍历步长. Defaults to 1.
        desc (str, optional): 进度条说明文字. Defaults to "".

    Yields:
        _type_:  list 类型的数据
    """
    for r in tqdm([data[i:i + step] for i in range(0, len(data), step)], desc=desc):
        yield r
        
def reverse_dict_list(data_list):
    keys = data_list[0].keys()
    res = dict()
    for f in data_list:
        for key in keys:
            if key not in res:
                res[key] = []
            item = [f[key]] if isinstance(f[key], int) else f[key]
            res[key].append(item)     
    return res



def rename_columns(dataset, column_map):
    for old_name, new_name in column_map.items():
        dataset = dataset.rename_column(old_name, new_name)
    return dataset


def merge_multi_datasets(dataset1, dataset2):
    return Dataset.from_pandas(
        pd.concat([pd.DataFrame(dataset1), pd.DataFrame(dataset2)], axis=1)
    )



def load_data(config, tokenizer, return_dataloader=True, stage="train", used_for_eval=False):
    
    dataset_module = l.load_class(config.dataset_module_file)
        
    if stage == "train":
        data_file = config.train_data_file
    elif stage == "val":
        data_file = config.val_data_file
    elif stage == "test":
        data_file = config.test_data_file
        used_for_eval = True
    else:
        raise ValueError("Invalid stage")
    
    dataset = dataset_module(data_file, tokenizer, config.max_uer_input_length, config.max_uer_input_length, used_for_eval).dataset
    
        
    if return_dataloader:
        shuffle = True
        # data_collator = l.load_class(config.dataset_collator_function_file)
        if stage == "train":
            batch_size = config.get("train_batch_size", 1)
        elif stage == "val":
            batch_size = config.get("val_batch_size", 1)
            shuffle = False
        elif stage == "test":
            batch_size = config.get("test_batch_size", 1)
            shuffle = False
        else:
            raise ValueError("Invalid stage")
        vectorization_fn = vectorization(tokenizer.pad_token_id)
        return DataLoader(dataset, 
                          collate_fn=vectorization_fn, 
                          batch_size=batch_size, shuffle=shuffle, num_workers=config.get("dataloader_num_workers", 1))
    return dataset
   

class vectorization():
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        batch = reverse_dict_list(batch)
        keys = batch.keys()

        input_ids = batch["input_ids"]
        if "labels" in keys:
            labels = batch["labels"]
            
        # 找出batch中的最大长度
        lengths = [len(x) for x in labels]
        # 取出batch中的最大长度
        batch_max_len = max(lengths)


        for input_index, ids in enumerate(input_ids):
            input_id_pad_len = batch_max_len - len(ids)
            input_ids[input_index] = [self.pad_token_id] * input_id_pad_len + ids
            if "labels" in keys:
                label_pad_len = batch_max_len - len(labels[input_index])
                labels[input_index] = [self.pad_token_id] * label_pad_len + labels[input_index]

        batch["input_ids"] = input_ids
        if "labels" in keys:
            batch["labels"] = labels

        for key in keys:
            try:
                batch[key] = torch.LongTensor(batch[key])
            except ValueError:
                continue
            
        return batch
