from lazydl.utils.log import Logger
from torch.utils.data import Dataset, DataLoader
import lazydl as l

logger = Logger(__name__)


class BaseDataset(Dataset):
    def __init__(self, file, tokenizer, max_seq_length):
        self.tokenizer = tokenizer
        self.bos_token_id = tokenizer.bos_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.max_seq_length = max_seq_length
        logger.info('Loading data: {}'.format(file))
        with open(file, 'r', encoding='utf8') as f:
            data_list = f.readlines()
        logger.info("there are {} data in dataset".format(len(data_list)))
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)



def rename_columns(dataset, column_map):
    for old_name, new_name in column_map.items():
        dataset = dataset.rename_column(old_name, new_name)
    return dataset



def load_data(config, tokenizer, return_dataloader=True, stage="train"):
    
    dataset_module = l.load_class(config.dataset_module_file)
    
    if stage == "train":
        data_file = config.train_data_file
    elif stage == "val":
        data_file = config.val_data_file
    elif stage == "test":
        data_file = config.test_data_file
    else:
        raise ValueError("Invalid stage")
    
    dataset = dataset_module(data_file, tokenizer, config.max_seq_length)
        
    if return_dataloader:
        data_collator = l.load_class(config.dataset_collator_function_file)
        if stage == "train":
            batch_size = config.get("train_batch_size", 1)
        elif stage == "val":
            batch_size = config.get("val_batch_size", 1)
        elif stage == "test":
            batch_size = config.get("test_batch_size", 1)
        else:
            raise ValueError("Invalid stage")
        return DataLoader(dataset, collate_fn=data_collator(tokenizer, config.max_seq_length), batch_size=batch_size, shuffle=True, num_workers=config.get("dataloader_num_workers", 1))
    return dataset
    


