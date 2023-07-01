from omegaconf import DictConfig
import hydra
from datasets import load_dataset
from transformers import LlamaTokenizer
import zhei as j

log = j.Logger(__name__)

@hydra.main(version_base="1.2", config_path="configs/", config_name="default_config.yaml")
def main(config: DictConfig) -> float:
    j.hi()
    config = j.init_env(config)
    # ---------------------------------------------------------------------------- #
    #                          加载数据集                                    
    # ---------------------------------------------------------------------------- #
    log.info("Loading dataset...")
    log.warning("This may take a while...")
    log.error("This may take a long while...")
    # dataset = load_dataset("cais/mmlu", "high_school_geography")
    # dataset.push_to_hub("cais/mmlu", "high_school_geography")
    
    # tokenizer = LlamaTokenizer.from_pretrained("chainyo/alpaca-lora-7b")
    # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    # columns_map = {
    #     "input_ids": "question",
    #     "labels": "answer",
    # }

    # dataset = j.advance_tokenize(tokenizer, dataset, columns_map, truncation=True, padding=True)
    
    config = j.set_processing_units(config)
    
    print()

    
    
    
if __name__ == "__main__":
    main()