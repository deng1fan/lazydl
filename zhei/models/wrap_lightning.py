import torch
from general_files.models.pl_base_model import BasePLModel
from general_files.utils.common_util import Result
import importlib
from transformers import (
    AutoModel,
    AutoModelForPreTraining,
    AutoModelForQuestionAnswering,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoModelForCausalLM,
    AutoModelWithLMHead,
    LlamaForCausalLM
)
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)

MODEL_MODE = {
    "base": AutoModel,
    "sequence-classification": AutoModelForSequenceClassification,
    "question-answering": AutoModelForQuestionAnswering,
    "pretraining": AutoModelForPreTraining,
    "token-classification": AutoModelForTokenClassification,
    "language-modeling": AutoModelForCausalLM,
    "seq2seq": AutoModelForSeq2SeqLM,
    "base-lm_head": AutoModelWithLMHead,
    "llama": LlamaForCausalLM,
}


class ModelNet(BasePLModel):
    def __init__(self, config, tokenizer, as_pipeline=False):
        super(ModelNet, self).__init__(config, tokenizer)

        # ---------------------------------------------------------------------------- #
        #                         加载模型的参数设置
        # ---------------------------------------------------------------------------- #
        model_from_pretrained_args = dict(
            model_name_or_path=config.pretrained_ckpt,
            cache_dir=self.config.cache_dir,
        )

        if config.get("fp16"):
            model_from_pretrained_args["load_in_8bit"] = True

        model_from_pretrained_args["device_map"] = self.config.get(
            "device_map", "auto")

        if ":" in config.pretrained_ckpt:
            model_from_pretrained_args["hyparam"] = config
            model_from_pretrained_args["tokenizer"] = tokenizer

        # ---------------------------------------------------------------------------- #
        #                         加载模型的类设置
        # ---------------------------------------------------------------------------- #
        if ":" in config.pretrained_ckpt:
            model_processor_name = self.config.pretrained_ckpt.split(':')[0]
            module_path = config.logger_project + '.modules.' + model_processor_name
            try:
                module = importlib.import_module(module_path)
            except ModuleNotFoundError as r:
                raise ValueError(f"Please add a processor for this model: {model_processor_name}\n"
                                 f"Error module path：{module_path}")
            processor_name = 'CustomModel'
            processor_class = getattr(module, processor_name)
        else:
            processor_class = MODEL_MODE[config.hf_model_type]

        # ---------------------------------------------------------------------------- #
        #                         实例化模型
        # ---------------------------------------------------------------------------- #
        self.backbone = processor_class.from_pretrained(
            **model_from_pretrained_args)

        # ---------------------------------------------------------------------------- #
        #                         模型设置
        # ---------------------------------------------------------------------------- #
        if as_pipeline:
            self.eval()
        else:
            self.train()

        if config.get("fp16"):
            self.backbone = prepare_model_for_int8_training(self.backbone)

        # ---------------------------------------------------------------------------- #
        #                         LoRA 设置
        # ---------------------------------------------------------------------------- #
        if config.get("lora"):
            config = LoraConfig(
                r=int(self.config.lora_r),
                lora_alpha=int(self.config.lora_alpha),
                target_modules=self.config.lora_target_modules,
                lora_dropout=float(self.config.lora_dropout),
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, config)

    def get_backbone(self):
        return self.backbone

    def train(self):
        self.backbone.resize_token_embeddings(self.tokenizer.vocab_size)
        self.backbone = self.backbone.train()
        self.stage = "train"

    def eval(self):
        self.backbone = self.backbone.eval()
        self.stage = "test"

    def forward(self,
                # batch, seq_len
                input_ids,
                # batch, seq_len
                labels=None,
                # 其他参数或特征
                **other_features  # 如果有其他特征参数，建议加decoder前缀，保持框架一致性
                ):
        result = Result()

        # ---------------------------------------------------------------------------- #
        #                         组装模型输入参数
        # ---------------------------------------------------------------------------- #
        forward_args = dict(
            input_ids=input_ids,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            output_hidden_states=True,
            output_attentions=True,
            return_dict=True,
            use_cache=False if self.stage == 'test' else True,
            decoder_stage=self.stage,
            **other_features,
        )
        if labels is not None:
            forward_args["labels"] = torch.where(
                labels == self.tokenizer.pad_token_id, -100, labels)

        # ---------------------------------------------------------------------------- #
        #                         模型计算
        # ---------------------------------------------------------------------------- #
        if self.stage != "train":
            with torch.no_grad():
                outputs = self.backbone(**forward_args)
        else:
            outputs = self.backbone(**forward_args)

        # ---------------------------------------------------------------------------- #
        #                         模型输出处理
        # ---------------------------------------------------------------------------- #
        lm_logits = outputs[0]

        if len(self.config.get("loss")) < 1:
            raise Exception("请至少选择一个损失函数！")
        loss = 0
        ###############################################
        # 计算交叉熵损失
        ###############################################
        if "lm_loss" in self.config.get("loss") and labels is not None:
            result.add(labels=labels)
            lm_loss = self.CrossEntropyLoss(logits=lm_logits, labels=labels)
            result.add(lm_loss=lm_loss)
            loss += lm_loss

        if self.stage != "test" and (
                loss != loss or isinstance(loss, int)
        ):
            raise Exception("Loss为Nan或无梯度，请先检查数据正确性以及超参中 Loss 是否正确选择！")

        result.add(loss=loss)
        return result
