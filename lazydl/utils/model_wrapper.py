from typing import Any
import lightning.pytorch as L
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers.optimization import (
    Adafactor,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_constant_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)


# update this and the import above to support new schedulers from transformers.optimization
arg_to_scheduler = {
    "linear": get_linear_schedule_with_warmup,
    "cosine": get_cosine_schedule_with_warmup,
    "cosine_w_restarts": get_cosine_with_hard_restarts_schedule_with_warmup,
    "polynomial": get_polynomial_decay_schedule_with_warmup,
    "constant": get_constant_schedule_with_warmup,  # not supported for now
}


class BaseLitModel(L.LightningModule):
    def __init__(self, config, tokenizer):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.stage = 'train'
        if "dropout" not in config:
            self.dropout = None
        else:
            self.dropout = nn.Dropout(config.dropout)
        self.validation_step_outputs = []

    def training_step(self, batch: Any, batch_idx: int):
        outputs = self(**batch)
        if outputs['loss'] != outputs['loss']:
            raise Exception("Loss为Nan，请先检查数据正确性！")
        self.log("loss", outputs['loss'], prog_bar=True, logger=True, sync_dist=True, on_step=True,
                         on_epoch=True, rank_zero_only=True)
        for key in outputs.keys():
            if '_loss' in key:
                key_loss = round(float(torch.clamp(outputs[key].cpu(), max=99, min=0)), 3)
                self.log("train/" + key, key_loss, prog_bar=False, logger=True, sync_dist=True, on_step=True,
                         on_epoch=True, rank_zero_only=True)
        if 'lm_loss' in outputs:
            ppl = round(float(torch.clamp(torch.exp(outputs['lm_loss']).cpu(), max=99, min=0)), 3)
            self.log("ppl", ppl, prog_bar=False, logger=True, sync_dist=True, rank_zero_only=True)
        
        self.log("lr", round(self.opt.param_groups[0]['lr'], 9), prog_bar=True, logger=True, on_step=True,
                 rank_zero_only=True, sync_dist=True)
        self.log("current_epoch", float(self.current_epoch))
        self.log("progress", round(self.global_step / self.total_steps(), 2))
        return {
            'loss': outputs['loss'],
        }
    
    def on_train_epoch_end(self):
        self.log("progress", round(self.global_step / self.total_steps(), 2))

    def validation_step(self, batch: Any, batch_idx: int):
        outputs = self(**batch)
        self.log("val/step_loss", outputs['loss'], prog_bar=False)
        self.validation_step_outputs.append(outputs['loss'])
        return {"val_step_loss": outputs['loss']}

    def on_validation_epoch_end(self):
        mean_loss = torch.stack([x for x in self.validation_step_outputs]).mean().item()
        log = {
            'val_loss': mean_loss,
        }
        self.log('val_loss', mean_loss)
        self.validation_step_outputs.clear()
        return log

    def prepare_other_features_for_generation(self, batch):
        ignore_keys = ['input_ids', 'labels', 'decoder_input_ids']
        other_features = dict()
        for key in batch.keys():
            if key not in ignore_keys:
                try:
                    other_features[key] = torch.LongTensor(batch[key]).to(self.device)
                except ValueError as e:
                    other_features[key] = batch[key]
        return other_features


    def get_lr_scheduler(self):
        lr_scheduler = self.config.get("lr_scheduler", "constant")
        get_schedule_func = arg_to_scheduler[lr_scheduler]
        total_steps = self.total_steps()
        if self.config.get("warmup_ratio", 0.1) > 0:
            warmup_steps = self.config.get("warmup_ratio", 0.1) * total_steps
        else:
            warmup_steps = self.config.get("warmup_steps", 100)

        if lr_scheduler != "constant":
            scheduler = get_schedule_func(self.opt, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
        else:
            scheduler = get_schedule_func(self.opt, num_warmup_steps=warmup_steps)

        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return scheduler

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.config.get("weight_decay", 0.0),
            },
            {
                "params": [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        if self.config.get("adafactor", None):
            optimizer = Adafactor(
                optimizer_grouped_parameters,
                lr=self.config.get("lr", 1e-5),
                scale_parameter=False,
                relative_step=False,
            )

        else:
            optimizer = AdamW(
                optimizer_grouped_parameters,
                lr=self.config.get("lr", 1e-5),
                eps=self.config.get("adam_epsilon", 0.1),
            )
        self.opt = optimizer
        scheduler = self.get_lr_scheduler()
        return [optimizer] , [scheduler]

    def total_steps(self) -> int:
        return self.trainer.estimated_stepping_batches
    
    def generate(self, *args, **kargs):
        return self.backbone.generate(*args, **kargs)
    
    

    

