import transformers
from transformers import (
    PreTrainedModel,
    TrainingArguments,
    DataCollator,
    PreTrainedTokenizerBase,
    EvalPrediction,
    TrainerCallback,
)
from typing import Callable, Dict, List, Optional, Tuple, Union, Any
from torch import nn
from torch.utils.data import Dataset
from transformers.utils import (
    logging,
)
import os
import torch
import lightning.pytorch as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, StochasticWeightAveraging
from lightning.pytorch.loggers import CometLogger
from lightning.pytorch.callbacks.progress import TQDMProgressBar
from lightning.pytorch.utilities import rank_zero_only
from colorama import Fore



logger = logging.get_logger(__name__)

# Name of the files used for checkpointing
TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
SCHEDULER_NAME = "scheduler.pt"
SCALER_NAME = "scaler.pt"


class HFTrainer(transformers.Trainer):
    """
    主要修改逻辑：通过传入compute_loss，支持自定义loss计算方式
    """
    def __init__(
            self,
            model: Union[PreTrainedModel, nn.Module] = None,
            args: TrainingArguments = None,
            data_collator: Optional[DataCollator] = None,
            train_dataset: Optional[Dataset] = None,
            eval_dataset: Optional[Dataset] = None,
            tokenizer: Optional[PreTrainedTokenizerBase] = None,
            model_init: Callable[[], PreTrainedModel] = None,
            compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
            callbacks: Optional[List[TrainerCallback]] = None,
            optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
            preprocess_logits_for_metrics: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = None,
            compute_loss=None,
    ):
        super(Trainer_HF, self).__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )
        self.loss_func = compute_loss

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        重写loss的计算方式
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        return self.loss_func(model, inputs, self.args, return_outputs)
    
    """
    修改checkkpoint的保存逻辑，只保存lora
    """
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")
        # 保存lora权重和配置
        self.model.save_pretrained(
            output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
        )

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))



class LitTrainer:
    def __init__(
        self,
        config,
        model,
        train_dataloader,
        val_dataloader,
        tokenizer,
        experiment,
    ):
        self.config = config
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.tokenizer = tokenizer
        
        
        if experiment:
            logger = CustomCometLoggerForLit()
            logger._experiment = experiment
        else:
            logger = None

        # checkpoint_callback = ModelCheckpoint(
        #     monitor="val_loss",
        #     mode="min",
        #     save_top_k=config.save_top_k,
        #     save_last=True,
        #     verbose=True,
        #     dirpath=config.output_dir,
        #     filename="best_model",
        #     auto_insert_metric_name=False,
        # )

        early_stop_callback = EarlyStopping(
            monitor="val_loss",
            min_delta=0.001,
            patience=5,
            verbose=True,
            mode="min",
        )

        callbacks = [
            early_stop_callback,
            # checkpoint_callback,
            LiteProgressBar(),
        ]

        if config.get("use_swa"):
            callbacks.append(StochasticWeightAveraging(swa_lrs=1e-2))

        self.trainer = L.Trainer(
            logger=logger, callbacks=callbacks, **config.lit_args
        )


    def train(self):
        if self.config.auto_lr_find:
            lr_finder = self.trainer.tuner.lr_find(self.model, train_dataloaders=self.train_dataloader)
            # 展示loss和学习率的曲线
            fig = lr_finder.plot(suggest=True)
            fig.show()
            # 设置为推荐的学习率
            self.model.config.lr = lr_finder.suggestion()
            
        return self.trainer.fit(model=self.model,
                                train_dataloaders=self.train_dataloader,
                                val_dataloaders=self.val_dataloader)
        
    def save_model(self, save_path: str):
        self.model.backbone.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        logger.info(f"Model and tokenizer have saved to {save_path}")
        
        

class CustomCometLoggerForLit(CometLogger):
    def __init__(self):
        super(CustomCometLoggerForLit, self).__init__()

    @rank_zero_only
    def finalize(self, status: str) -> None:
        r"""
        When calling ``self.experiment.end()``, that experiment won't log any more data to Comet.
        That's why, if you need to log any more data, you need to create an ExistingCometExperiment.
        For example, to log data when testing your model after training, because when training is
        finalized :meth:`CometLogger.finalize` is called.

        This happens automatically in the :meth:`~CometLogger.experiment` property, when
        ``self._experiment`` is set to ``None``, i.e. ``self.reset_experiment()``.
        """
        # self.experiment.end()
        # self.reset_experiment()
        
        
        
class LiteProgressBar(TQDMProgressBar):
    def __init__(self, refresh_rate: int = 1, process_position: int = 0):
        super().__init__(refresh_rate, process_position)

    def get_metrics(
        self, trainer: "L.Trainer", pl_module: "L.LightningModule"
    ) -> Dict[str, Union[int, str]]:
        items = super().get_metrics(trainer, pl_module)
        # items['ppl'] = round(items['ppl'], 1) if 'ppl' in items else None
        items["lr"] = round(items["lr"], 7) if "lr" in items else None
        items.pop("v_num", None)
        return items

    def init_train_tqdm(self):
        bar = super().init_train_tqdm()
        bar.bar_format = "%s{l_bar}%s{bar}%s{r_bar}" % (
            Fore.GREEN,
            Fore.GREEN,
            Fore.GREEN,
        )
        return bar

    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        bar.set_description("Validating")
        bar.bar_format = "%s{l_bar}%s{bar}%s{r_bar}" % (
            Fore.GREEN,
            Fore.GREEN,
            Fore.GREEN,
        )
        bar.leave = False
        return bar