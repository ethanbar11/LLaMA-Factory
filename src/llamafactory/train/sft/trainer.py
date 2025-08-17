# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/trainer_seq2seq.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
from types import MethodType
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np
import torch
from transformers import Seq2SeqTrainer
from typing_extensions import override

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from ...extras.packages import is_transformers_version_greater_than
from ..callbacks import SaveProcessorCallback
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler
import torch.nn.functional as F


if TYPE_CHECKING:
    from torch.utils.data import Dataset
    from transformers import PreTrainedTokenizer, ProcessorMixin
    from transformers.trainer import PredictionOutput

    from ...hparams import FinetuningArguments


logger = logging.get_logger(__name__)


class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    r"""Inherits Seq2SeqTrainer to compute generative metrics such as BLEU and ROUGE."""

    def __init__(
        self,
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"],
        gen_kwargs: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        if is_transformers_version_greater_than("4.46"):
            kwargs["processing_class"] = kwargs.pop("tokenizer")
        else:
            self.processing_class: PreTrainedTokenizer = kwargs.get("tokenizer")

        super().__init__(**kwargs)
        if processor is not None:
            # avoid wrong loss under gradient accumulation
            # https://github.com/huggingface/transformers/pull/36044#issuecomment-2746657112
            self.model_accepts_loss_kwargs = False

        self.finetuning_args = finetuning_args
        if gen_kwargs is not None:
            # https://github.com/huggingface/transformers/blob/v4.45.0/src/transformers/trainer_seq2seq.py#L287
            self._gen_kwargs = gen_kwargs

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

        if finetuning_args.use_dft_loss:
            from ..trainer_utils import dft_loss_func

            self.compute_loss_func = dft_loss_func

    @override
    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    @override
    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    @override
    def _get_train_sampler(self, *args, **kwargs) -> Optional["torch.utils.data.Sampler"]:
        if self.finetuning_args.disable_shuffling:
            return torch.utils.data.SequentialSampler(self.train_dataset)

        return super()._get_train_sampler(*args, **kwargs)

    def transition(self, x_0, sigma, maskable_mask):
        # move_chance = 1 - (-sigma).exp()
        move_chance = sigma
        move_indices = (torch.rand(*x_0.shape, device=x_0.device) < move_chance) & maskable_mask
        x_t = torch.where(move_indices, self.processing_class.mask_token_id, x_0)
        return x_t
        
    def diffusion_forward(self, model, x, src_mask, sampling_eps=1e-3):
        batch_size = x.shape[0]
        if src_mask is None:
            src_mask = torch.zeros_like(x, device=x.device, dtype=torch.bool)
        
        # Sample noise level
        t = (1 - sampling_eps) * torch.rand(x.shape[0], device=x.device) + sampling_eps
        
        # Compute sigma and dsigma
        sigma = t
        dsigma = torch.reciprocal(t)
        
        # Apply noise to input
        x_t = self.transition(x, sigma[:, None], maskable_mask=~src_mask)
        
        # Forward pass
        logits = model(input_ids=x_t, attention_mask=None).logits
        
        # Apply mask for loss computation
        loss_mask = x_t == self.processing_class.mask_token_id
        
        # Shift loss
        logits = logits[:, :-1]
        loss_mask = loss_mask[:, 1:]
        x_target = x[:, 1:]
        
        # Compute loss
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), 
            x_target.reshape(-1), 
            reduction="none"
        ).float().reshape(batch_size, -1)
        
        loss = loss.masked_fill(~loss_mask, 0)
        final_loss = (dsigma[:, None] * loss).sum() / loss_mask.sum()
        unweighted_loss = loss.sum() / loss_mask.sum()
        
        return final_loss, unweighted_loss, x_t

    def _get_dynamic_ratio(self) -> float:
        """Calculate the dynamic ratio based on training progress.
        At the start (0% progress), ratio is near 0.
        At 2/3 of training, ratio becomes 1.
        """
        if not hasattr(self.state, 'global_step') or not hasattr(self.state, 'max_steps'):
            logger.warning("State not initialized, using default ratio")
            return 0.9  # fallback to default if state not initialized
        
        progress = self.state.global_step / self.state.max_steps
        if progress < 1/5:
            # Linear increase from 0 to 1 during first 1/5 of training
            return progress * 5  # multiply by 5 to reach 1 at 1/5 progress
        else:
            return 1.0

    @override
    def compute_loss(self, model, inputs, *args, **kwargs):
        # if torch.rand(1) < self._get_dynamic_ratio():
        #     src_mask = inputs["labels"] == IGNORE_INDEX
        # else:
        #     src_mask = None

        # # src_mask = None

        src_mask = inputs["labels"] == IGNORE_INDEX

        # Apply diffusion forward
        final_loss, unweighted_loss, x_t = self.diffusion_forward(
            model, 
            inputs["input_ids"], 
            src_mask
        )
        # wandb.log({"loss": final_loss.item(), "unweighted_loss": unweighted_loss.item()})
        
        return final_loss

    @override
    def prediction_step(
        self,
        model: "torch.nn.Module",
        inputs: dict[str, Union["torch.Tensor", Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
        **gen_kwargs,
    ) -> tuple[Optional[float], Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        r"""Remove the prompt part in the generated tokens.

        Subclass and override to inject custom behavior.
        """
        if self.args.predict_with_generate:  # do not pass labels to model when generate
            labels = inputs.pop("labels", None)
        else:
            labels = inputs.get("labels")
            
        # Create source mask where label is IGNORE_INDEX
        src_mask = labels == IGNORE_INDEX if labels is not None else None
        loss, _, x_t = self.diffusion_forward(model, inputs["input_ids"], src_mask)

        return loss, _, _


    def save_predictions(
        self, dataset: "Dataset", predict_results: "PredictionOutput", skip_special_tokens: bool = True
    ) -> None:
        r"""Save model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.
        """
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info_rank0(f"Saving prediction results to {output_prediction_file}")

        labels = np.where(
            predict_results.label_ids != IGNORE_INDEX, predict_results.label_ids, self.processing_class.pad_token_id
        )
        preds = np.where(
            predict_results.predictions != IGNORE_INDEX,
            predict_results.predictions,
            self.processing_class.pad_token_id,
        )

        for i in range(len(preds)):
            pad_len = np.nonzero(preds[i] != self.processing_class.pad_token_id)[0]
            if len(pad_len):  # move pad token to last
                preds[i] = np.concatenate((preds[i][pad_len[0] :], preds[i][: pad_len[0]]), axis=-1)

        decoded_inputs = self.processing_class.batch_decode(dataset["input_ids"], skip_special_tokens=False)
        decoded_preds = self.processing_class.batch_decode(preds, skip_special_tokens=skip_special_tokens)
        decoded_labels = self.processing_class.batch_decode(labels, skip_special_tokens=skip_special_tokens)

        with open(output_prediction_file, "w", encoding="utf-8") as f:
            for text, pred, label in zip(decoded_inputs, decoded_preds, decoded_labels):
                f.write(json.dumps({"prompt": text, "predict": pred, "label": label}, ensure_ascii=False) + "\n")
