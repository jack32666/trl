# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

import gc
import math
import os
import textwrap
import time
from collections import defaultdict
from typing import Optional, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import broadcast, gather_object
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import (
    BaseImageProcessor,
    DataCollatorWithPadding,
    FeatureExtractionMixin,
    GenerationConfig,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    Trainer,
    TrainerCallback,
    TrainerControl,
    is_wandb_available,
)
from transformers.integrations import get_reporting_integration_callbacks
from transformers.trainer import DEFAULT_CALLBACKS, DEFAULT_PROGRESS_CALLBACK
from transformers.trainer_callback import CallbackHandler, ExportableState, PrinterCallback
from transformers.utils.deprecation import deprecate_kwarg

from ..models.utils import unwrap_model_for_generation
from ..trainer.utils import (
    OnlineTrainerState,
    batch_generation,
    disable_dropout_in_model,
    exact_div,
    first_true_indices,
    forward,
    get_reward,
    get_beta,
    prepare_deepspeed,
    print_rich_table,
    truncate_response,
    sample_timesteps,
)
from .rloo_config import RLOOConfig
from .utils import generate_model_card
import random


if is_wandb_available():
    import wandb

INVALID_LOGPROB = 1.0


class RLOOTrainer(Trainer):
    _tag_names = ["trl", "rloo"]

    @deprecate_kwarg(
        "tokenizer", "0.14.0", "processing_class", warn_if_greater_or_equal_version=True, raise_if_both_names=True
    )
    def __init__(
        self,
        config: RLOOConfig,
        processing_class: Optional[
            Union[PreTrainedTokenizerBase, BaseImageProcessor, FeatureExtractionMixin, ProcessorMixin]
        ],
        policy: nn.Module,
        ref_policy: nn.Module,
        reward_model: nn.Module,
        beta_policy: nn.Module,
        train_dataset: Dataset,
        data_collator: Optional[DataCollatorWithPadding] = None,
        eval_dataset: Optional[Union[Dataset, dict[str, Dataset]]] = None,
        # less commonly used
        optimizers: tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        beta_optimizers: tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        callbacks: Optional[list[TrainerCallback]] = None,
    ) -> None:
        if ref_policy is policy:
            raise ValueError(
                "`policy` and `ref_policy` cannot be the same object. If you want `ref_policy` to be the "
                "same as `policy`, you must mass a copy of it, or `None` if you use peft."
            )

        self.args = config
        args = config
        self.processing_class = processing_class
        self.policy = policy

        # Define the collator if not provided
        if data_collator is None:
            data_collator = DataCollatorWithPadding(self.processing_class)

        self.policy.generation_config.eos_token_id = (
            None  # disable `pad_token_id` and `eos_token_id` because we just want to
        )
        self.policy.generation_config.pad_token_id = None  # generate tokens without truncation / padding

        self.ref_policy = ref_policy
        self.reward_model = reward_model
        self.beta_policy = beta_policy
        self.train_dataset = train_dataset
        self.train_dataset_len = len(train_dataset)
        self.data_collator = data_collator
        self.eval_dataset = eval_dataset
        self.optimizer, self.lr_scheduler = optimizers
        self.beta_optimizer, self.beta_lr_scheduler = beta_optimizers
        self.optimizer_cls_and_kwargs = None  # needed for transformers >= 4.47



        #########
        # calculate various batch sizes
        #########
        if args.total_episodes is None:  # allow the users to define episodes in terms of epochs.
            args.total_episodes = int(args.num_train_epochs * self.train_dataset_len)
        accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
        self.accelerator = accelerator
        args.world_size = accelerator.num_processes
        args.local_batch_size = (
            args.per_device_train_batch_size * args.gradient_accumulation_steps * args.num_mini_batches
        )
        args.micro_batch_size = int(args.per_device_train_batch_size * args.world_size)
        args.batch_size = int(args.local_batch_size * args.world_size)
        args.mini_batch_size = exact_div(
            args.batch_size, args.num_mini_batches, "`batch_size` must be a multiple of `num_mini_batches`"
        )
        args.local_mini_batch_size = exact_div(
            args.local_batch_size, args.num_mini_batches, "`local_batch_size` must be a multiple of `num_mini_batches`"
        )
        args.num_total_batches = math.ceil(
            args.total_episodes / args.batch_size
        )  # we may train for more than `total_episodes`
        time_tensor = torch.tensor(int(time.time()), device=accelerator.device)
        time_int = broadcast(time_tensor, 0).item()  # avoid different timestamps across processes
        args.run_name = f"{args.exp_name}__{args.seed}__{time_int}"
        self.local_seed = args.seed + accelerator.process_index * 100003  # Prime
        if args.num_sample_generations > 0:
            self.sample_generations_freq = max(1, args.num_total_batches // args.num_sample_generations)
        self.local_dataloader_batch_size = exact_div(
            args.local_batch_size, args.rloo_k, "`local_batch_size` must be a multiple of rloo_k"
        )  # RLOO logic: needed because RLOO repeats the same prompt args.rloo_k times

        #########
        # setup model, optimizer, and others
        #########
        for module in [policy, ref_policy, reward_model, beta_policy]:
            disable_dropout_in_model(module)
        if args.stop_token and args.stop_token == "eos":
            args.stop_token_id = self.processing_class.eos_token_id
        self.model = policy
        self.create_optimizer_and_scheduler(
            num_training_steps=args.num_total_batches
        )  # note that we are calling `self.lr_scheduler.step()` manually only at the batch level

        #########
        ### trainer specifics
        #########
        default_callbacks = DEFAULT_CALLBACKS + get_reporting_integration_callbacks(self.args.report_to)
        self.callbacks = default_callbacks if callbacks is None else default_callbacks + callbacks
        self.callback_handler = CallbackHandler(
            self.callbacks, self.model, self.processing_class, self.optimizer, self.lr_scheduler
        )
        self.add_callback(PrinterCallback if self.args.disable_tqdm else DEFAULT_PROGRESS_CALLBACK)
        self.control = TrainerControl()
        self.state = OnlineTrainerState(
            is_local_process_zero=self.is_local_process_zero(),
            is_world_process_zero=self.is_world_process_zero(),
            stateful_callbacks=[
                cb for cb in self.callback_handler.callbacks + [self.control] if isinstance(cb, ExportableState)
            ],
        )

        self.current_flos = 0
        self.hp_search_backend = None
        self.is_deepspeed_enabled = getattr(self.accelerator.state, "deepspeed_plugin", None) is not None
        self.is_fsdp_enabled = getattr(self.accelerator.state, "fsdp_plugin", None) is not None
        # Create distant repo and output directory if needed
        self.hub_model_id = None
        if self.args.push_to_hub:
            self.init_hf_repo()
        if self.args.should_save:
            os.makedirs(self.args.output_dir, exist_ok=True)
        self.backup_model = None

        # Add tags for models that have been loaded with the correct transformers version
        if hasattr(self.model, "add_model_tags"):
            self.model.add_model_tags(self._tag_names)

        #########
        ### setup dataloader
        #########
        self.dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.local_dataloader_batch_size,
            shuffle=True,
            collate_fn=self.data_collator,
            drop_last=True,  # needed; otherwise the last batch will be of ragged shape
        )
        # sync random states for DataLoader(shuffle=True) before `accelerator.prepare`
        # see https://gist.github.com/vwxyzjn/2581bff1e48e185e0b85b6dfe1def79c
        torch.manual_seed(args.seed)
        self.model, self.optimizer, self.dataloader = accelerator.prepare(self.model, self.optimizer, self.dataloader)
        torch.manual_seed(self.local_seed)  # reset the local seed again

        self.eval_dataloader = DataLoader(
            self.eval_dataset,
            batch_size=args.per_device_eval_batch_size,
            collate_fn=self.data_collator,
            drop_last=True,
        )  # no need to shuffle eval dataset
        self.eval_dataloader = accelerator.prepare(self.eval_dataloader)

        if self.is_deepspeed_enabled:
            self.reward_model = prepare_deepspeed(
                self.reward_model, args.per_device_train_batch_size, args.fp16, args.bf16
            )
            self.ref_policy = prepare_deepspeed(
                self.ref_policy, args.per_device_train_batch_size, args.fp16, args.bf16
            )
            self.beta_policy = prepare_deepspeed(
                self.beta_policy, args.per_device_train_batch_size, args.fp16, args.bf16
            )
            self.deepspeed = self.model
        else:
            self.ref_policy = self.ref_policy.to(self.accelerator.device)
            self.reward_model = self.reward_model.to(self.accelerator.device)
            self.beta_policy = self.beta_policy.to(self.accelerator.device)

    def get_train_dataloader(self) -> DataLoader:
        return self.dataloader

    def get_eval_dataloader(self) -> DataLoader:
        return self.eval_dataloader

    def train(self):
        args = self.args
        accelerator = self.accelerator
        optimizer = self.optimizer
        beta_optimizer = self.beta_optimizer
        model = self.model
        self.model_wrapped = self.model
        ref_policy = self.ref_policy
        reward_model = self.reward_model
        beta_policy = self.beta_policy
        processing_class = self.processing_class
        dataloader = self.dataloader
        device = accelerator.device

        def repeat_generator():
            while True:
                yield from dataloader

        iter_dataloader = iter(repeat_generator())
        generation_config = GenerationConfig(
            max_new_tokens=args.response_length,
            temperature=(args.temperature + 1e-7),
            top_k=0.0,
            top_p=1.0,
            do_sample=True,
        )

        accelerator.print("===training policy===")
        start_time = time.time()
        stats_shape = (args.num_ppo_epochs, args.num_mini_batches, args.gradient_accumulation_steps)
        approxkl_stats = torch.zeros(stats_shape, device=device)
        pg_clipfrac_stats = torch.zeros(stats_shape, device=device)
        pg_loss_stats = torch.zeros(stats_shape, device=device)
        vf_clipfrac_stats = torch.zeros(stats_shape, device=device)
        entropy_stats = torch.zeros(stats_shape, device=device)
        ratio_stats = torch.zeros(stats_shape, device=device)
        model.train()

        # trainer state initialization
        self.state.global_step = 0
        self.state.episode = 0
        self.state.max_steps = args.num_total_batches * args.num_mini_batches
        self.state.num_train_epochs = args.total_episodes / self.train_dataset_len
        # Compute absolute values for logging, eval, and save if given as ratio
        if args.logging_steps is not None:
            if args.logging_steps < 1:
                self.state.logging_steps = math.ceil(self.state.max_steps * args.logging_steps)
            else:
                self.state.logging_steps = args.logging_steps
        if args.eval_steps is not None:
            if args.eval_steps < 1:
                self.state.eval_steps = math.ceil(self.state.max_steps * args.eval_steps)
            else:
                self.state.eval_steps = args.eval_steps
        if args.save_steps is not None:
            if args.save_steps < 1:
                self.state.save_steps = math.ceil(self.state.max_steps * args.save_steps)
            else:
                self.state.save_steps = args.save_steps
        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        for update in range(1, args.num_total_batches + 1):
            self.state.episode += 1 * args.batch_size
            data = next(iter_dataloader)
            with torch.no_grad():
                queries = data["input_ids"].to(device) # torch.Size([8, 64])
                queries = queries.repeat(args.rloo_k, 1) # torch.Size([16, 64]). For computing variance of the rewards per prompt
                context_length = queries.shape[1] # 64
                responses = []
                postprocessed_responses = []
                logprobs = []
                ref_logprobs = []
                scores = []
                sequence_lengths = []
                alpha_list = []
                beta_list = []
                with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
                    # query_responses.shape: torch.Size([16, 117])
                    # logitss.shape: torch.Size([16, 53, 50304]). [batch_size, sequence_length, vocab_size]. sequence_length: number of tokens in the generated response. vocab_size: logits over the entire vocabulary
                    query_responses, logitss = batch_generation(
                        unwrapped_model,
                        queries, # torch.Size([16, 64])
                        args.local_rollout_forward_batch_size, # Per rank no grad forward pass in the rollout phase.
                        processing_class.pad_token_id, # processing_class.pad_token_id: 50277, corresponds to the padding token <pad> in the tokenizer's vocabulary.
                        generation_config,
                    )

                for i in range(0, queries.shape[0], args.local_rollout_forward_batch_size):
                    query = queries[i : i + args.local_rollout_forward_batch_size] # torch.Size([1, 64])
                    query_response = query_responses[i : i + args.local_rollout_forward_batch_size] # torch.Size([1, 117])
                    response = query_response[:, context_length:] # torch.Size([1, 53]), since query_response is composed of cat(query, response)
                    logits = logitss[i : i + args.local_rollout_forward_batch_size] # torch.Size([1, 53, 50304])
                    all_logprob = F.log_softmax(logits, dim=-1) # torch.Size([1, 53, 50304])
                    logprob = torch.gather(all_logprob, 2, response.unsqueeze(-1)).squeeze(-1) # torch.Size([1, 53]). log prob of each token in response
                    del logits, all_logprob
                    torch.cuda.empty_cache() # clear up any memory 

                    ref_output = forward(ref_policy, query_response, processing_class.pad_token_id)
                    ref_logits = ref_output.logits[:, context_length - 1 : -1]
                    ref_logits /= args.temperature + 1e-7
                    ref_all_logprob = F.log_softmax(ref_logits, dim=-1)
                    ref_logprob = torch.gather(ref_all_logprob, 2, response.unsqueeze(-1)).squeeze(-1) # log prob of the response in reference policy
                    del ref_output, ref_logits, ref_all_logprob
                    torch.cuda.empty_cache()

                    # Response Processing 1. truncate response after the first occurrence of `stop_token_id`
                    postprocessed_response = response
                    if args.stop_token_id is not None:  # handle the edge case when stop_token_id exists but is 0
                        postprocessed_response = truncate_response(
                            args.stop_token_id, processing_class.pad_token_id, response
                        )

                    # Response Processing 2. run reward model on the truncated responses
                    postprocessed_query_response = torch.cat((query, postprocessed_response), 1) # torch.Size([1, 116])
                    sequence_length = first_true_indices(postprocessed_response == processing_class.pad_token_id) - 1 # torch.Size([1]): tensor([52])
                    _, score, _ = get_reward( # torch.Size([1]): tensor([-3.6719])
                        reward_model, postprocessed_query_response, processing_class.pad_token_id, context_length
                    )

                    # generate beta
                    alpha, beta = get_beta(  # torch.Size([1])
                        beta_policy, postprocessed_query_response, processing_class.pad_token_id
                    )

                    responses.append(response)
                    postprocessed_responses.append(postprocessed_response)
                    logprobs.append(logprob)
                    ref_logprobs.append(ref_logprob)
                    sequence_lengths.append(sequence_length)
                    scores.append(score)
                    alpha_list.append(alpha)
                    beta_list.append(beta)
                    
                responses = torch.cat(responses, 0) # torch.Size([16, 53]). concatenate a list of tensors along the first dimension (dimension 0).. If yo;u use torck.stack, it would inttroduce a new dimension at first dimension.
                postprocessed_responses = torch.cat(postprocessed_responses, 0) # torch.Size([16, 53])
                logprobs = torch.cat(logprobs, 0) # torch.Size([16, 53])
                ref_logprobs = torch.cat(ref_logprobs, 0) # torch.Size([16, 53])
                sequence_lengths = torch.cat(sequence_lengths, 0) # torch.Size([16])
                scores = torch.cat(scores, 0) # torch.Size([16])
                alpha_tensor = torch.cat(alpha_list, 0) # torch.Size([16])
                beta_tensor = torch.cat(beta_list, 0) # torch.Size([16])
                

                #################### BEFORE RLHF UPDATE ####################
                evaluate_score_before = self.evaluate_score(sampling=False) # torch.Size([16])
                sampled_kl_coef, beta_log_probs, beta_distr_entropy = sample_timesteps(alpha_tensor, beta_tensor) # torch.Size([16]), torch.Size([16]), torch.Size([16])
                
                del (logprob, ref_logprob, score, alpha_list, beta_list)
                torch.cuda.empty_cache()
                gc.collect() # release memory 

                # Response Processing 3. filter response. Ensure that the sample contains stop_token_id
                # responses not passing that filter will receive a low (fixed) score
                # only query humans on responses that pass that filter
                contain_eos_token = torch.any(postprocessed_responses == processing_class.eos_token_id, dim=-1) # torch.Size([16])
                if args.missing_eos_penalty is not None:
                    scores[~contain_eos_token] -= self.args.missing_eos_penalty # penalize responses missing EOS token by subtracting 1
                # accelerator.print(f"{scores=}, {(contain_eos_token.sum() / len(contain_eos_token))=}")

                # be very careful with `padding_mask_p1`; see https://excalidraw.com/#json=LWnzG4w2k5DjF_EOL_xPt,e2w3a-hFJ_gX5vOfeyXGTw
                response_idxs = torch.arange(responses.shape[1], device=responses.device).repeat(responses.shape[0], 1) # torch.Size([16, 53])
                padding_mask = response_idxs > sequence_lengths.unsqueeze(1) # torch.Size([16, 53])
                logprobs = torch.masked_fill(logprobs, padding_mask, INVALID_LOGPROB) # INVALID_LOGPROB: 1.0. torch.Size([16, 53]). Ignore padding tokens in calculations. Having a 1.0 ensures a large, noticeable penalty for invalid positions (log(1)=0).
                ref_logprobs = torch.masked_fill(ref_logprobs, padding_mask, INVALID_LOGPROB) # torch.Size([16, 53])

                # 4. compute rewards
                kl = logprobs - ref_logprobs # torch.Size([16, 53])
                non_score_reward = (-args.kl_coef * kl).sum(1) # args.kl_coef: 0.05
                rlhf_reward = scores + non_score_reward # torch.Size([16])

                # vectorized RLOO advantages implementation
                rlhf_reward = rlhf_reward.reshape(args.rloo_k, -1) # torch.Size([2, 8])
                baseline = (rlhf_reward.sum(0) - rlhf_reward) / (args.rloo_k - 1) # rlhf_reward.sum(0): torch.Size([8])
                advantages = rlhf_reward - baseline # torch.Size([2, 8])
                advantages = advantages.flatten() # torch.Size([16])
                torch.cuda.empty_cache()


            #################### RLHF UPDATE ####################
            # Do multiple epochs of PPO training, with a fresh random shuffle in each epoch
            for ppo_epoch_idx in range(args.num_ppo_epochs):
                b_inds = np.random.permutation(args.local_batch_size)
                minibatch_idx = 0
                for mini_batch_start in range(0, args.local_batch_size, args.local_mini_batch_size): # PPO update. args.local_mini_batch_size: 16
                    mini_batch_end = mini_batch_start + args.local_mini_batch_size
                    mini_batch_inds = b_inds[mini_batch_start:mini_batch_end]
                    gradient_accumulation_idx = 0
                    for micro_batch_start in range(0, args.local_mini_batch_size, args.per_device_train_batch_size): # Gradient Accumulation
                        with accelerator.accumulate(model):
                            micro_batch_end = micro_batch_start + args.per_device_train_batch_size # args.per_device_train_batch_size: 1
                            micro_batch_inds = mini_batch_inds[micro_batch_start:micro_batch_end] # (1,)
                            mb_advantage = advantages[micro_batch_inds] # torch.Size([1])
                            mb_responses = responses[micro_batch_inds] # torch.Size([1, 53])
                            mb_query_responses = query_responses[micro_batch_inds] # torch.Size([1, 117])
                            mb_logprobs = logprobs[micro_batch_inds] # torch.Size([1, 53])

                            output = forward(model, mb_query_responses, processing_class.pad_token_id) # odict_keys(['logits', 'past_key_values', 'hidden_states'])
                            logits = output.logits[:, context_length - 1 : -1] # output.logit: torch.Size([1, 117, 50304]), logits: torch.Size([1, 53, 50304])
                            logits /= args.temperature + 1e-7 # temperature: T>1, reduces their magnitude, making the differences between logits smaller, increasing randomness and encouraging diverse outputs. Ex. Generating poetry, stories, or jokes
                                                              # temperature: T<1, amplifies logit differences, making the highest logit dominate the probabilities. Ex. answers to factual questions, summarization
                            new_all_logprobs = F.log_softmax(logits, dim=-1) # torch.Size([1, 53, 50304])
                            new_logprobs = torch.gather(new_all_logprobs, 2, mb_responses.unsqueeze(-1)).squeeze(-1) # torch.Size([1, 53])
                            new_logprobs = torch.masked_fill(
                                new_logprobs, padding_mask[micro_batch_inds], INVALID_LOGPROB
                            )
                            new_ratio = (new_logprobs - mb_logprobs).exp() # torch.Size([1, 53])
                            new_logprobs = new_logprobs.sum(1) # torch.Size([1])
                            mb_logprobs = mb_logprobs.sum(1) # torch.Size([1])
                            logprobs_diff = new_logprobs - mb_logprobs # torch.Size([1])
                            ratio = torch.exp(logprobs_diff) # torch.Size([1])
                            pg_losses = -mb_advantage * ratio # torch.Size([1])
                            pg_losses2 = -mb_advantage * torch.clamp(ratio, 1.0 - args.cliprange, 1.0 + args.cliprange) # torch.Size([1])
                            pg_loss_max = torch.max(pg_losses, pg_losses2) # torch.Size([1])
                            pg_loss = pg_loss_max.mean()
                            loss = pg_loss
                            accelerator.backward(loss)
                            optimizer.step()
                            optimizer.zero_grad()
                            with torch.no_grad():
                                pg_clipfrac = (pg_losses2 > pg_losses).float().mean() # torch.Size([])
                                prob_dist = torch.nn.functional.softmax(logits, dim=-1) # torch.Size([1, 53, 50304])
                                entropy = torch.logsumexp(logits, dim=-1) - torch.sum(prob_dist * logits, dim=-1) # torch.Size([1, 53])
                                approxkl = 0.5 * (logprobs_diff**2).mean() # torch.Size([])
                                approxkl_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = approxkl
                                pg_clipfrac_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = (
                                    pg_clipfrac
                                )
                                pg_loss_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = pg_loss
                                entropy_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = entropy.mean()
                                ratio_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = new_ratio.mean()
                        gradient_accumulation_idx += 1
                    minibatch_idx += 1
                    # del everything and empty cache
                    # fmt: off
                    del (
                        output, logits, new_all_logprobs, new_logprobs,
                        logprobs_diff, ratio, pg_losses, pg_losses2,
                        pg_loss, loss, pg_clipfrac, prob_dist, entropy, approxkl,
                        mb_advantage, mb_responses, mb_query_responses, mb_logprobs,
                    )
                    # fmt: on
                    torch.cuda.empty_cache()
                    
            #################### AFTER RLHF UPDATE ####################
            evaluate_score_after = self.evaluate_score(sampling=False) # torch.Size([16])
                    
            beta_reward = evaluate_score_after - evaluate_score_before # torch.Size([16])
            actor_loss = -(beta_log_probs * beta_reward.detach()) # torch.Size([16])
            actor_loss = actor_loss.mean() # torch.Size([])
            accelerator.backward(actor_loss)
            beta_optimizer.step()
            beta_optimizer.zero_grad()
                    
            with torch.no_grad():
                mean_kl = kl.sum(1).mean()
                mean_entropy = (-logprobs).sum(1).mean()
                mean_non_score_reward = non_score_reward.mean()
                eps = int(self.state.episode / (time.time() - start_time))
                metrics = {}
                metrics["eps"] = eps
                metrics["objective/kl"] = self.accelerator.gather(mean_kl).mean().item() # gather tensors from all processes (across multiple GPUs or devices) during distributed training.
                metrics["objective/entropy"] = self.accelerator.gather(mean_entropy).mean().item()
                metrics["objective/non_score_reward"] = self.accelerator.gather(mean_non_score_reward).mean().item()
                metrics["objective/rlhf_reward"] = self.accelerator.gather(rlhf_reward).mean().item()
                metrics["objective/scores"] = self.accelerator.gather(scores.mean()).mean().item()
                metrics["policy/approxkl_avg"] = self.accelerator.gather(approxkl_stats).mean().item()
                metrics["policy/clipfrac_avg"] = self.accelerator.gather(pg_clipfrac_stats).mean().item()
                metrics["loss/policy_avg"] = self.accelerator.gather(pg_loss_stats).mean().item()
                metrics["val/clipfrac_avg"] = self.accelerator.gather(vf_clipfrac_stats).mean().item()
                metrics["policy/entropy_avg"] = self.accelerator.gather(entropy_stats).mean().item()
                metrics["val/ratio"] = self.accelerator.gather(ratio_stats).mean().item()
                metrics["val/ratio_var"] = self.accelerator.gather(ratio_stats).var().item()
                metrics["val/num_eos_tokens"] = (responses == processing_class.eos_token_id).sum().item()
                metrics["lr"] = self.lr_scheduler.get_last_lr()[0]
                metrics["episode"] = self.state.episode
                self.state.epoch = self.state.episode / (args.rloo_k * self.train_dataset_len)  # used by self.log
                self.log(metrics)
            del kl, mean_kl, mean_entropy, scores

            self.lr_scheduler.step()
            self.state.global_step += 1
            self.control = self.callback_handler.on_step_end(args, self.state, self.control)
            if self.control.should_save:
                self._save_checkpoint(model, trial=None)
                self.control = self.callback_handler.on_save(self.args, self.state, self.control)
            torch.cuda.empty_cache()
            gc.collect()

            # args.num_sample_generations: 10. update: 1. self.sample_generations_freq: 15.
            if args.num_sample_generations > 0 and (update - 1) % self.sample_generations_freq == 0:
                self.generate_completions(sampling=True)

        # HF trainer specifics
        self.control = self.callback_handler.on_train_end(args, self.state, self.control)
        if self.control.should_save:
            self._save_checkpoint(model, trial=None, metrics=None)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

    def generate_completions(self, sampling: bool = False):
        args = self.args
        processing_class = self.processing_class
        generation_config = GenerationConfig(
            max_new_tokens=self.args.response_length,
            temperature=(0.01 + 1e-7),
            top_k=0.0,
            top_p=1.0,
            do_sample=True,
        )

        table = defaultdict(list)
        with unwrap_model_for_generation(self.model, self.accelerator) as unwrapped_model:
            for batch in self.eval_dataloader:
                query = batch["input_ids"] # torch.Size([8, 63])
                with torch.no_grad():
                    context_length = query.shape[1] # 63
                    query_response, _ = batch_generation( # torch.Size([8, 116])
                        unwrapped_model,
                        query,
                        query.shape[0],
                        processing_class.pad_token_id,
                        generation_config,
                    )
                    response = query_response[:, context_length:] # torch.Size([8, 53])
                    postprocessed_response = response # torch.Size([8, 53])
                    if args.stop_token_id is not None:  # handle the edge case when stop_token_id exists but is 0
                        postprocessed_response = truncate_response(
                            args.stop_token_id, processing_class.pad_token_id, response
                        )
                    table["query"].extend(
                        gather_object(processing_class.batch_decode(query, skip_special_tokens=True))
                    )
                    table["model response"].extend(
                        gather_object(processing_class.batch_decode(postprocessed_response))
                    )

                    postprocessed_query_response = torch.cat((query, postprocessed_response), 1) # query: torch.Size([8, 60]). postprocessed_response: torch.Size([8, 53]).
                    _, score, _ = get_reward( # torch.Size([8])
                        self.reward_model, postprocessed_query_response, processing_class.pad_token_id, context_length
                    )
                    table["score"].extend(self.accelerator.gather(score).float().cpu().numpy())

                if sampling:
                    break
        df = pd.DataFrame(table)

        if self.accelerator.is_main_process:
            print_rich_table(df.iloc[0 : 0 + 5])
            if "wandb" in args.report_to:
                import wandb

                if wandb.run is not None:
                    wandb.log({"completions": wandb.Table(dataframe=df)})
                    wandb.log({"val/scores": np.mean(df['score'])})
                    
    def evaluate_score(self, sampling: bool = False):
        args = self.args
        processing_class = self.processing_class
        generation_config = GenerationConfig(
            max_new_tokens=self.args.response_length,
            temperature=(0.01 + 1e-7),
            top_k=0.0,
            top_p=1.0,
            do_sample=True,
        )

        score_list = []
        with unwrap_model_for_generation(self.model, self.accelerator) as unwrapped_model:
            dataloader_batches = list(self.eval_dataloader)
            random.shuffle(dataloader_batches)  # Shuffle the batches
            batch_count = 0

            for batch in dataloader_batches:
                query = batch["input_ids"]  # torch.Size([8, 63])
                with torch.no_grad():
                    context_length = query.shape[1]  # 63
                    query_response, _ = batch_generation(  # torch.Size([8, 116])
                        unwrapped_model,
                        query,
                        query.shape[0],
                        processing_class.pad_token_id,
                        generation_config,
                    )
                    response = query_response[:, context_length:]  # torch.Size([8, 53])
                    postprocessed_response = response  # torch.Size([8, 53])
                    if args.stop_token_id is not None:  # handle the edge case when stop_token_id exists but is 0
                        postprocessed_response = truncate_response(
                            args.stop_token_id, processing_class.pad_token_id, response
                        )

                    postprocessed_query_response = torch.cat((query, postprocessed_response), 1)  # query: torch.Size([8, 60]). postprocessed_response: torch.Size([8, 53]).
                    _, score, _ = get_reward(  # torch.Size([8])
                        self.reward_model, postprocessed_query_response, processing_class.pad_token_id, context_length
                    )
                    score_list.append(score)

                batch_count += 1
                if batch_count == 2 or sampling:
                    break

        score_tensor = torch.cat(score_list, 0)  # torch.Size([16])

        return score_tensor

                    
    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, list[str], None] = None,
    ):
        """
        Creates a draft of a model card using the information available to the `Trainer`.

        Args:
            model_name (`str`, *optional*, defaults to `None`):
                The name of the model.
            dataset_name (`str`, *optional*, defaults to `None`):
                The name of the dataset used for training.
            tags (`str`, `list[str]` or `None`, *optional*, defaults to `None`):
                Tags to be associated with the model card.
        """
        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        tags = tags or []
        if isinstance(tags, str):
            tags = [tags]

        if hasattr(self.model.config, "unsloth_version"):
            tags.append("unsloth")

        citation = textwrap.dedent("""\
        @inproceedings{ahmadian2024back,
            title        = {{Back to Basics: Revisiting REINFORCE-Style Optimization for Learning from Human Feedback in LLMs}},
            author       = {Arash Ahmadian and Chris Cremer and Matthias Gall{\'{e}} and Marzieh Fadaee and Julia Kreutzer and Olivier Pietquin and Ahmet {\"{U}}st{\"{u}}n and Sara Hooker},
            year         = 2024,
            booktitle    = {Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), {ACL} 2024, Bangkok, Thailand, August 11-16, 2024},
            publisher    = {Association for Computational Linguistics},
            pages        = {12248--12267},
            editor       = {Lun{-}Wei Ku and Andre Martins and Vivek Srikumar},
        }""")

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.get_url() if is_wandb_available() and wandb.run is not None else None,
            trainer_name="RLOO",
            trainer_citation=citation,
            paper_title="Back to Basics: Revisiting REINFORCE-Style Optimization for Learning from Human Feedback in LLMs",
            paper_id="2402.14740",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))
