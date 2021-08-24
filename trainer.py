import copy
import os
import time
from collections import OrderedDict
from logging import getLogger

import numpy as np
import paddle
from paddle.distributed import fleet
from paddle.io import DataLoader, DistributedBatchSampler

from src.dataset.wiki_dataset import wiki_rr_trainset, wiki_qlm_trainset, DataCollatorForRelevanceRanking, \
    DataCollatorForMaskedQueryPrediction
from src.optim import get_optimizer

logger = getLogger()


class Trainer(object):

    def __init__(self, model, tokenizer, params):
        """
        Initialize trainer.
        """

        self.model = model
        self.params = params

        # epoch / iteration size
        self.epoch_size = params.epoch_size

        # tokenizer
        self.tokenizer = tokenizer

        # data iterators
        self.iterators = {}

        # data collators
        self.rr_collator = DataCollatorForRelevanceRanking(self.tokenizer, "long" in params.model_type,
                                                           params.max_length)
        self.qlm_collator = DataCollatorForMaskedQueryPrediction(self.tokenizer, params.mlm_probability,
                                                                 "long" in params.model_type, params.qlm_mask_mode,
                                                                 params.max_length)

        # set parameters
        self.set_parameters()

        # set optimizers
        self.set_optimizers()

        # float16 / distributed (no AMP)
        if params.multi_gpu:
            logger.info("Using paddle.distributed.fleet ...")
            self.model = fleet.distributed_model(self.model)
            # self.dist_strategy = fleet.DistributedStrategy()
            for opt_name, optimizer in self.optimizers.items():
                self.optimizers[opt_name] = fleet.distributed_optimizer(optimizer)

        # # float16 / distributed (AMP), not implemented for paddle yet
        if params.fp16:
            self.scaler = paddle.amp.GradScaler()

        # training statistics
        self.epoch = 0
        self.n_iter = 0
        self.n_total_iter = 0
        self.n_pairs = 0
        stat_keys = [('processed_p', 0)]
        if params.qlm_steps is not None and params.rr_steps is not None:
            stat_keys += [('QLM+RR-%s' % lang_pair, []) for lang_pair in params.qlm_steps]
        elif params.qlm_steps is not None:
            stat_keys += [('QLM-%s' % lang_pair, []) for lang_pair in params.qlm_steps]
        elif params.rr_steps is not None:
            stat_keys += [('RR-%s' % lang_pair, []) for lang_pair in params.rr_steps]
        else:
            assert False, "Please specify qlm_steps or rr_steps."
        self.stats = OrderedDict(stat_keys)
        stat_keys.pop(0)
        self.epoch_scores = OrderedDict(copy.deepcopy(stat_keys))
        self.last_time = time.time()

    def set_parameters(self):
        """
        Set parameters.
        """
        params = self.params
        self.parameters = {}
        named_params = [(k, p) for k, p in self.model.named_parameters() if not p.stop_gradient]

        # model (excluding memory values)
        self.parameters['model'] = [p for k, p in named_params]

        # log
        for k, v in self.parameters.items():
            logger.info("Found %i parameters in %s." % (len(v), k))
            assert len(v) >= 1

    def set_optimizers(self):
        """
        Set optimizers.
        """
        params = self.params
        self.optimizers = {}

        # model optimizer (excluding memory values)
        self.optimizers['model'] = get_optimizer(self.parameters['model'], params.optimizer)
        # self.optimizers['mlm'] = get_optimizer(self.parameters['model'], params.optimizer)
        # self.optimizers['seqcls'] = get_optimizer(self.parameters['model'], params.optimizer)

        # log
        logger.info("Optimizers: %s" % ", ".join(self.optimizers.keys()))

    def optimize(self, loss, mode='model'):
        """
        Optimize. Not ready for AMP paddle yet
        """
        assert mode in ['mlm', 'seqcls', 'model']
        # check NaN
        if (loss != loss).detach().numpy().any():
            logger.warning("NaN detected")
            exit(1)

        params = self.params

        # optimizers
        # optimizers = [v for k, v in self.optimizers.items()]
        optimizer = self.optimizers[mode]
        # for optimizer in optimizers:
        #    optimizer.clear_grad()

        # regular optimization: already updated for paddle
        if not params.fp16:
            # print("Loss before backward:", loss)
            loss.backward()
            # print("Loss after backward:", loss)
            # check unused parameters
            for name, param in self.model.named_parameters():
                if param.grad is None:
                    print("Find unused parameters with none grad:", name, ", param.stop_gradient:", param.stop_gradient)
                # else:
                #     print("grad is valid:", name, ", param.stop_gradient:", param.stop_gradient)
            optimizer.step()
            # for optimizer in optimizers:
            #     optimizer.step()
            # optimizer.minimize(loss)

        # AMP/FP16 optimization
        else:
            # with paddle.amp.auto_cast():
            scaled_loss = self.scaler.scale(loss)
            scaled_loss.backward()  # do backward
            self.scaler.minimize(optimizer, scaled_loss)
            # for optimizer in optimizers:
            #     self.scaler.minimize(optimizer, scaled_loss)  # update parameters
        # self.model.clear_gradients()
        optimizer.clear_grad()

    def iter(self):
        """
        End of iteration.
        """
        self.n_iter += 1
        self.n_total_iter += 1
        self.print_stats()

    def print_stats(self):
        """
        Print statistics about the training.
        """
        if self.n_total_iter % 5 != 0:
            return

        s_iter = "%7i - " % self.n_total_iter
        s_stat = ' || '.join([
            '{}: {:7.3f}'.format(k, np.mean(v)) for k, v in self.stats.items()
            if type(v) is list and len(v) > 0
        ])
        for k in self.stats.keys():
            if type(self.stats[k]) is list:
                del self.stats[k][:]

        # # learning rates
        # s_lr = " - "
        # for k, v in self.optimizers.items():
        #     s_lr = s_lr + (" - %s LR: " % k) + " / ".join("{:.3e}".format(group['lr']) for group in v.param_groups)

        # processing speed
        new_time = time.time()
        diff = new_time - self.last_time
        p_speed = "{:7.2f} qd pair/s - ".format(
            self.stats['processed_p'] * 1.0 / diff
        )
        self.stats['processed_p'] = 0
        self.last_time = new_time

        # log speed + stats + learning rate
        # logger.info(s_iter + p_speed + s_stat + s_lr)
        logger.info(s_iter + p_speed + s_stat)

    def save_checkpoint(self):
        """
        Save the model / checkpoints.
        """
        if not self.params.is_master:
            return

        # huggingface saves (more useful in our case for finetuning)

        logger.info(f"Saving epoch {self.epoch} ...")
        path = os.path.join(self.params.dump_path, f"paddle-{self.epoch}")
        if not os.path.exists(path): os.makedirs(path)
        model_to_save = self.model._layers if hasattr(self.model, '_layers') else self.model
        # model_to_save = self.model._layers if isinstance(
        #     self.model, paddle.DataParallel) else self.model
        model_to_save.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def end_epoch(self):
        """
        End the epoch.
        """

        # print epoch loss 
        params = self.params
        self.epoch_stat = ' || '.join([
            '{}: {:7.3f}'.format(k, np.mean(v)) for k, v in self.epoch_scores.items()
            if type(v) is list and len(v) > 0
        ])

        for k in self.epoch_scores.keys():
            if type(self.epoch_scores[k]) is list:
                del self.epoch_scores[k][:]

        logger.info("EPOCH LOSS: " + self.epoch_stat)
        if params.is_master:
            self.save_checkpoint()
        self.epoch += 1
        self.n_iter = 0

    def get_iterator(self, obj_name, lang_pair):

        params = self.params

        if obj_name == "rr":
            dataset = wiki_rr_trainset(
                lang_pair=lang_pair,
                num_neg=params.num_neg,
                neg_val=params.neg_val,
                params=params
            )
        elif obj_name == "qlm":
            dataset = wiki_qlm_trainset(
                lang_pair=lang_pair,
                neg_val=params.neg_val,
                params=params
            )
        else:
            assert False

        batch_sampler = DistributedBatchSampler(dataset, batch_size=params.batch_size, shuffle=True, drop_last=True)

        dataloader = DataLoader(
            dataset,
            shuffle=False,
            num_workers=0,
            drop_last=False,
            collate_fn=self.rr_collator if obj_name == "rr" else self.qlm_collator,
            batch_sampler=batch_sampler
        )

        iterator = iter(dataloader)
        self.iterators[(obj_name, lang_pair)] = iterator
        logger.info("Created new training data iterator (%s) ..." % ','.join([str(x) for x in [obj_name, lang_pair]]))

        return iterator

    def get_batch(self, obj_name, lang_pair):

        iterator = self.iterators.get(
            (obj_name, lang_pair),
            None
        )

        if iterator is None:
            iterator = self.get_iterator(obj_name, lang_pair)  # if there is no such iterator, create one

        try:
            x = next(iterator)
        except StopIteration:
            iterator = self.get_iterator(obj_name, lang_pair)
            x = next(iterator)

        return x

    def step(self, lang_pair, lambda_coeff_mlm, lambda_coeff_seqcls):

        assert lambda_coeff_mlm >= 0
        if lambda_coeff_mlm == 0 and lambda_coeff_seqcls == 0:
            return

        params = self.params
        self.model.train()

        # return {
        #     "input_ids": input_ids,
        #     "attention_mask": attention_mask,
        #     "token_type_ids": token_type_ids,
        #     "labels": labels
        # }

        # inputs = self.get_batch("qlm", lang_pair)
        mlm_input_ids, mlm_attention_mask, mlm_token_type_ids, mlm_labels = self.get_batch("qlm", lang_pair)
        # print("mlm_labels shape:", mlm_labels.shape)
        mlm_inputs = {
            "input_ids": mlm_input_ids,
            "attention_mask": mlm_attention_mask,
            "token_type_ids": mlm_token_type_ids,
            "labels": mlm_labels
        }
        # mlm_inputs["mode"] = "mlm"

        # if 'long' in params.model_type:
        #     if self.check_for_long_queries(inputs['attention_mask']):
        #         ## fail the test: long queries detected
        #         logger.info("QLM step skipping long queries")
        #         return

        input_ids, attention_mask, token_type_ids, labels = self.get_batch("rr", lang_pair)
        seqcls_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "labels": labels
        }

        with paddle.amp.auto_cast(enable=params.fp16):
            # loss = self.model(mlm_inputs, seqcls_inputs, lambda_coeff_mlm, lambda_coeff_seqcls)
            loss = self.model(mlm_input=mlm_inputs, seqcls_input=seqcls_inputs, lambda_coeff_mlm=lambda_coeff_mlm,
                              lambda_coeff_seqcls=lambda_coeff_seqcls)[0]

        self.stats[('QLM+RR-%s' % lang_pair)].append(loss.numpy().squeeze())
        self.epoch_scores[('QLM+RR-%s' % lang_pair)].append(loss.numpy().squeeze())
        # optimize
        self.optimize(loss)

        # number of processed sentences / words
        self.stats['processed_p'] += mlm_inputs["attention_mask"].shape[0] + seqcls_inputs["attention_mask"].shape[0]
        self.n_pairs += mlm_inputs["attention_mask"].shape[0] + seqcls_inputs["attention_mask"].shape[0]
        # print("QLM loss:", loss)
        return loss

    def qlm_step(self, lang_pair, lambda_coeff):

        assert lambda_coeff >= 0
        if lambda_coeff == 0:
            return

        params = self.params
        self.model.train()
        # return {
        #     "input_ids": input_ids,
        #     "attention_mask": attention_mask,
        #     "token_type_ids": token_type_ids,
        #     "labels": labels
        # }

        # inputs = self.get_batch("qlm", lang_pair)
        input_ids, attention_mask, token_type_ids, labels = self.get_batch("qlm", lang_pair)
        mlm_input = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "labels": labels
        }
        # if 'long' in params.model_type:
        #     if self.check_for_long_queries(inputs['attention_mask']):
        #         ## fail the test: long queries detected
        #         logger.info("QLM step skipping long queries")
        #         return

        if 'long' in params.model_type:
            mlm_input['attention_mask'] = self.global_attention_safety_check(mlm_input['attention_mask'])

        # inputs = dict_to_cuda(inputs)
        # inputs["mode"] = "mlm"

        with paddle.amp.auto_cast(enable=params.fp16):
            loss = self.model(mlm_input=mlm_input, lambda_coeff_mlm=lambda_coeff)[0]

        self.stats[('QLM-%s' % lang_pair)].append(loss.numpy().squeeze())
        self.epoch_scores[('QLM-%s' % lang_pair)].append(loss.numpy().squeeze())
        # optimize
        self.optimize(loss)

        # number of processed sentences / words
        self.stats['processed_p'] += mlm_input["attention_mask"].shape[0]
        self.n_pairs += mlm_input["attention_mask"].shape[0]
        # print("QLM loss:", loss)
        return loss

    def rr_step(self, lang_pair, lambda_coeff):

        assert lambda_coeff >= 0
        if lambda_coeff == 0:
            return

        params = self.params
        self.model.train()

        # inputs = self.get_batch("rr", lang_pair)

        input_ids, attention_mask, token_type_ids, labels = self.get_batch("rr", lang_pair)
        seq_input = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "labels": labels
        }

        # if 'long' in params.model_type:
        #     if self.check_for_long_queries(inputs['attention_mask']):
        #         ## fail the test: long queries detected
        #         logger.info("RR step skipping long queries")
        #         return

        if 'long' in params.model_type:
            seq_input['attention_mask'] = self.global_attention_safety_check(seq_input['attention_mask'])

        # inputs = dict_to_cuda(inputs)
        # inputs["mode"] = "seqcls"
        with paddle.amp.auto_cast(enable=params.fp16):
            loss = self.model(seqcls_input=seq_input, lambda_coeff_seqcls=lambda_coeff)[0]

        self.stats[('RR-%s' % lang_pair)].append(loss.numpy().squeeze())
        self.epoch_scores[('RR-%s' % lang_pair)].append(loss.numpy().squeeze())
        # optimize
        self.optimize(loss)

        # number of processed sentences / words
        qd_pairs = seq_input["attention_mask"].shape[0]
        pos_qd_pairs = int(qd_pairs / (1 + params.num_neg))
        self.stats['processed_p'] += seq_input["attention_mask"].shape[0]
        self.n_pairs += pos_qd_pairs
        # print("RR loss:", loss)
        return loss

    def check_for_long_queries(self, tensor, length=128):

        ## for models that use longformer attention mechanism
        ## when query is obnormally long, it may cause unusual high GPU memory usage
        ## and therefore program failure
        ## thus, we check for those long queries and skip them!

        ## 07/24/2020: deprecating this method because skipping batches can cause waiting with DDP

        if 'long' not in self.params.model_type:
            assert False, "only check for long queries with mBERT-long!"

        # return any((tensor == 2).sum(dim=1) >= length)
        return paddle.cast((tensor == 2).sum(dim=1) >= length, 'int64').sum() > 0

    def global_attention_safety_check(self, tensor):

        if 'long' not in self.params.model_type:
            return tensor
        else:
            # idxs = ((tensor == 2).sum(dim=1) >= 256).nonzero().squeeze()
            idxs = paddle.nonzero(paddle.cast((tensor == 2).sum(axis=1) >= 256, 'int64')).squeeze()
            if len(idxs.shape) != 0:
                if idxs.shape[0] == 0:
                    return tensor
            else:
                # just one row to replace
                idxs = idxs.unsqueeze(axis=0)

            replacement_attention_mask = paddle.to_tensor([1] * 512 + [0] * 512, dtype="in64")
            for idx in idxs:
                tensor[idx] = replacement_attention_mask
            return tensor
