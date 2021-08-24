import argparse
import copy
import glob
import logging
import os
import pickle
import time
from time import sleep

import numpy as np
import paddle
import pytrec_eval
from paddle.distributed import fleet, get_rank
from paddle.io import DataLoader, DistributedBatchSampler
from paddlenlp.transformers import BertTokenizer

from BertForXLRetrieval import BertForXLRetrieval
from src.dataset.clef_dataset import Retrieval_Trainset, Retrieval_Testset, DataCollatorForTraining, DataCollatorForTest
from src.utils import CustomFormatter, set_seed

# from transformers import BertTokenizer, BertForSequenceClassification
# from transformers import XLMRobertaForSequenceClassification, XLMRobertaTokenizer
# from transformers import XLMTokenizer, XLMForSequenceClassification

parser = argparse.ArgumentParser()

parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')

parser.add_argument('--model_type', choices=["mbert", "mbert-long"])
parser.add_argument('--model_path', type=str, default=None)
parser.add_argument('--source_lang', type=str)
parser.add_argument('--target_lang', type=str)

parser.add_argument("--eval_step", type=int, default=1)
parser.add_argument("--log_step", type=int, default=10)
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--num_neg', type=int, default=1)
parser.add_argument('--num_epochs', type=int, default=20)

# float16 / AMP API
parser.add_argument("--fp16", type=bool, default=False,
                    help="Run model with float16. Not working well under paddle version.")

parser.add_argument('--cased', action='store_true', default=False)

parser.add_argument('--encoder_lr', type=float, default=2e-5)
parser.add_argument('--projector_lr', type=float, default=1e-4)
parser.add_argument('--num_ft_encoders', type=int, default=3, help='Number of encoder layers to finetune')

parser.add_argument('--dataset', type=str, choices=["clef", "wiki-clir", "mix"])

parser.add_argument('--seed', type=int, default=611)
parser.add_argument('--debug', action='store_true', default=False)
parser.add_argument('--full_doc_length', action='store_true', default=True)
args = parser.parse_args()

strategy = fleet.DistributedStrategy()
fleet.init(is_collective=True, strategy=strategy)

# set up logging-related stuff
# slurm_id = os.environ.get('SLURM_JOB_ID')
model_type_in_path = args.model_path
model_type_in_path = model_type_in_path.replace("/", "-")
if model_type_in_path.startswith("-"):
    model_type_in_path = model_type_in_path[1:]

cased_dir = "cased" if args.cased else "uncased"
log_dir = f'logs_paddle/finetune/{args.dataset}/{cased_dir}/{args.model_type}/{args.source_lang}{args.target_lang}'
log_dir += f"/{model_type_in_path}"
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)
handler = logging.FileHandler(f'{log_dir}/process-{get_rank()}.log')
formatter = CustomFormatter('%(adjustedTime)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.info(args)

if args.model_type == "mbert":
    # paddlenlp's save_pretrained and from_pretrained have bugs, bypass it here
    model = BertForXLRetrieval.from_pretrained("bert-base-multilingual-uncased")
    tokenizer = BertTokenizer.from_pretrained(args.model_path)
    pretrained_params = f"{args.model_path}/model_state.pdparams"
    state_dict = paddle.load(pretrained_params)
    model.set_state_dict(state_dict)
    max_len, out_dim = model.bert.embeddings.position_embeddings.weight.shape
    num_encoder = model.bert.encoder.num_layers  # len(model.bert.encoder.layer)

elif args.model_type == "mbert-long":
    print("Paddle version does not support mbert-long option yet")
    exit(1)
else:
    assert False


class CLIR:

    def __init__(self, model):

        self._model = model
        self._rank = get_rank()
        self.best_dev_map, self.best_test_map, self.best_epoch = 0.0, 0.0, 0
        self.eval_interval = args.eval_step
        self.cased_dir = "cased" if args.cased else "uncased"

        self.train_collator = DataCollatorForTraining(tokenizer, "long" in args.model_type, max_len)
        self.test_collator = DataCollatorForTest(tokenizer, "long" in args.model_type, max_len)
        logger.info(f"Evaluating every {self.eval_interval} epochs ...")

        if os.path.exists("/home/your_user_name"):
            home_dir = "/home/your_user_name"  # GPU server
        else:
            home_dir = "/mnt/home/your_user_name"  # cluster

        # read data
        if args.dataset != "mix":

            # if args.dataset == 'clef':

            #     # CLEF data
            #     data_dir = f"{home_dir}/CLIR-project/Evaluation_data/process-clef/{self.cased_dir}"

            #     logger.info(f"reading data from {data_dir}")

            #     rel = pickle.load(open(f"{data_dir}/relevance/{args.target_lang}_rel.pkl", 'rb'))
            #     split = pickle.load(open(f"{data_dir}/relevance/{args.target_lang}_split.pkl", 'rb'))
            #     queries = pickle.load(open(f"{data_dir}/queries/{args.source_lang}_query.pkl", 'rb'))
            #     documents = pickle.load(open(f"{data_dir}/full_documents/{args.target_lang}_document.pkl", 'rb'))

            if args.dataset == "wiki-clir":

                # wiki-CLIR data
                data_dir = f"{home_dir}/wiki-clir/{self.cased_dir}"

                logger.info(f"reading data from {data_dir}")

                rel = pickle.load(open(f"{data_dir}/relevance/{args.target_lang}_rel.pkl", 'rb'))
                split = pickle.load(open(f"{data_dir}/relevance/{args.target_lang}_split.pkl", 'rb'))
                queries = pickle.load(open(f"{data_dir}/queries/{args.source_lang}_query.pkl", 'rb'))
                documents = pickle.load(open(f"{data_dir}/documents/{args.target_lang}_document.pkl", 'rb'))

            else:
                assert False

            # MAP evaluator
            self.qrel = self.get_qrel(rel)
            self.evaluator = pytrec_eval.RelevanceEvaluator(self.qrel, {'map'})

            # dataloaders
            train_dataset = Retrieval_Trainset(id2query=queries, id2doc=documents, rel=rel, split=split,
                                               num_neg=args.num_neg, neg_value=0)
            train_sampler = DistributedBatchSampler(train_dataset, batch_size=args.batch_size, shuffle=True)
            self.train_loader = DataLoader(train_dataset, shuffle=False, num_workers=0, use_shared_memory=False,
                                           collate_fn=self.train_collator, batch_sampler=train_sampler)

            dev_dataset = Retrieval_Testset(id2query=queries, id2doc=documents, rel=rel, split=split, mode="dev",
                                            neg_value=0)
            dev_sampler = DistributedBatchSampler(dev_dataset, batch_size=2 * args.batch_size * (1 + args.num_neg),
                                                  shuffle=False)
            self.dev_loader = DataLoader(dev_dataset, shuffle=False, num_workers=0, use_shared_memory=False,
                                         collate_fn=self.test_collator, batch_sampler=dev_sampler)

            test_dataset = Retrieval_Testset(id2query=queries, id2doc=documents, rel=rel, split=split, mode="test",
                                             neg_value=0)
            test_sampler = DistributedBatchSampler(test_dataset, batch_size=2 * args.batch_size * (1 + args.num_neg),
                                                   shuffle=False)
            self.test_loader = DataLoader(test_dataset, shuffle=False, num_workers=0, use_shared_memory=False,
                                          collate_fn=self.test_collator, batch_sampler=test_sampler)

        else:
            # mixed evaluation
            # train with wiki-clir data, and test on clef data (2fold)

            wiki_data_dir = f"{home_dir}/wiki-clir/{self.cased_dir}"
            rel = pickle.load(open(f"{wiki_data_dir}/relevance/{args.target_lang}_rel.pkl", 'rb'))
            split = pickle.load(open(f"{wiki_data_dir}/relevance/{args.target_lang}_split.pkl", 'rb'))
            queries = pickle.load(open(f"{wiki_data_dir}/queries/{args.source_lang}_query.pkl", 'rb'))
            documents = pickle.load(open(f"{wiki_data_dir}/documents/{args.target_lang}_document.pkl", 'rb'))

            train_dataset = Retrieval_Trainset(id2query=queries, id2doc=documents, rel=rel, split=split,
                                               num_neg=args.num_neg, neg_value=0)
            train_sampler = DistributedBatchSampler(train_dataset, batch_size=args.batch_size, shuffle=True)
            self.train_loader = DataLoader(train_dataset, shuffle=False, num_workers=0,
                                           collate_fn=self.train_collator, batch_sampler=train_sampler)

            clef_data_dir = f"{home_dir}/CLIR-project/Evaluation_data/process-clef/{self.cased_dir}"
            rel = pickle.load(open(f"{clef_data_dir}/relevance/{args.target_lang}_rel.pkl", 'rb'))
            split = pickle.load(
                open(f"{clef_data_dir}/relevance/{args.target_lang}_split_2f.pkl", 'rb'))  # different split file here
            queries = pickle.load(open(f"{clef_data_dir}/queries/{args.source_lang}_query.pkl", 'rb'))
            documents = pickle.load(open(f"{clef_data_dir}/full_documents/{args.target_lang}_document.pkl", 'rb'))

            logger.info(f"reading data from {wiki_data_dir} and {clef_data_dir}")

            dev_dataset = Retrieval_Testset(id2query=queries, id2doc=documents, rel=rel, split=split, mode="f1",
                                            neg_value=0)
            dev_sampler = DistributedBatchSampler(dev_dataset, batch_size=2 * args.batch_size * (1 + args.num_neg),
                                                  shuffle=False)
            self.dev_loader = DataLoader(dev_dataset, shuffle=False, num_workers=0, collate_fn=self.test_collator,
                                         batch_sampler=dev_sampler)

            test_dataset = Retrieval_Testset(id2query=queries, id2doc=documents, rel=rel, split=split, mode="f2",
                                             neg_value=0)
            test_sampler = DistributedBatchSampler(test_dataset, batch_size=2 * args.batch_size * (1 + args.num_neg),
                                                   shuffle=False)
            self.test_loader = DataLoader(test_dataset,
                                          shuffle=False, num_workers=0, collate_fn=self.test_collator,
                                          batch_sampler=test_sampler)

            # MAP evaluator
            self.qrel = self.get_qrel(rel)
            self.evaluator = pytrec_eval.RelevanceEvaluator(self.qrel, {'map'})
            # logger.info(f"f1 has {len(self.dev_loader.dataset.query_ids)} queries ...")
            # logger.info(f"f2 has {len(self.test_loader.dataset.query_ids)} queries ...")
            logger.info(f"f1 has {len(dev_dataset.query_ids)} queries ...")
            logger.info(f"f2 has {len(test_dataset.query_ids)} queries ...")

        logger.info("Data reading done ...")

    def init_weights(self, m):
        """
        change to paddle paddle.framework.ParamAttr
        initializer=paddle.nn.initializer.XavierUniform()
        :param m:
        :return:
        """
        if type(m) == paddle.nn.Linear:
            paddle.nn.initializer.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)

    def run(self):

        set_seed(args.seed)

        self.model = copy.deepcopy(self._model)

        encoder_params = []
        projecter_params = []

        if 'mbert' in args.model_type:

            for param in self.model.bert.embeddings.parameters():
                # param.requires_grad = False
                param.stop_gradient = True
            for l in range(0, num_encoder - args.num_ft_encoders):
                for param in self.model.bert.encoder.layers[l].parameters():
                    # param.requires_grad = False
                    param.stop_gradient = True

            # Here change to paddle layer naming space, taking care of distributed case
            model_to_load = self.model._layers if hasattr(self.model, '_layers') else self.model
            for l in range(num_encoder - args.num_ft_encoders, num_encoder):
                logger.info("adding {}-th encoder to optimizer...".format(l))
                # encoder_params += self.model.module.bert.encoder.layer[l].parameters()
                encoder_params += model_to_load.bert.encoder.layers[l].parameters()
            # encoder_params += self.model.module.bert.pooler.parameters()
            encoder_params += model_to_load.bert.pooler.parameters()

            # projecter_params += self.model.module.seqcls_classifier.parameters()
            projecter_params += self.model.seqcls_classifier.parameters()

            # # Apex DDP
            self.model = fleet.distributed_model(self.model)

        else:
            assert False

        # reset top layers
        for param in encoder_params:
            self.init_weights(param)

        # change to paddle adam optimizer, paddle don't support per-layer learning rates
        # self.optimizer = torch.optim.Adam(parameters=[
        #     {'params': encoder_params},
        #     {'params': projecter_params, 'lr': args.projector_lr}
        # ], lr=args.encoder_lr)
        clip = paddle.nn.ClipGradByNorm(clip_norm=5.0)

        self.optimizer = paddle.optimizer.Adam(parameters=encoder_params + projecter_params, grad_clip=clip,
                                               learning_rate=args.encoder_lr)

        self.optimizers = fleet.distributed_optimizer(self.optimizer)

        # # apex
        # self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level=args.apex_level)

        if args.dataset != "mix":

            for epoch in range(args.num_epochs):

                self.epoch = epoch

                logger.info("process[{}]: training epoch {} ...".format(self._rank, self.epoch))

                self.train()

                if self.epoch % self.eval_interval == 0:
                    # skip half of evaluation for speed
                    # dev eval first: if it is the best ever, we do test
                    logger.info("process[{}]: evaluating epoch {} on dev ...".format(self._rank, self.epoch))
                    with paddle.no_grad():
                        dev_map = self.eval("dev")
                        if dev_map > self.best_dev_map:
                            self.best_dev_map = dev_map
                            self.best_epoch = self.epoch
                            logger.info("process[{}]: evaluating epoch {} on test ...".format(self._rank, self.epoch))
                            self.best_test_map = self.eval("test")
                        else:
                            pass

            logger.info("best test MAP: {:.3f} @ epoch {}".format(self.best_test_map, self.best_epoch))

        else:
            self.f1_maps, self.f2_maps = [], []
            for epoch in range(args.num_epochs):

                self.epoch = epoch

                logger.info("process[{}]: training epoch {} ...".format(self._rank, self.epoch))
                self.train()

                if self.epoch % self.eval_interval == 0:
                    logger.info("process[{}]: evaluating epoch {} on f1 ...".format(self._rank, self.epoch))
                    with paddle.no_grad():
                        dev_map = self.eval("f1")
                        self.f1_maps.append(dev_map)

                    logger.info("process[{}]: evaluating epoch {} on f2 ...".format(self._rank, self.epoch))
                    with paddle.no_grad():
                        test_map = self.eval("f2")
                        self.f2_maps.append(test_map)

            # dist.barrier()

            if self._rank == 0:
                dev_len, test_len = len(self.dev_loader.dataset.query_ids), len(self.test_loader.dataset.query_ids)
                best_f1_map = self.f1_maps[np.argmax(self.f2_maps)]
                best_f2_map = self.f2_maps[np.argmax(self.f1_maps)]
                logger.info(self.f1_maps)
                logger.info(self.f2_maps)
                logger.info(best_f1_map)
                logger.info(best_f2_map)
                best_map = (best_f1_map * dev_len + best_f2_map * test_len) / (dev_len + test_len)
                logger.info(f"best MAP: {best_map:.3f}")

    def train(self):
        self.model.train()
        losses = []
        nw = 0
        n_pos_qd_pair = 0
        t = time.time()

        # chang to paddle case
        if isinstance(self.train_loader.batch_sampler, DistributedBatchSampler):
            self.train_loader.batch_sampler.set_epoch(self.epoch)
        step_cnt = 0
        # for qids, dids, queries, documents, y in self.train_loader:
        for qids, dids, input_ids, attention_mask, token_type_ids, y in self.train_loader:
            n_pos_qd_pair += paddle.cast(y == 1, 'int32').sum() // (1 + args.num_neg)

            # Paddle only accepts array with bool, float16, float32, float64, int8, int16, int32, int64, uint8 or uint16
            # Move batch_encode to collate_fn to convert to numpy arrays then pass to_tensor below.
            # encoded = tokenizer.batch_encode(batch_text_or_text_pairs=list(zip(queries, documents)),
            #                                  truncation_strategy="longest_first", add_special_tokens=True,
            #                                  max_seq_len=max_len, pad_to_max_seq_len=True,
            #                                  is_split_into_words=False, return_attention_mask=True,
            #                                  return_token_type_ids=True)

            # get lengths
            lengths = (max_len - paddle.cast(input_ids == tokenizer.pad_token_id, 'int64')).sum(axis=1)

            # longformer's global attention
            if args.model_type == "mbert-long":
                attention_mask = 2 * attention_mask - token_type_ids
                attention_mask[attention_mask < 0] = 0

            if args.debug:
                # check data
                print(tokenizer.decode(input_ids[0].detach().cpu().tolist()))
                print(tokenizer.decode(input_ids[1].detach().cpu().tolist()))
                print(attention_mask[0].detach().cpu().tolist())
                print(attention_mask[1].detach().cpu().tolist())
                print(token_type_ids[0].detach().cpu().tolist())
                print(token_type_ids[1].detach().cpu().tolist())
                assert False

            self.optimizer.clear_grad()

            seqcls_input = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
                "labels": y,
            }
            with paddle.amp.auto_cast(enable=args.fp16):
                if 'mbert' in args.model_type:
                    outputs = self.model(seqcls_input=seqcls_input)
                else:
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        labels=y
                    )

            loss = outputs[0]

            if paddle.isnan(loss):
                logger.info(tokenizer.decode(input_ids[0].detach().cpu().tolist()))
                logger.info(tokenizer.decode(input_ids[1].detach().cpu().tolist()))
                logger.info(tokenizer.decode(input_ids[2].detach().cpu().tolist()))
                logger.info(tokenizer.decode(input_ids[3].detach().cpu().tolist()))
                logger.info(input_ids.shape)
                logger.info(attention_mask.shape)
                logger.info(token_type_ids.shape)

                logger.info(outputs)

                assert False

            # with amp.scale_loss(loss, self.optimizer) as scaled_loss:
            #     scaled_loss.backward()
            loss.backward()

            self.optimizer.step()

            nw += lengths.sum().numpy()[0]
            losses.append(loss.numpy()[0])
            step_cnt += 1

            # log
            if step_cnt % args.log_step == 0:
                logger.info(
                    f"process[{self._rank}] - epoch {self.epoch} - train iter {n_pos_qd_pair.numpy()[0]} - {nw / (time.time() - t):.1f} words/s - loss: {sum(losses) / len(losses):.4f}")
                nw, t = 0, time.time()
                losses = []

    def eval(self, splt):

        # clean tmp folder for previous round of running
        if self._rank == 0 and self.epoch >= args.eval_step:
            # delete data from all ranks
            for file in glob.glob(
                    f"tmp/{args.model_type}_{args.source_lang}{args.target_lang}_{splt}*{self.epoch - args.eval_step}.txt"):
                logger.info("removing tmp file {}".format(file))
                os.remove(file)

        self.model.eval()

        if splt in ["dev", "f1"]:
            loader = self.dev_loader
        elif splt in ["test", "f2"]:
            loader = self.test_loader
        else:
            assert False

        os.makedirs("tmp", exist_ok=True)
        record_path = f"tmp/{args.model_type}_{args.source_lang}{args.target_lang}_{splt}_{self._rank}_{self.epoch}.txt"
        fout = open(record_path, 'w')

        for qids, dids, input_ids, attention_mask, token_type_ids, y in loader:
            # longformer's global attention
            if args.model_type == "mbert-long":
                attention_mask = 2 * attention_mask - token_type_ids
                attention_mask[attention_mask < 0] = 0

            seqcls_input = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            }

            # outputs = self.model(inputs)

            if 'mbert' in args.model_type:
                outputs = self.model(seqcls_input=seqcls_input)
            else:
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids
                )
            # mlm_input is None and no labels is provided, outputs[0] will be the relevance logit score
            logits = outputs[0][:, 1].detach().cpu().numpy().squeeze()

            if len(qids) > 1:
                for q, d, s in zip(qids, dids, logits):
                    fout.write("{}\t{}\t{}\n".format(q, d, s))
            else:
                fout.write("{}\t{}\t{}\n".format(qids[0], dids[0], logits))

        fout.flush()
        fout.close()

        # dist.barrier()
        # sleep 60 sec so that all other processes have enough time to finish their evaluation
        sleep(45)
        num_record = 0
        run = {}
        for rank in range(paddle.distributed.get_world_size()):
            log_path = f"tmp/{args.model_type}_{args.source_lang}{args.target_lang}_{splt}_{rank}_{self.epoch}.txt"
            with open(log_path, 'r') as fin:
                for line in fin.readlines():
                    q, d, s = line.strip("\n").split("\t")
                    if q not in run:
                        run[q] = {}
                    run[q][d] = float(s)
                    num_record += 1

        if num_record > len(loader.dataset):
            logger.info("{} set during evaluation: {}/{}".format(splt, num_record, len(loader.dataset)))
        results = self.evaluator.evaluate(run)
        mean_ap = np.mean([v["map"] for k, v in results.items()])
        logger.info(
            f"process[{self._rank}] - {splt} - epoch {self.epoch} - mAP: {mean_ap:.3f} w/ {len(results)} queries")
        # dist.barrier()
        return mean_ap

    def get_qrel(self, rel):

        qrel = {}

        for query_id, tmp in rel.items():
            qrel[query_id] = {}
            for pos_doc_id in tmp["p"]:
                qrel[query_id][pos_doc_id] = 1

        return qrel


clir = CLIR(model)
clir.run()
