import logging
import os
import pickle
import random

import paddle
from nltk.tokenize import sent_tokenize
from paddle.io import Dataset
from paddlenlp.transformers import BertTokenizer

from src.utils import paddle_masked_fill

logger = logging.getLogger(__name__)


class wiki_rr_trainset(Dataset):

    def __init__(self, lang_pair, num_neg, neg_val, params):

        self.lang_pair = lang_pair
        self.source_lang = self.lang_pair[:2]
        self.target_lang = self.lang_pair[2:]
        self.num_neg = num_neg
        self.neg_val = neg_val
        self.max_pairs = params.max_pairs

        if os.path.exists("/home/your_user_name"):
            # gpu server
            data_path = f"/home/your_user_name/Multilingual-Wiki/data/{self.lang_pair}"
        else:
            # cluster
            data_path = f"/mnt/home/your_user_name/Multilingual-Wiki/data/{self.lang_pair}"

        with open(f"{data_path}/{self.source_lang}_np.pkl", 'rb') as fin:
            self.source_np = pickle.load(fin)
        with open(f"{data_path}/{self.target_lang}_np.pkl", 'rb') as fin:
            self.target_np = pickle.load(fin)

        self.max_pairs = min(self.max_pairs, len(self.source_np))

        # sample some positive pairs

        source2target = {k: v['p'] for k, v in self.source_np.items()}

        source_to_keep = set(random.sample(list(self.source_np.keys()), self.max_pairs))
        target_to_keep = set([source2target[x] for x in source_to_keep])

        self.source_np = {k: v for k, v in self.source_np.items() if k in source_to_keep}
        self.target_np = {k: v for k, v in self.target_np.items() if k in target_to_keep}

        # we have to consider negative documents, so we cannot use *_to_keep

        source_docs, target_docs = source_to_keep, target_to_keep

        for k, v in self.source_np.items():
            target_docs.add(v["p"])
            target_docs.update(v["n"])
        for k, v in self.target_np.items():
            source_docs.add(v["p"])
            source_docs.update(v["n"])

        with open(f"{data_path}/{self.source_lang}_text.pkl", 'rb') as fin:
            self.source_text = pickle.load(fin)
            self.source_text = {k: v for k, v in self.source_text.items() if k in source_docs}
        with open(f"{data_path}/{self.target_lang}_text.pkl", 'rb') as fin:
            self.target_text = pickle.load(fin)
            self.target_text = {k: v for k, v in self.target_text.items() if k in target_docs}

        logger.info(f"# parallel sections pairs: {len(self.source_np)}")

        # FIXME: might have to do some GC here

        assert len(self.source_np) == len(self.target_np)
        self.flatten_parallel = list(self.source_np.keys())
        self.all_src_sections = list(self.source_text.keys())
        self.all_tgt_sections = list(self.target_text.keys())

    def __len__(self):
        return len(self.source_np)

    def __getitem__(self, idx):

        first_segment_ids, second_segment_ids, queries, documents, y = [], [], [], [], []

        src_id = self.flatten_parallel[idx]
        tgt_id = self.source_np[src_id]["p"]
        note = 0

        # randomly switch source and target

        if random.random() < 0.5:
            # normal order
            note = 0
            first_segment_id, second_segment_id = src_id, tgt_id
            first_segment_ids.append(first_segment_id)
            second_segment_ids.append(second_segment_id)
            first_text = self.source_text[first_segment_id]
            second_text = self.target_text[second_segment_id]
            query = random.choice(sent_tokenize(first_text))
            queries.append(query)
            documents.append(second_text)
            y.append(1)
            neg_second_segment_ids = []
            hard_second_segment_ids = self.source_np[first_segment_id]["n"]
            sampling_prob = (3 / 4) ** len(
                hard_second_segment_ids)  # the probability of sampling an "easy" negative document
            while len(neg_second_segment_ids) < self.num_neg:
                if random.random() < sampling_prob:
                    neg_second_segment_ids.append(random.choice(self.all_tgt_sections))
                else:
                    neg_second_segment_ids.append(random.choice(hard_second_segment_ids))
            for neg_second_segment_id in neg_second_segment_ids:
                first_segment_ids.append(first_segment_id)
                second_segment_ids.append(neg_second_segment_id)
                queries.append(query)
                documents.append(self.target_text[neg_second_segment_id])
                y.append(self.neg_val)
        else:
            # reverse order
            note = 1
            first_segment_id, second_segment_id = tgt_id, src_id
            first_segment_ids.append(first_segment_id)
            second_segment_ids.append(second_segment_id)
            first_text = self.target_text[first_segment_id]
            second_text = self.source_text[second_segment_id]
            query = random.choice(sent_tokenize(first_text))
            queries.append(query)
            documents.append(second_text)
            y.append(1)
            neg_second_segment_ids = []
            hard_second_segment_ids = self.target_np[first_segment_id]["n"]
            sampling_prob = (3 / 4) ** len(
                hard_second_segment_ids)  # the probability of sampling an "easy" negative document
            while len(neg_second_segment_ids) < self.num_neg:
                if random.random() < sampling_prob:
                    neg_second_segment_ids.append(random.choice(self.all_src_sections))
                else:
                    neg_second_segment_ids.append(random.choice(hard_second_segment_ids))
            for neg_second_segment_id in neg_second_segment_ids:
                first_segment_ids.append(first_segment_id)
                second_segment_ids.append(neg_second_segment_id)
                queries.append(query)
                documents.append(self.source_text[neg_second_segment_id])
                y.append(self.neg_val)

        return note, first_segment_ids, second_segment_ids, queries, documents, y


class wiki_qlm_trainset(Dataset):

    def __init__(self, lang_pair, neg_val, params):

        self.lang_pair = lang_pair
        self.source_lang = self.lang_pair[:2]
        self.target_lang = self.lang_pair[2:]
        self.neg_val = neg_val
        self.max_pairs = params.max_pairs

        if os.path.exists("/home/puxuan"):
            # gpu server
            data_path = f"/home/puxuan/Multilingual-Wiki/data/{self.lang_pair}"
        else:
            # asimov cluster
            data_path = f"/mnt/home/puxuan/Multilingual-Wiki/data/{self.lang_pair}"

        with open(f"{data_path}/{self.source_lang}_np.pkl", 'rb') as fin:
            self.source_np = pickle.load(fin)
        with open(f"{data_path}/{self.target_lang}_np.pkl", 'rb') as fin:
            self.target_np = pickle.load(fin)

        self.max_pairs = min(self.max_pairs, len(self.source_np))

        # sample some positive pairs

        source2target = {k: v['p'] for k, v in self.source_np.items()}

        source_to_keep = set(random.sample(list(self.source_np.keys()), self.max_pairs))
        target_to_keep = set([source2target[x] for x in source_to_keep])

        self.source_np = {k: v for k, v in self.source_np.items() if k in source_to_keep}
        self.target_np = {k: v for k, v in self.target_np.items() if k in target_to_keep}

        # we don't have to consider negative documents, so we can use *_to_keep

        with open(f"{data_path}/{self.source_lang}_text.pkl", 'rb') as fin:
            self.source_text = pickle.load(fin)
            self.source_text = {k: v for k, v in self.source_text.items() if k in source_to_keep}
        with open(f"{data_path}/{self.target_lang}_text.pkl", 'rb') as fin:
            self.target_text = pickle.load(fin)
            self.target_text = {k: v for k, v in self.target_text.items() if k in target_to_keep}

        logger.info(f"# parallel sections pairs: {len(self.source_np)}")

        assert len(self.source_np) == len(self.target_np)
        self.flatten_parallel = list(self.source_np.keys())
        self.all_src_sections = list(self.source_text.keys())
        self.all_tgt_sections = list(self.target_text.keys())

    def __len__(self):
        return len(self.source_np)

    def __getitem__(self, idx):

        first_segment_ids, second_segment_ids, queries, documents, y = [], [], [], [], []

        src_id = self.flatten_parallel[idx]
        tgt_id = self.source_np[src_id]["p"]
        note = 0

        # randomly switch source and target

        if random.random() < 0.5:
            # normal order
            note = 0
            first_segment_id, second_segment_id = src_id, tgt_id
            first_segment_ids.append(first_segment_id)
            second_segment_ids.append(second_segment_id)
            first_text = self.source_text[first_segment_id]
            second_text = self.target_text[second_segment_id]
            query = random.choice(sent_tokenize(first_text))
            queries.append(query)
            documents.append(second_text)
            y.append(1)

        else:
            # reverse order
            note = 1
            first_segment_id, second_segment_id = tgt_id, src_id
            first_segment_ids.append(first_segment_id)
            second_segment_ids.append(second_segment_id)
            first_text = self.target_text[first_segment_id]
            second_text = self.source_text[second_segment_id]
            query = random.choice(sent_tokenize(first_text))
            queries.append(query)
            documents.append(second_text)
            y.append(1)

        return note, first_segment_ids, second_segment_ids, queries, documents, y


def wiki_collate(batch):
    '''
    this collate function is to be used when you do not use huggingface trainer
    it returns only organized raw inputs, and you encode them on yourself during training
    '''
    notes, qids, dids, queries, documents, y = [], [], [], [], [], []
    queries = [" ".join(x.split(" ")[:64]) for x in queries]
    for (note, b_qids, b_dids, b_queries, b_documents, b_y) in batch:
        notes.append(note)
        qids.extend(b_qids)
        dids.extend(b_dids)
        queries.extend(b_queries)
        documents.extend(b_documents)
        y.extend(b_y)

    return notes, qids, dids, queries, documents, y


class DataCollatorForRelevanceRanking:
    '''
    this collate class is to be used when you use huggingface trainer
    as the outputs of the __call__ function can be directly used as the input for huggingface model
    '''

    def __init__(
            self,
            tokenizer: BertTokenizer,
            glb_att: bool,
            max_len: int = 512
    ):

        self.tokenizer = tokenizer
        self.max_len = max_len
        self.glb_att = glb_att  # if we need to consider global attention; True for models with LongformerAttention, and False otherwise

        logger.info(f"Creating data collator for relevance ranking:")
        logger.info(f"    max_len = {self.max_len}")
        logger.info(f"    glb_att = {self.glb_att}")

    def __call__(self, examples):

        notes, qids, dids, queries, documents, y = [], [], [], [], [], []

        queries = [" ".join(x.split(" ")[:64]) for x in queries]

        for (note, b_qids, b_dids, b_queries, b_documents, b_y) in examples:
            notes.append(note)
            qids.extend(b_qids)
            dids.extend(b_dids)
            queries.extend(b_queries)
            documents.extend(b_documents)
            y.extend(b_y)

        # change to paddle batch_encode method, returning numpy array
        encoded = self.tokenizer.batch_encode(
            batch_text_or_text_pairs=list(zip(queries, documents)),
            truncation_strategy="longest_first",
            max_seq_len=self.max_len,
            pad_to_max_seq_len=True,
            is_split_into_words=False,
            return_attention_mask=True,
            return_token_type_ids=True
        )
        # encoded = self.tokenizer.batch_encode_plus(
        #     batch_text_or_text_pairs=list(zip(queries, documents)),
        #     truncation="longest_first",
        #     max_length=self.max_len,
        #     padding="max_length",
        #     add_special_tokens=True,
        #     is_pretokenized=False,
        #     return_tensors="pt",
        #     return_attention_mask=True,
        #     return_token_type_ids=True
        # )

        input_ids = paddle.to_tensor([encoded_i["input_ids"] for encoded_i in encoded])
        attention_mask = paddle.to_tensor([encoded_i["attention_mask"] for encoded_i in encoded])
        token_type_ids = paddle.to_tensor([encoded_i["token_type_ids"] for encoded_i in encoded])
        # y = torch.tensor(y).unsqueeze(1)
        y = paddle.to_tensor(y).unsqueeze_(1)

        if self.glb_att:
            attention_mask = 2 * attention_mask - token_type_ids
            # attention_mask[attention_mask < 0] = 0
            attention_mask = paddle.where(attention_mask < 0, paddle.zeros_like(attention_mask), attention_mask)

        # return {
        #     "input_ids": input_ids,
        #     "attention_mask": attention_mask,
        #     "token_type_ids": token_type_ids,
        #     "labels": y
        # }
        return input_ids, attention_mask, token_type_ids, y


class DataCollatorForMaskedQueryPrediction:
    """
    """

    def __init__(
            self,
            tokenizer: BertTokenizer,
            mlm_probability: float = 0.15,
            glb_att: bool = False,
            mask_mode: str = "query",
            max_len: int = 512
    ):

        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability
        self.max_len = max_len
        self.glb_att = glb_att  # if we need to consider global attention; True for models with LongformerAttention, and False otherwise
        self.mask_mode = mask_mode

        logger.info(f"Creating data collator for query language model:")
        logger.info(f"    mlm_prob = {self.mlm_probability}")
        logger.info(f"    max_len = {self.max_len}")
        logger.info(f"    glb_att = {self.glb_att}")

    def __call__(self, examples):

        notes, qids, dids, queries, documents, y = [], [], [], [], [], []

        queries = [" ".join(x.split(" ")[:64]) for x in queries]

        for (note, b_qids, b_dids, b_queries, b_documents, b_y) in examples:
            notes.append(note)
            qids.extend(b_qids)
            dids.extend(b_dids)
            queries.extend(b_queries)
            documents.extend(b_documents)
            y.extend(b_y)

        # return a list of dictionaries, with len=batch_size
        encoded = self.tokenizer.batch_encode(
            batch_text_or_text_pairs=list(zip(queries, documents)),
            truncation_strategy="longest_first",
            max_seq_len=self.max_len,
            pad_to_max_seq_len=True,
            is_split_into_words=False,
            return_attention_mask=True,
            return_special_tokens_mask=True,
            return_token_type_ids=True
        )
        # encoded = self.tokenizer.batch_encode_plus(
        #     batch_text_or_text_pairs=list(zip(queries, documents)),
        #     truncation="longest_first",
        #     max_length=self.max_len,
        #     padding="max_length",
        #     add_special_tokens=True,
        #     is_pretokenized=False,
        #     return_tensors="pt",
        #     return_attention_mask=True,
        #     return_token_type_ids=True
        # )

        input_ids = paddle.to_tensor([encoded_i["input_ids"] for encoded_i in encoded])
        attention_mask = paddle.to_tensor([encoded_i["attention_mask"] for encoded_i in encoded])
        token_type_ids = paddle.to_tensor([encoded_i["token_type_ids"] for encoded_i in encoded])
        special_tokens_mask = paddle.to_tensor([encoded_i["special_tokens_mask"] for encoded_i in encoded])

        labels = input_ids.clone()  # [B x seq_len]
        # probability_matrix = torch.full(labels.shape, self.mlm_probability)
        probability_matrix = paddle.full(labels.shape, self.mlm_probability)

        # no need to mask out special tokens, change corresponding prob entry to 0.0
        # special_tokens_mask = [self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in
        #                        labels.tolist()]
        # probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        probability_matrix = paddle_masked_fill(probability_matrix, special_tokens_mask, 0.0)

        # no need to mask out padding tokens
        if self.tokenizer.pad_token is not None:
            # padding_mask = labels.eq(self.tokenizer.pad_token_id)
            padding_mask = paddle.cast(labels == self.tokenizer.pad_token_id, 'int32')
            # probability_matrix.masked_fill_(padding_mask, value=0.0)
            probability_matrix = paddle_masked_fill(probability_matrix, padding_mask, 0.0)
        else:
            assert False

        if self.mask_mode == "query":
            ## no need to mask out document tokens
            document_mask = (attention_mask == 1).logical_and(token_type_ids == 1)
            # probability_matrix.masked_fill_(document_mask, value=0.0)
            probability_matrix = paddle_masked_fill(probability_matrix, paddle.cast(document_mask, 'int32'), 0.0)
        elif self.mask_mode == "document":
            ## no need to mask out query tokens
            query_mask = (attention_mask == 1).logical_and(token_type_ids == 0)
            # probability_matrix.masked_fill_(query_mask, value=0.0)
            probability_matrix = paddle_masked_fill(probability_matrix, paddle.cast(query_mask, 'int32'), 0.0)
        elif self.mask_mode == "mixed":
            ## query and document can all be masked out
            pass
        else:
            assert False

        # masked_indices = torch.bernoulli(probability_matrix).bool()
        masked_indices = paddle.bernoulli(probability_matrix)
        # labels[~masked_indices] = -100
        # We only compute loss on masked tokens
        labels = paddle_masked_fill(labels, 1 - masked_indices, -100)

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        # indices_replaced = (paddle.bernoulli(paddle.full(labels.shape, 0.8)) != 0) & masked_indices
        # input_ids[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        indices_replaced = (paddle.bernoulli(paddle.full(labels.shape, 0.8)) != 0).logical_and(masked_indices != 0)
        input_ids = paddle_masked_fill(input_ids, paddle.cast(indices_replaced, 'int32'),
                                       self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token))

        # 10% of the time, we replace masked input tokens with random word
        indices_random = (paddle.bernoulli(paddle.full(labels.shape, 0.5)) != 0).logical_and(
            (masked_indices != 0).logical_and(indices_replaced.logical_not()))
        random_words = paddle.randint(low=self.tokenizer.vocab_size, shape=labels.shape, dtype="int64")
        # input_ids[indices_random] = random_words[indices_random]
        input_ids = paddle.where(indices_random, random_words, input_ids)

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged

        if self.glb_att:
            attention_mask = 2 * attention_mask - token_type_ids
            # attention_mask[attention_mask < 0] = 0
            attention_mask = paddle.where(attention_mask < 0, paddle.zeros_like(attention_mask), attention_mask)

        # return {
        #     "input_ids": input_ids,
        #     "attention_mask": attention_mask,
        #     "token_type_ids": token_type_ids,
        #     "labels": labels
        # }
        # print("input_ids shape:", input_ids.shape, ", labels shape:", labels.shape)
        return input_ids, attention_mask, token_type_ids, labels
