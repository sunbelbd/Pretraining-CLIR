import logging
import random

import paddle
from paddle.io import Dataset
from paddlenlp.transformers import BertTokenizer
from tqdm.auto import tqdm

logger = logging.getLogger()


class Retrieval_Trainset(Dataset):
    def __init__(self, id2query, id2doc, rel, split, num_neg, neg_value):

        super().__init__()
        self.rel = rel
        self.split = split
        self.id2doc = id2doc
        self.num_neg = num_neg
        self.neg_val = neg_value
        self.id2query = id2query

        # query ids in training split
        self.query_ids = set(self.split["train"])

        # for training
        self.positive_qd_pairs = []
        for query_id, value in tqdm(self.rel.items()):
            if query_id in self.query_ids and len(value["n"]) >= self.num_neg:
                for doc_id in value["p"]:
                    self.positive_qd_pairs.append((query_id, doc_id))

        logger.info("Number of positive query-document pairs in [train] set: {}".format(len(self.positive_qd_pairs)))

    def __len__(self):
        return len(self.positive_qd_pairs)

    def __getitem__(self, idx):

        qids, dids, queries, documents, y = [], [], [], [], []

        # positive qd pair
        tmp = self.positive_qd_pairs[idx]

        if isinstance(tmp, tuple):
            # one positive qd pair
            (q, pos_d) = tmp
            qids.append(q)
            dids.append(pos_d)
            queries.append(self.id2query[q])
            documents.append(self.id2doc[pos_d] if len(self.id2doc[pos_d].strip()) > 0 else "!")
            y.append(1)

            # negative qd pair
            neg_ds = random.sample(self.rel[q]["n"], self.num_neg)
            for neg_d in neg_ds:
                qids.append(q)
                dids.append(neg_d)
                queries.append(self.id2query[q])
                documents.append(self.id2doc[neg_d] if len(self.id2doc[neg_d].strip()) > 0 else ".")
                y.append(self.neg_val)

        elif isinstance(tmp, list):
            # multiple positive qd pairs
            for (q, pos_d) in tmp:
                qids.append(q)
                dids.append(pos_d)
                queries.append(self.id2query[q])
                documents.append(self.id2doc[pos_d] if len(self.id2doc[pos_d].strip()) > 0 else "!")
                y.append(1)

                # negative qd pair
                neg_ds = random.sample(self.rel[q]["n"], self.num_neg)
                for neg_d in neg_ds:
                    qids.append(q)
                    dids.append(neg_d)
                    queries.append(self.id2query[q])
                    documents.append(self.id2doc[neg_d] if len(self.id2doc[neg_d].strip()) > 0 else ".")
                    y.append(self.neg_val)

        else:
            print(type(tmp))
            assert False
        return qids, dids, queries, documents, y


class Retrieval_Testset(Dataset):
    def __init__(self, id2query, id2doc, rel, split, mode, neg_value):

        super().__init__()
        self.rel = rel
        self.mode = mode
        self.split = split
        self.id2doc = id2doc
        self.neg_val = neg_value
        self.id2query = id2query

        # query ids in the split
        self.query_ids = set(self.split[self.mode])

        # for testing
        self.all_qd_pairs = []
        for query_id, value in tqdm(self.rel.items()):
            if query_id in self.query_ids:
                for doc_id in value["p"]:
                    self.all_qd_pairs.append([query_id, doc_id, 1])
                for doc_id in value["n"]:
                    self.all_qd_pairs.append([query_id, doc_id, self.neg_val])

        logger.info("Number of labelled query-document pairs in [{}] set: {}".format(self.mode, len(self.all_qd_pairs)))

    def __len__(self):
        return len(self.all_qd_pairs)

    def __getitem__(self, idx):

        (q, d, y) = self.all_qd_pairs[idx]
        query = self.id2query[q]
        document = self.id2doc[d]

        return q, d, query, document, y


class DataCollatorForTraining:
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
        self.glb_att = glb_att

        logger.info(f"Creating data collator for relevance ranking:")
        logger.info(f"    max_len = {self.max_len}")

    def __call__(self, examples):

        qids, dids, queries, documents, y = [], [], [], [], []

        for (b_qids, b_dids, b_queries, b_documents, b_y) in examples:
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
        qids = [int(x) for x in qids]
        dids = [int(x) for x in dids]
        return qids, dids, input_ids, attention_mask, token_type_ids, y


class DataCollatorForTest:
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
        self.glb_att = glb_att

        logger.info(f"Creating data collator for relevance ranking:")
        logger.info(f"    max_len = {self.max_len}")

    def __call__(self, examples):

        qids, dids, queries, documents, y = [], [], [], [], []

        for (b_qids, b_dids, b_queries, b_documents, b_y) in examples:
            qids.append(b_qids)
            dids.append(b_dids)
            queries.append(b_queries)
            documents.append(b_documents)
            y.append(b_y)

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
        return qids, dids, input_ids, attention_mask, token_type_ids, y

# def train_collate(batch):
#     qids, dids, queries, documents, y = [], [], [], [], []
#     for (b_qids, b_dids, b_queries, b_documents, b_y) in batch:
#         qids.extend(b_qids)
#         dids.extend(b_dids)
#         queries.extend(b_queries)
#         documents.extend(b_documents)
#         y.extend(b_y)
#
#     return qids, dids, queries, documents, y
#
#
# def test_collate(batch):
#     qids, dids, queries, documents, ys = [], [], [], [], []
#     for (q, d, query, document, y) in batch:
#         qids.append(q)
#         dids.append(d)
#         queries.append(query)
#         documents.append(document)
#         ys.append(y)
#
#     return qids, dids, queries, documents, ys
