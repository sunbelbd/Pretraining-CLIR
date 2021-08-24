import warnings

import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn import Layer
from paddlenlp.transformers import BertModel, BertPretrainedModel


# from torch import nn
# from torch.nn import CrossEntropyLoss, MSELoss
# Check BertOnlyMLMHead's correspondence in paddle: BertLMPredictionHead
# from transformers.modeling_bert import BertModel, BertOnlyMLMHead


# from paddlenlp.transformers.bert.modeling import BertLMPredictionHead


class BertPredictionHeadTransform(Layer):
    def __init__(self, hidden_size, activation):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.transform_act_fn = getattr(nn.functional, activation)
        self.LayerNorm = nn.LayerNorm(hidden_size)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        return hidden_states


class BertLMPredictionHead(Layer):
    def __init__(self, hidden_size,
                 vocab_size,
                 activation):
        super().__init__()
        self.transform = BertPredictionHeadTransform(hidden_size, activation)
        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(hidden_size, vocab_size, bias_attr=False)

        self.bias = self.create_parameter(
            shape=[vocab_size], dtype=self.decoder.weight.dtype, is_bias=True)

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class BertOnlyMLMHead(Layer):
    def __init__(self, hidden_size, vocab_size, activation):
        super(BertOnlyMLMHead, self).__init__()
        self.predictions = BertLMPredictionHead(hidden_size, vocab_size, activation)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class BertForXLRetrieval(BertPretrainedModel):
    def __init__(self,
                 vocab_size=105879,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=16,
                 initializer_range=0.02,
                 pad_token_id=0,
                 num_labels=2
                 ):
        # print(type(config), ",", config)
        super(BertForXLRetrieval, self).__init__()
        self.vocab_size = vocab_size
        # print(type(config), ",", config)
        # assert (
        #     not config.is_decoder
        # ), "If you want to use `BertForMaskedLM` make sure `config.is_decoder=False` for bi-directional self-attention."

        # print("vocab size", self.vocab_size)
        # self.bert = BertModel(vocab_size,
        #                       hidden_size=hidden_size,
        #                       num_hidden_layers=num_hidden_layers,
        #                       num_attention_heads=num_attention_heads,
        #                       intermediate_size=intermediate_size,
        #                       hidden_act=hidden_act,
        #                       hidden_dropout_prob=hidden_dropout_prob,
        #                       attention_probs_dropout_prob=attention_probs_dropout_prob,
        #                       max_position_embeddings=max_position_embeddings,
        #                       type_vocab_size=type_vocab_size,
        #                       initializer_range=initializer_range,
        #                       pad_token_id=pad_token_id
        #                       )
        self.bert = BertModel.from_pretrained("bert-base-multilingual-uncased")
        # for mlm: change to paddle
        self.mlm_cls = BertOnlyMLMHead(hidden_size=hidden_size, vocab_size=vocab_size, activation=hidden_act)
        # for seqcls
        self.num_labels = num_labels
        self.seqcls_dropout = nn.Dropout(hidden_dropout_prob)
        self.seqcls_classifier = nn.Linear(hidden_size, num_labels)

        self.init_weights(self.mlm_cls)
        self.init_weights(self.seqcls_classifier)

    # def get_output_embeddings(self):
    #     return self.cls.predictions.decoder

    # paddle bert forward func.

    # def forward(self,
    #             input_ids,
    #             token_type_ids=None,
    #             position_ids=None,
    #             attention_mask=None):
    #     return sequence_output, pooled_output

    # def forward(self, **kwargs):
    #
    #     # FIXME: not sure why parameters are passed like this
    #     # input = input[0]
    #     mode = kwargs['mode']
    #     del kwargs['mode']
    #     # kwargs = input
    #
    #     if mode == "mlm":
    #         return self.forward_mlm(**kwargs)
    #     elif mode == "seqcls":
    #         return self.forward_seqcls(**kwargs)
    #     else:
    #         raise Exception(f"Unknown mode: {mode}")

    def forward(self, mlm_input=None, seqcls_input=None, lambda_coeff_mlm=1.0, lambda_coeff_seqcls=1.0):
        """
        :param lambda_coeff_seqcls:
        :param lambda_coeff_mlm:
        :param mlm_input:
        :param seqcls_input:
        :return:
        """
        assert mlm_input is not None or seqcls_input is not None
        outputs = ()
        loss1, loss2 = 0.0, 0.0
        if mlm_input is not None:
            mlm_outputs = self.forward_mlm(**mlm_input)
            # print(loss1, ", mlm_input[labels]:", mlm_input['labels'].shape)
            if 'labels' in mlm_input:
                loss1 = mlm_outputs[0]
                outputs = (mlm_outputs[1],)
            else:
                outputs = (mlm_outputs[0],)

        if seqcls_input is not None:
            rr_outputs = self.forward_seqcls(**seqcls_input)
            # print(loss2, ", seqcls_input[labels]:", seqcls_input['labels'].shape)
            if 'labels' in seqcls_input:
                loss2 = rr_outputs[0]
                outputs = outputs + (rr_outputs[1],)
            else:
                outputs = outputs + (rr_outputs[0],)
        if (mlm_input is not None and 'labels' in mlm_input) or (seqcls_input is not None and 'labels' in seqcls_input):
            loss = lambda_coeff_mlm * loss1 + lambda_coeff_seqcls * loss2
            outputs = (loss,)  # + outputs return loss only to avoid unnecessary paddle bugs

        # outputs = (loss, mlm_prediction_scores (may or may not), rr_logit_scores (may or may not))
        return outputs

    # Batch data format:
    # {
    #     "mode": 'mlm' or "seqcls"
    #     "input_ids": input_ids,
    #     "attention_mask": attention_mask,
    #     "token_type_ids": token_type_ids,
    #     "labels": labels
    # }

    def forward_mlm(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            labels=None,
            **kwargs
    ):

        if "masked_lm_labels" in kwargs:
            warnings.warn(
                "The `masked_lm_labels` argument is deprecated and will be removed in a future version, use `labels` instead.",
                DeprecationWarning,
            )
            labels = kwargs.pop("masked_lm_labels")
        assert "lm_labels" not in kwargs, "Use `BertWithLMHead` for autoregressive language modeling task."
        assert kwargs == {}, f"Unexpected keyword arguments: {list(kwargs.keys())}."
        # sequence_output, pooled_output
        sequence_output, pooled_output = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids
        )

        # sequence_output = outputs[0]
        prediction_scores = self.mlm_cls(sequence_output)

        outputs = (prediction_scores,)
        # print(prediction_scores.numpy())
        # print("# of tokens to predict", (labels.numpy() != -100).sum(1))
        if labels is not None:
            # loss_fct = CrossEntropyLoss()  # -100 index = padding token
            # masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
            # print(prediction_scores.shape, labels.shape)
            masked_lm_loss = F.cross_entropy(input=prediction_scores.reshape_(shape=(-1, self.vocab_size)),
                                             label=labels.reshape_(shape=(-1,)))
            # print("MLM loss:", masked_lm_loss)
            outputs = (masked_lm_loss,) + outputs

        return outputs  # (masked_lm_loss), prediction_scores, (hidden_states), (attentions)

    def forward_seqcls(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            labels=None
    ):

        sequence_output, pooled_output = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids
        )

        pooled_output = self.seqcls_dropout(pooled_output)
        logits = self.seqcls_classifier(pooled_output)

        outputs = (logits,)

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                # loss_fct = MSELoss()
                loss = F.mse_loss(input=logits.reshape_(shape=(-1,)), label=labels.reshape(shape=(-1,)))
            else:
                # loss_fct = CrossEntropyLoss()
                # loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                loss = F.cross_entropy(input=logits.reshape_(shape=(-1, self.num_labels)),
                                       label=labels.reshape_(shape=(-1,)))
            outputs = (loss,) + outputs

        return outputs
