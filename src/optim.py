# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import re

import paddle


def get_optimizer(parameter_list, s):
    """
    Parse optimizer parameters.
    Input should be of the form:
        - "sgd,lr=0.01"
        - "adagrad,lr=0.1,lr_decay=0.05"
    """
    if "," in s:
        method = s[:s.find(',')]
        optim_params = {}
        for x in s[s.find(',') + 1:].split(','):
            split = x.split('=')
            assert len(split) == 2
            assert re.match("^[+-]?(\d+(\.\d*)?|\.\d+)$", split[1]) is not None
            optim_params[split[0]] = float(split[1])
    else:
        method = s
        optim_params = {}
        optim_params['lr'] = 0.00001
    # AdamOptimizer
    if method == 'adam':
        optim_fn = paddle.optimizer.Adam
    elif method == 'sgd':
        optim_fn = paddle.optimizer.SGD
    elif method == 'adamW':
        optim_fn = paddle.optimizer.AdamW
    # SGD
    else:
        raise Exception("We only support sgd and adam now. Feel free to add yours!")
    assert 'lr' in optim_params
    clip = paddle.nn.ClipGradByNorm(clip_norm=5.0)
    return optim_fn(learning_rate=optim_params['lr'], parameters=parameter_list, grad_clip=clip)

# def get_optimizer(parameters, s):
#     """
#     Parse optimizer parameters.
#     Input should be of the form:
#         - "sgd,lr=0.01"
#         - "adagrad,lr=0.1,lr_decay=0.05"
#     """
#     if "," in s:
#         method = s[:s.find(',')]
#         optim_params = {}
#         for x in s[s.find(',') + 1:].split(','):
#             split = x.split('=')
#             assert len(split) == 2
#             assert re.match("^[+-]?(\d+(\.\d*)?|\.\d+)$", split[1]) is not None
#             optim_params[split[0]] = float(split[1])
#     else:
#         method = s
#         optim_params = {}
#
#     if method == 'adadelta':
#         optim_fn = optim.Adadelta
#     elif method == 'adagrad':
#         optim_fn = optim.Adagrad
#     elif method == 'adam':
#         optim_fn = Adam
#         optim_params['betas'] = (optim_params.get('beta1', 0.9), optim_params.get('beta2', 0.999))
#         optim_params.pop('beta1', None)
#         optim_params.pop('beta2', None)
#     elif method == 'adam_inverse_sqrt':
#         optim_fn = AdamInverseSqrtWithWarmup
#         optim_params['betas'] = (optim_params.get('beta1', 0.9), optim_params.get('beta2', 0.999))
#         optim_params.pop('beta1', None)
#         optim_params.pop('beta2', None)
#     elif method == 'adam_cosine':
#         optim_fn = AdamCosineWithWarmup
#         optim_params['betas'] = (optim_params.get('beta1', 0.9), optim_params.get('beta2', 0.999))
#         optim_params.pop('beta1', None)
#         optim_params.pop('beta2', None)
#     elif method == 'adamax':
#         optim_fn = optim.Adamax
#     elif method == 'asgd':
#         optim_fn = optim.ASGD
#     elif method == 'rmsprop':
#         optim_fn = optim.RMSprop
#     elif method == 'rprop':
#         optim_fn = optim.Rprop
#     elif method == 'sgd':
#         optim_fn = optim.SGD
#         assert 'lr' in optim_params
#     else:
#         raise Exception('Unknown optimization method: "%s"' % method)
#
#     # check that we give good parameters to the optimizer
#     expected_args = inspect.getargspec(optim_fn.__init__)[0]
#     assert expected_args[:2] == ['self', 'params']
#     if not all(k in expected_args[2:] for k in optim_params.keys()):
#         raise Exception('Unexpected parameters: expected "%s", got "%s"' % (
#             str(expected_args[2:]), str(optim_params.keys())))
#
#     return optim_fn(parameters, **optim_params)
