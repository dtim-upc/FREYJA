#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
from enum import Enum
import torch.nn as nn

class TransformerVersion(Enum):
    PYTORCH_PRETRAINED_BERT = 0
    TRANSFORMERS = 1

TRANSFORMER_VERSION = None

try:
    # Try legacy pytorch_pretrained_bert (very old)
    from pytorch_pretrained_bert.modeling import (
        BertForMaskedLM, BertForPreTraining, BertModel,
        BertConfig,
        BertSelfOutput, BertIntermediate, BertOutput,
        BertLMPredictionHead, BertLayerNorm, gelu
    )
    from pytorch_pretrained_bert.tokenization import BertTokenizer

    hf_flag = 'old'
    TRANSFORMER_VERSION = TransformerVersion.PYTORCH_PRETRAINED_BERT
    logging.warning('You are using the old version of `pytorch_pretrained_bert`.')

except ImportError:
    # Modern Hugging Face transformers (v4+)
    from transformers import (
        BertTokenizer, BertModel, BertForMaskedLM, BertForPreTraining,
        BertConfig
    )
    from transformers.models.bert.modeling_bert import (
        BertSelfOutput, BertIntermediate, BertOutput,
        BertLMPredictionHead
    )

    # Replace missing BertLayerNorm with standard LayerNorm
    BertLayerNorm = nn.LayerNorm

    # Modern versions expose gelu under transformers.activations
    try:
        from transformers.activations import gelu
    except ImportError:
        import torch
        def gelu(x):
            return 0.5 * x * (1 + torch.erf(x / 1.4142))

    hf_flag = 'new'
    TRANSFORMER_VERSION = TransformerVersion.TRANSFORMERS
    logging.info('Using modern Hugging Face `transformers` library.')
