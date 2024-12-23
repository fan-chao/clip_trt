#!/usr/bin/env python3
import os
import json
import time
import psutil
import logging
import traceback

import torch
import torch2trt
import tensorrt


from packaging.version import Version
from transformers import ChineseCLIPModel, AutoTokenizer
from .utils import AttributeDict, convert_tensor, clip_model_type, trt_model_filename


_clip_text_models = {}


class ChineseCLIPTextModel():
    """
    ChineseCLIP text encoder and tokenizer for generating text embeddings
    """
    ModelCache = {}

    @staticmethod
    def from_pretrained(model="OFA-Sys/chinese-clip-vit-large-patch14-336px", dtype=torch.float16, use_cache=True, **kwargs):
        """
        Load a CLIP or SigLIP text encoder model from HuggingFace Hub or a local checkpoint.
        Will use TensorRT for inference if ``use_tensorrt=True``, otherwise falls back to Transformers.
        """
        if use_cache and model in ChineseCLIPTextModel.ModelCache:
            return ChineseCLIPTextModel.ModelCache[model]

        instance = ChineseCLIPTextModel(model, dtype=dtype, **kwargs)

        if use_cache:
            ChineseCLIPTextModel.ModelCache[model] = instance

        return instance

    def __init__(self, model, dtype=torch.float16, projector=False, use_tensorrt=False, **kwargs):
        model_types = {
            'chinese_clip':  dict(model=ChineseCLIPModel),
        }

        model_type = clip_model_type(model, types=model_types.keys())

        if model_type is None:
            raise ValueError(f"tried loading unrecognized ChineseCLIP model from {model} - supported model types are ChineseCLIP")

        self.config = AttributeDict(name=model, type=model_type, projector=projector)
        self.stats = AttributeDict()

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.stream = None

        self.dtype = dtype
        self.output_dtype = dtype  # still output the embeddings with the requested dtype
        self.embed_cache = {}

        logging.info(f'loading {model_type} text model {model}')

        factory = model_types[model_type]

        self.model = factory['model'].from_pretrained(model, torch_dtype=self.dtype)

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True, trust_remote_code=True)
        except:
            self.tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False, trust_remote_code=True)


        class TextEncoder(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
                self.config = model.config

            def forward(self, input_ids, attention_mask=None, position_ids=None):
                return self.model.get_text_features(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    output_attentions=False,
                    output_hidden_states=False,
                    return_dict=True
                )

        self.model = TextEncoder(self.model)
        self.model.to(dtype=self.dtype, device=self.device).eval()

        logging.debug(f"{model_type} text model {model}\n\n{self.model}")
        logging.debug(f"{self.config.type} text model warmup ({self.config.name})")

        self.embed_text("猫")

        logging.info(f"loaded {model_type} text model {model}")

    def tokenize(self, text, padding='max_length', truncation=True, dtype=torch.int64, return_tensors='pt', return_dict=False, device=None, **kwargs):
        """
        Tokenize the given string and return the encoded token ID's and attention mask (either in a dict or as a tuple).

        Args:
          text (str): the text to tokenize.
          dtype (type): the numpy or torch datatype of the tensor to return.
          return_tensors (str): ``'np'`` to return a `np.ndarray` or ``'pt'`` to return a `torch.Tensor`
          kwargs:  additional arguments forwarded to the HuggingFace `transformers.AutoTokenizer <https://huggingface.co/docs/transformers/main/en/model_doc/auto#transformers.AutoTokenizer>`_ encode function.

        Returns:
          The token ID's with the tensor type as indicated by `return_tensors` (either `'np'` for `np.ndarray`
          or `'pt'` for `torch.Tensor`) and datatype as indicated by `dtype` (by default ``int65``)
        """
        output = self.tokenizer(
            text,
            padding=padding,
            truncation=truncation,
            return_tensors=return_tensors,
            return_attention_mask=True,
            **kwargs
        )

        output.input_ids = convert_tensor(output.input_ids, return_tensors=return_tensors, dtype=dtype, device=device)
        output.attention_mask = convert_tensor(output.attention_mask, return_tensors=return_tensors, dtype=dtype, device=device)

        if return_dict:
            return output
        else:
            return output.input_ids, output.attention_mask

    def embed_tokens(self, tokens, attention_mask=None, return_tensors='pt', stream=None, **kwargs):
        """
        Return the embedding features of the given tokens. The attention mask is typically used as these models use padding.
        """
        with torch.cuda.StreamContext(stream), torch.inference_mode():
            time_begin_enc = time.perf_counter()

            tokens = convert_tensor(tokens, return_tensors='pt', device=self.device)
            attention_mask = convert_tensor(attention_mask, return_tensors='pt', device=self.device)

            if len(tokens.shape) == 1:
                tokens = tokens.unsqueeze(0)

            if attention_mask is not None and len(attention_mask.shape) == 1:
                attention_mask = attention_mask.unsqueeze(0)

            output = self.model(tokens, attention_mask)
            output = convert_tensor(output, return_tensors=return_tensors, device=self.device, dtype=self.output_dtype)

            self.config.input_shape = tokens.shape
            self.config.output_shape = output.shape

        time_end_enc = time.perf_counter()

        self.stats.time = time_end_enc - time_begin_enc
        self.stats.rate = 1.0 / self.stats.time
        self.stats.input_shape = self.config.input_shape
        self.stats.output_shape = self.config.output_shape

        return output

    def embed_text(self, text, use_cache=False, **kwargs):
        """
        Return the embedding features of the given text.
        """
        output = None

        if use_cache:
            output = self.embed_cache.get(text)
            logging.debug(f"{self.config.type} text embedding cache hit `{text}`".replace('\n', '\\n'))

        if output is None:
            tokens, attention_mask = self.tokenize(text, **kwargs)
            output = self.embed_tokens(tokens, attention_mask=attention_mask, **kwargs)
            if use_cache:
                self.embed_cache[text] = output

        return output

    def __call__(self, text, **kwargs):
        if text is None:
            return
        elif isinstance(text, str):
            return self.embed_text(text, **kwargs)
        else:
            return self.embed_tokens(text, **kwargs)


