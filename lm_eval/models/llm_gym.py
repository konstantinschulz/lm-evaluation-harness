
from typing import Union, List, Optional
import torch
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, BatchEncoding
from llm_gym.models.gpt2.gpt2_evaluation import PretrainedGPTConfig, PretrainedGPTModel
from .huggingface import AutoCausalLM

TokenSequence = Union[List[int], torch.LongTensor, torch.Tensor, BatchEncoding]


class LLMGym(AutoCausalLM):
    def __init__(self, *args, **kwargs):
        AutoConfig.register("llm_gym_gpt2", PretrainedGPTConfig)
        AutoModelForCausalLM.register(PretrainedGPTConfig, PretrainedGPTModel)
        # TODO load our own tokenizer
        super().__init__(tokenizer="gpt2", *args, **kwargs)

    def _model_call(
        self, inputs: TokenSequence, labels: Optional[TokenSequence] = None
    ) -> TokenSequence:
        return self.model(inputs)
