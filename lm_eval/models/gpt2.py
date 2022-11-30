import os

import transformers
import torch
from lm_eval.base import BaseLM


class HFLM(BaseLM):
    def __init__(
        self,
        device="cuda",
        pretrained="gpt2",
        revision="main",
        subfolder=None,
        tokenizer=None,
        batch_size=1,
        no_tokenizer_check=False,
    ):
        super().__init__()

        assert isinstance(device, str)
        assert isinstance(pretrained, str)
        assert isinstance(batch_size, int)

        if device:
            if device not in ["cuda", "cpu"]:
                device = int(device)
            self._device = torch.device(device)
            print(f"Using device '{device}'")
        else:
            print("Device not specified")
            print(f"Cuda Available? {torch.cuda.is_available()}")
            self._device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )

        # TODO: update this to be less of a hack once subfolder is fixed in HF
        self.gpt2 = transformers.AutoModelForCausalLM.from_pretrained(
            pretrained,
            revision=revision + ("/" + subfolder if subfolder is not None else ""),
        ).to(self.device)
        self.gpt2.eval()

        print(os.path.exists('/Volumes/data/repo/data/bert/gpt2/config.json'))

        # pretrained tokenizer for neo is broken for now so just hard-coding this to gpt2
        print(f'{pretrained=} {revision=} {subfolder=}')
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            pretrained if tokenizer is None else tokenizer,
            # revision=revision,
            # subfolder=subfolder,
        )
        # self.tokenizer = transformers.AutoTokenizer.from_pretrained(
        #     '/Volumes/data/repo/data/bert/gpt2/',
        # )

        assert isinstance(
            self.tokenizer,
            (
                transformers.GPT2Tokenizer,
                transformers.GPT2TokenizerFast,
                transformers.T5Tokenizer,
                transformers.T5TokenizerFast,
                transformers.XGLMTokenizer,
                transformers.XGLMTokenizerFast,
                transformers.BloomTokenizerFast,
            ),
        ), "this tokenizer has not been checked for compatibility yet!"

        self.vocab_size = self.tokenizer.vocab_size

        if no_tokenizer_check:
            print(
                "Tokenizer NOT checked (only do this when NOT using the standard GPT2 tokenizer (English vocab)"
            )
        else:
            if isinstance(
                self.tokenizer,
                (transformers.GPT2Tokenizer, transformers.GPT2TokenizerFast),
            ):
                assert self.tokenizer.encode("hello\n\nhello") == [
                    31373,
                    198,
                    198,
                    31373,
                ], self.tokenizer.encode("hello\n\nhello")

        # multithreading and batching
        self.batch_size_per_gpu = batch_size  # todo: adaptive batch size

        # TODO: fix multi-gpu
        # gpus = torch.cuda.device_count()
        # if gpus > 1:
        #     self.gpt2 = nn.DataParallel(self.gpt2)

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        if self.gpt2.config.model_type == "bloom":
            return self.gpt2.config.seq_length

        try:
            return self.gpt2.config.n_ctx
        except AttributeError:
            # gptneoconfig doesn't have n_ctx apparently
            return self.gpt2.config.max_position_embeddings

    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        # TODO: fix multi-gpu
        return self.batch_size_per_gpu  # * gpus

    @property
    def device(self):
        # TODO: fix multi-gpu
        return self._device

    def tok_encode(self, string: str):
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def _model_call(self, inps):
        """
        inps: a torch tensor of shape [batch, sequence]
        the size of sequence may vary from call to call

        returns: a torch tensor of shape [batch, sequence, vocab] with the
        logits returned from the model
        """
        with torch.no_grad():
            return self.gpt2(inps)[0][:, :, : len(self.tokenizer)]

    def _model_generate(self, context, max_length, eos_token_id):
        res = self.gpt2.generate(
            context, max_length=max_length, eos_token_id=eos_token_id, do_sample=False
        )
        print(f'_model_generate: {res=}')

        return res


# for backwards compatibility
GPT2LM = HFLM
