import transformers
import torch
import torch.nn.functional as F
from lm_eval.base import BaseLM, LM
from lm_eval import utils
from tqdm import tqdm


class GPTXLM(BaseLM):

    def __init__(self,
                 device='cuda',
                 pretrained='gpt2',
                 batch_size=1,
                 revision='main',
                 subfolder=None,
                 tokenizer=None
    ):
        super().__init__()

        assert isinstance(device, str)
        assert isinstance(pretrained, str)
        assert isinstance(batch_size, int)

        if device:
            self._device = torch.device(device)
        else:
            self._device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.gptx = transformers.AutoModelForCausalLM.from_pretrained(pretrained).to(self.device)
        self.gptx.eval()

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            pretrained if tokenizer is None else tokenizer, revision=revision, subfolder=subfolder)

        assert isinstance(self.tokenizer, (
            transformers.GPT2Tokenizer, transformers.GPT2TokenizerFast,
            transformers.T5Tokenizer, transformers.T5TokenizerFast,
        )), "this tokenizer has not been checked for compatibility yet!"

        self.vocab_size = self.tokenizer.vocab_size

        # if isinstance(self.tokenizer, (transformers.GPT2Tokenizer, transformers.GPT2TokenizerFast)):
        #     assert self.tokenizer.encode('Hallo\n\nHallo') == [5568, 203, 203, 5568], \
        #         self.tokenizer.encode('Hallo\n\nHallo')

        # multithreading and batching
        gpus = torch.cuda.device_count()
        batch_size_per_gpu = batch_size  # todo: adaptive batch size

        self.batch_size_per_gpu = batch_size_per_gpu * gpus

        self.max_generate_tokens = self.max_gen_toks

        # TODO: fix multi-gpu
        # if gpus > 1:
        #     self.gpt2 = nn.DataParallel(self.gpt2)


    @property
    def max_gen_toks(self):
        return 256

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

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

    @property
    def max_length(self):
        try:
            return self.gptx.config.n_ctx
        except AttributeError:
            # gptneoconfig doesn't have n_ctx apparently
            return self.gptx.config.max_position_embeddings

    # @classmethod
    # def create_from_arg_string(cls, arg_string, additional_config={}):
    #     args = utils.simple_parse_args_string(arg_string)
    #     args2 = {k: v for k, v in additional_config.items() if v is not None}
    #     return cls(**args, **args2)

    # def loglikelihood(self, requests):
    #     new_reqs = []
    #     for context, continuation in requests:
    #         if context == "":
    #             # end of text as context
    #             context_enc = [self.eot_token_id]
    #         else:
    #             context_enc = self.tokenizer.encode(context)
    #
    #         continuation_enc = self.tokenizer.encode(continuation)
    #
    #         new_reqs.append(((context, continuation), context_enc, continuation_enc))
    #
    #     return self._loglikelihood_tokens(new_reqs)

    def loglikelihood_rolling(self, requests):
        # TODO: Implement caching once we've confirmed the perplexity implementation
        # TODO: automatic batch size detection for vectorization

        loglikelihoods = []
        with torch.no_grad():
            for string, in tqdm(requests):
                rolling_token_windows = list(map(utils.make_disjoint_window, utils.get_rolling_token_windows(
                    token_list=self.tokenizer.encode(string),
                    prefix_token=self.eot_token_id,
                    max_seq_len=self.max_length,
                    context_len=1,
                )))

                rolling_token_windows = [(None,) + x for x in rolling_token_windows]

                # TODO: extract out this call so it only gets called once and also somehow figure out partial caching for that
                string_nll = self._loglikelihood_tokens(rolling_token_windows, disable_tqdm=True)

                # discard is_greedy
                string_nll = [x[0] for x in string_nll]

                string_nll = sum(string_nll)
                loglikelihoods.append(string_nll)

        return loglikelihoods

    def _loglikelihood_tokens(self, requests, disable_tqdm=False):
        # TODO: implement some kind of efficient-request-middleware that lumps together requests with the same context
        res = []
        with torch.no_grad():

            def _collate(x):
                # the negative sign on len(toks) sorts descending - this has a few advantages:
                # - time estimates will always be over not underestimates, which is more useful for planning
                # - to know the size of a batch when going through the list, you know the first one is always the batch padded context length.
                #   this is useful to simplify the batching logic and more importantly to make automatic adaptive batches much much easier to implement
                # - any OOMs will happen right away rather than near the end

                toks = x[1] + x[2]
                return (-len(toks), tuple(toks))

            # TODO: automatic (variable) batch size detection for vectorization
            reord = utils.Reorderer(requests, _collate)
            for chunk in utils.chunks(tqdm(reord.get_reordered(), disable=disable_tqdm), self.batch_size):
                inps = []
                contlens = []
                inplens = []

                padding_length = None

                # because vectorizing is annoying, we first convert each (context, continuation) pair to padded
                # tensors, then we pack them together into a batch, call the model, and then pick it all apart
                # again because vectorizing is annoying

                for _, context_enc, continuation_enc in chunk:
                    # sanity check
                    assert len(context_enc) > 0
                    assert len(continuation_enc) > 0
                    assert len(continuation_enc) <= self.max_length

                    # how this all works:
                    #          CTX      CONT
                    # inp    0 1 2 3|4 5 6 7 8 9 <- last token is deleted by inp[:, :-1]
                    # gpt2    \               \
                    # logits   1 2 3|4 5 6 7 8 9   <- the ctx half gets tossed out by the [:, -len(continuation_enc):, :self.VOCAB_SIZE] slice
                    # cont_toks      4 5 6 7 8 9

                    # when too long to fit in context, truncate from the left
                    inp = torch.tensor(
                        (context_enc + continuation_enc)[-(self.max_length + 1):][:-1]
                        , dtype=torch.long).to(self.device)
                    inplen, = inp.shape

                    cont = continuation_enc

                    # since in _collate we make sure length is descending, the longest is always the first one.
                    padding_length = padding_length if padding_length is not None else inplen

                    # pad to length
                    inp = torch.cat([
                        inp,  # [seq]
                        torch.zeros(padding_length - inplen, dtype=torch.long).to(inp.device)  # [padding_length - seq]
                    ], dim=0)

                    inps.append(inp.unsqueeze(0))
                    contlens.append(cont)
                    inplens.append(inplen)

                multi_logits = F.log_softmax(self._model_call(torch.cat(inps, dim=0)),
                                             dim=-1).cpu()  # [batch, seq, vocab]

                for (cache_key, _, _), logits, inp, inplen, cont_toks in zip(chunk, multi_logits, inps, inplens,
                                                                             contlens):
                    contlen = len(cont_toks)

                    logits = logits[inplen - contlen:inplen].unsqueeze(0)  # [1, seq, vocab]

                    greedy_tokens = logits.argmax(dim=-1)

                    # cont_toks :: [1, seq]
                    cont_toks = torch.tensor(cont_toks, dtype=torch.long).unsqueeze(0)

                    max_equal = (greedy_tokens == cont_toks).all()

                    # last_token_slice = logits[:, -1, :].squeeze(0).tolist()

                    logits = torch.gather(logits, 2, cont_toks.unsqueeze(-1)).squeeze(-1)  # [1, seq]

                    answer = (float(logits.sum()), bool(max_equal))

                    # partial caching
                    if cache_key is not None:
                        self.cache_hook.add_partial("loglikelihood", cache_key, answer)

                    res.append(answer)

        return reord.get_original(res)

    def _model_call(self, inps):
        """
        inps: a torch tensor of shape [batch, sequence]
        the size of sequence may vary from call to call

        returns: a torch tensor of shape [batch, sequence, vocab] with the
        logits retuned from the model
        """
        return self.gptx(inps)[0][:, :, :self.vocab_size]

    def greedy_until(self, requests):
        # TODO: implement fully general `until` that handles untils that are
        # multiple tokens or that span multiple tokens correctly
        res = []

        def _collate(x):
            toks = self.tokenizer.encode(x[0])
            return (len(toks), x[0])

        reord = utils.Reorderer(requests, _collate)

        if len(reord.get_reordered()[0]) == 5:

            for context, until, do_sample, top_k, max_gen_tokens in tqdm(reord.get_reordered()):
                if isinstance(until, str): until = [until]

                if max_gen_tokens is not None:
                    self.max_generate_tokens = max_gen_tokens

                context_enc = torch.tensor([self.tok_encode(context)[self.max_generate_tokens - self.max_length:]])\
                    .to(self.device)

                primary_until, = self.tok_encode(until[0])

                if top_k is not None:
                    cont = self.gptx.generate(
                        context_enc,
                        max_length=context_enc.shape[1] + self.max_generate_tokens,
                        top_k=top_k,
                        eos_token_id=primary_until,
                        do_sample=do_sample,
                    )
                else:
                    cont = self.gptx.generate(
                        context_enc,
                        max_length=context_enc.shape[1] + self.max_generate_tokens,
                        eos_token_id=primary_until,
                        do_sample=do_sample,
                    )

                s = self.tok_decode(cont[0].tolist()[context_enc.shape[1]:])

                for term in until:
                    s = s.split(term)[0]

                # partial caching
                self.cache_hook.add_partial("greedy_until", (context, until), s)

                res.append(s)
        else:
            for context, until in tqdm(reord.get_reordered()):
                if isinstance(until, str): until = [until]

                context_enc = torch.tensor([self.tokenizer.encode(context)[self.max_generate_tokens
                                                                           - self.max_length:]]).to(self.device)

                primary_until, = self.tokenizer.encode(until[0])

                cont = self.gptx.generate(
                    context_enc,
                    max_length=context_enc.shape[1] + self.max_generate_tokens,
                    eos_token_id=primary_until,
                    do_sample=False,
                )

                s = self.tokenizer.decode(cont[0].tolist()[context_enc.shape[1]:])

                for term in until:
                    s = s.split(term)[0]

                # partial caching
                self.cache_hook.add_partial("greedy_until", (context, until), s)

                res.append(s)

        return reord.get_original(res)


    def _model_generate(self, context, max_length, eos_token_id):
        pass
