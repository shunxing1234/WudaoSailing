from data_utils.tokenizations.tokenization_base import BaseTokenizer
from data_utils.vocabs.byte_bpe_vocab import ByteBPEVocab
from data_utils.tokenizations.tokenization_utils import *
import os
import regex as re


class GPT2Tokenizer(BaseTokenizer):
    def __init__(
            self,
            vocab_file,
            merges_file,
            errors="replace",
            unk_token="<|endoftext|>",
            bos_token="<|endoftext|>",
            eos_token="<|endoftext|>",
            **kwargs
    ):
        super().__init__(
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            **kwargs
        )
        if not os.path.isfile(vocab_file):
            raise ValueError(f"Can't find a vocabulary file at path '{vocab_file}'.")
        if not os.path.isfile(merges_file):
            raise ValueError(f"Can't find a merges file at path '{merges_file}'.")
        self.vocab = ByteBPEVocab(vocab_file, merges_file)
        self.errors = errors
        self.cache = {}
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        self.model_input_names = ["input_ids", "token_type_ids"]

    def tokenize(self, text: str) -> 'list[str]':
        bpe_tokens = []
        for token in re.findall(self.pat, text):
            token = "".join(self.vocab.byte_encoder[b] for b in token.encode("utf-8"))
            bpe_tokens.extend(bpe_token for bpe_token in self.bpe(token).split(" "))
        return bpe_tokens

    def _subword_merge(self, tokens: 'list[str]') -> 'list[str]':
        text = "".join(tokens)
        text = bytearray([self.vocab.byte_decoder[c] for c in text]).decode("utf-8", errors=self.errors)
        return text.split()

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        pairs = get_pairs(word)

        if not pairs:
            return token

        while True:
            bigram = min(pairs, key=lambda pair: self.vocab.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.vocab.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                else:
                    new_word.extend(word[i:j])
                    i = j

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = " ".join(word)
        self.cache[token] = word
        return word