from tokenizations.tokenization_base import BaseTokenizer
from tokenizations.tokenization_utils import replace_unicode_punct, remove_non_printing_char, get_pairs, romanian_preprocessing, lowercase_and_remove_accent
from vocabs.bpe_vocab import BPEVocab
import sacremoses as sm
import os
import sys
from typing import List, Optional, Tuple


class XLMTokenizer(BaseTokenizer):
    def __init__(
            self,
            vocab_file,
            merges_file,
            unk_token="<unk>",
            bos_token="<s>",
            sep_token="</s>",
            pad_token="<pad>",
            cls_token="</s>",
            mask_token="<special1>",
            additional_special_tokens=[
                "<special0>",
                "<special1>",
                "<special2>",
                "<special3>",
                "<special4>",
                "<special5>",
                "<special6>",
                "<special7>",
                "<special8>",
                "<special9>",
            ],
            lang2id=None,
            id2lang=None,
            do_lowercase_and_remove_accent=True,
            **kwargs
    ):
        super().__init__(
            unk_token=unk_token,
            bos_token=bos_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            additional_special_tokens=additional_special_tokens,
        )
        if not os.path.isfile(vocab_file):
            raise ValueError(f"Can't find a vocabulary file at path '{vocab_file}'.")
        if not os.path.isfile(merges_file):
            raise ValueError(f"Can't find a merges file at path '{merges_file}'.")

        self.model_input_names = ["input_ids", "token_type_ids", "attention_mask"]
        self.cache_moses_punct_normalizer = dict()
        self.cache_moses_tokenizer = dict()
        self.lang_with_custom_tokenizer = set(["zh", "th", "ja"])
        self.do_lowercase_and_remove_accent = do_lowercase_and_remove_accent
        self.lang2id = lang2id
        self.id2lang = id2lang
        if lang2id is not None and id2lang is not None:
            assert len(lang2id) == len(id2lang)

        self.ja_word_tokenizer = None
        self.zh_word_tokenizer = None

        self.vocab = BPEVocab(vocab_file, merges_file, unk_token)
        self.cache = {}

    def moses_punct_norm(self, text, lang):
        if lang not in self.cache_moses_punct_normalizer:
            punct_normalizer = sm.MosesPunctNormalizer(lang=lang)
            self.cache_moses_punct_normalizer[lang] = punct_normalizer
        else:
            punct_normalizer = self.cache_moses_punct_normalizer[lang]
        return punct_normalizer.normalize(text)

    def moses_tokenize(self, text, lang):
        if lang not in self.cache_moses_tokenizer:
            moses_tokenizer = sm.MosesTokenizer(lang=lang)
            self.cache_moses_tokenizer[lang] = moses_tokenizer
        else:
            moses_tokenizer = self.cache_moses_tokenizer[lang]
        return moses_tokenizer.tokenize(text, return_str=False, escape=False)

    def moses_pipeline(self, text, lang):
        text = replace_unicode_punct(text)
        text = self.moses_punct_norm(text, lang)
        text = remove_non_printing_char(text)
        return text

    def ja_tokenize(self, text):
        if self.ja_word_tokenizer is None:
            try:
                import Mykytea

                self.ja_word_tokenizer = Mykytea.Mykytea(
                    f"-model {os.path.expanduser('~')}/local/share/kytea/model.bin"
                )
            except (AttributeError, ImportError):
                raise ValueError("Make sure you install KyTea (https://github.com/neubig/kytea) and it's python wrapper (https://github.com/chezou/Mykytea-python) with the following steps")

        return list(self.ja_word_tokenizer.getWS(text))

    def bpe(self, token):
        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        if token in self.cache:
            return self.cache[token]
        pairs = get_pairs(word)

        if not pairs:
            return token + "</w>"

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
        if word == "\n  </w>":
            word = "\n</w>"
        self.cache[token] = word
        return word

    def tokenize(self, text: str, lang="en", bypass_tokenizer=False) -> List[str]:
        if lang and self.lang2id and lang not in self.lang2id:
            raise ValueError("Supplied language code not found in lang2id mapping. Please check that your language is supported by the loaded pretrained model.")
        if bypass_tokenizer:
            text = text.split()
        elif lang not in self.lang_with_custom_tokenizer:
            text = self.moses_pipeline(text, lang=lang)
            if lang == "ro":
                text = romanian_preprocessing(text)
            text = self.moses_tokenize(text, lang=lang)
        elif lang == "th":
            text = self.moses_pipeline(text, lang=lang)
            try:
                if "pythainlp" not in sys.modules:
                    from pythainlp.tokenize import word_tokenize as th_word_tokenize
                else:
                    th_word_tokenize = sys.modules["pythainlp"].word_tokenize
            except (AttributeError, ImportError):
                raise ValueError("Make sure you install PyThaiNLP (https://github.com/PyThaiNLP/pythainlp) with the following steps")
            text = th_word_tokenize(text)
        elif lang == "zh":
            try:
                if "jieba" not in sys.modules:
                    import jieba
                else:
                    jieba = sys.modules["jieba"]
            except (AttributeError, ImportError):
                raise ValueError("Make sure you install Jieba (https://github.com/fxsjy/jieba) with the following steps")
            text = " ".join(jieba.cut(text))
            text = self.moses_pipeline(text, lang=lang)
            text = text.split()
        elif lang == "ja":
            text = self.moses_pipeline(text, lang=lang)
            text = self.ja_tokenize(text)
        else:
            raise ValueError("It should not reach here")

        if self.do_lowercase_and_remove_accent and not bypass_tokenizer:
            text = lowercase_and_remove_accent(text)

        split_tokens = []
        for token in text:
            if token:
                split_tokens.extend([t for t in self.bpe(token).split(" ")])

        return split_tokens

    def _subword_merge(self, tokens: List[str]) -> List[str]:
        return "".join(tokens).replace("</w>", " ").strip().split()

    def build_inputs_with_special_tokens(self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None) -> List[int]:
        bos = [self.bos_token_id]
        sep = [self.sep_token_id]

        if token_ids_1 is None:
            return bos + token_ids_0 + sep
        return bos + token_ids_0 + sep + token_ids_1 + sep

    def create_token_type_ids_from_sequences(self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None) -> List[int]:
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

