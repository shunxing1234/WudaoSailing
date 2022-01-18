


from data_utils.tokenizations.tokenization_base import BaseTokenizer
from data_utils.vocabs.sentencepiece_vocab import SPVocab
import os
import unicodedata
from typing import Dict, List, Optional


SENTENCEPIECE_UNDERLINE = "▁"


class AlbertTokenizer(BaseTokenizer):
    def __init__(
            self,
            vocab_file,
            do_lower_case=True,
            remove_space=True,
            keep_accents=False,
            bos_token="[CLS]",
            eos_token="[SEP]",
            unk_token="<unk>",
            sep_token="[SEP]",
            pad_token="<pad>",
            cls_token="[CLS]",
            mask_token="[MASK]",
            **kwargs
    ):
        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
        )
        if not os.path.isfile(vocab_file):
            raise ValueError(f"Can't find a vocabulary file at path '{vocab_file}'.")
        self.vocab = SPVocab(vocab_file)
        self.do_lower_case = do_lower_case
        self.remove_space = remove_space
        self.keep_accents = keep_accents
        self.model_input_names = ["input_ids", "token_type_ids", "attention_mask"]

    def preprocess_text(self, inputs):
        if self.remove_space:
            outputs = " ".join(inputs.strip().split())
        else:
            outputs = inputs
        outputs = outputs.replace("``", '"').replace("''", '"')

        if not self.keep_accents:
            outputs = unicodedata.normalize("NFKD", outputs)
            outputs = "".join([c for c in outputs if not unicodedata.combining(c)])
        if self.do_lower_case:
            outputs = outputs.lower()

        return outputs

    def tokenize(self, text: str) -> 'list[str]':
        text = self.preprocess_text(text)
        pieces = self.vocab.sp_model.encode(text, out_type=str)
        new_pieces = []
        for piece in pieces:
            if len(piece) > 1 and piece[-1] == str(",") and piece[-2].isdigit():
                cur_pieces = self.sp_model.EncodeAsPieces(piece[:-1].replace(SENTENCEPIECE_UNDERLINE, ""))
                if piece[0] != SENTENCEPIECE_UNDERLINE and cur_pieces[0][0] == SENTENCEPIECE_UNDERLINE:
                    if len(cur_pieces[0]) == 1:
                        cur_pieces = cur_pieces[1:]
                    else:
                        cur_pieces[0] = cur_pieces[0][1:]
                cur_pieces.append(piece[-1])
                new_pieces.extend(cur_pieces)
            else:
                new_pieces.append(piece)
        return new_pieces

    def _subword_merge(self, tokens: 'List[str]') -> 'List[str]':
        return self.vocab.sp_model.decode(tokens).split()

    def build_inputs_with_special_tokens(self, token_ids_0: 'List[int]', token_ids_1: 'Optional[List[int]]' = None) -> 'List[int]':
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        if token_ids_1 is None:
            return cls + token_ids_0 + sep
        return cls + token_ids_0 + sep + token_ids_1 + sep

    def create_token_type_ids_from_sequences(self, token_ids_0: 'list[int]', token_ids_1: 'Optional[list[int]]' = None) -> 'list[int]':
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]