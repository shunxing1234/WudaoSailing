from abc import ABCMeta, abstractmethod
from dataprocessing.tokenizations.tokenization_utils import post_process_decode
from typing import Dict, List, Optional


class BaseTokenizer(metaclass=ABCMeta):

    def __init__(
            self,
            bos_token=None,
            eos_token=None,
            sep_token=None,
            cls_token=None,
            unk_token=None,
            pad_token=None,
            mask_token=None,
            additional_special_tokens=[]
    ):
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.sep_token = sep_token
        self.cls_token = cls_token
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.mask_token = mask_token
        self.special_tokens = [self.bos_token, self.eos_token, self.sep_token, self.cls_token, self.pad_token, self.mask_token] + additional_special_tokens
        # self.special_token_ids = self.get_all_special_token_ids(self.special_tokens)

    @abstractmethod
    def tokenize(self, text: str) -> List[str]:
        pass

    def _encode(self, text: str) -> List[int]:
        return self.convert_tokens_to_ids(self.tokenize(text))

    def encode(
            self,
            text0: str,
            text1: Optional[str] = None,
            add_special_tokens: bool = True,
            padding: bool = False,
            truncation: bool = False,
            max_length: Optional[int] = None
    ) -> List[int]:

        token_ids_0 = self._encode(text0)
        token_ids_1 = self._encode(text1) if text1 else None

        if add_special_tokens:
            sequence = self.build_inputs_with_special_tokens(token_ids_0, token_ids_1)
        else:
            sequence = token_ids_0 + token_ids_1 if text1 else token_ids_0
        return sequence

    def encode_plus(
        self,
        text0: str,
        text1: Optional[str] = None,
        add_special_tokens: bool = True,
        padding: bool = False,
        truncation: bool = False,
        max_length: Optional[int] = None
    ) -> Dict:
        enc = {}
        enc["input_ids"] = self.encode(text0, text1, add_special_tokens, padding, truncation, max_length)
        token_ids_0 = self._encode(text0)
        token_ids_1 = self._encode(text1) if text1 else None
        if "token_type_ids" in self.model_input_names:
            enc["token_type_ids"] = self.create_token_type_ids_from_sequences(token_ids_0, token_ids_1)
        if "attention_mask" in self.model_input_names:
            enc["attention_mask"] = self.get_pad_attention_mask(enc["input_ids"], self.pad_token_id)
        return enc

    def decode(
            self,
            token_ids: List[int],
            spaces_between_tokens: bool = True,
            skip_special_tokens: bool = True
    ) -> str:
        tokens_list = self.convert_ids_to_tokens(token_ids)
        tokens = []
        if skip_special_tokens:
            for token in tokens_list:
                if token not in self.special_tokens:
                    tokens.append(token)
        else:
            tokens = tokens_list
        tokens = self._subword_merge(tokens)

        return post_process_decode(" ".join(tokens) if spaces_between_tokens else "".join(tokens))

    def _subword_merge(self, tokens: List[str]) -> List[str]:
        return tokens

    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        ids = []
        for token in tokens:
            ids.append(self.vocab.convert_token_to_id(token))
        return ids

    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
        tokens = []
        for id in ids:
            tokens.append(self.vocab.convert_id_to_token(id))
        return tokens

    def get_pad_attention_mask(self, input_ids: List[int], pad_id: int) -> List[int]:
        return [0 if id == pad_id else 1 for id in input_ids]

    # def _remove_special_tokens(self, input_ids: List[int], special_tokens: list) -> List[int]:
    #     ids = []
    #     for id in input_ids:
    #         if id not in special_tokens:
    #             ids.append(id)
    #     return ids

    # def get_all_special_token_ids(self, special_token_list):
    #     special_token_ids_list = []
    #     for token in special_token_list:
    #         special_token_ids_list.append(self.vocab.convert_token_to_id(token))
    #     return special_token_ids_list

    def create_token_type_ids_from_sequences(self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None) -> List[int]:
        if token_ids_1 is None:
            return len(token_ids_0) * [0]
        return len(token_ids_0) * [0] + len(token_ids_1) * [1]

    def build_inputs_with_special_tokens(self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None) -> List[int]:
        if token_ids_1 is None:
            return token_ids_0
        return token_ids_0 + token_ids_1

    @property
    def sep_token_id(self):
        return self.vocab.convert_token_to_id(self.sep_token)

    @property
    def cls_token_id(self):
        return self.vocab.convert_token_to_id(self.cls_token)

    @property
    def pad_token_id(self):
        return self.vocab.convert_token_to_id(self.pad_token)
