from data_utils.tokenizations.tokenization_gpt2 import GPT2Tokenizer


class RobertaTokenizer(GPT2Tokenizer):
    def __init__(
            self,
            vocab_file,
            merges_file,
            errors="replace",
            bos_token="<s>",
            eos_token="</s>",
            sep_token="</s>",
            cls_token="<s>",
            unk_token="<unk>",
            pad_token="<pad>",
            mask_token="<mask>"
    ):
        super().__init__(
            vocab_file=vocab_file,
            merges_file=merges_file,
            errors=errors,
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            sep_token=sep_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token
        )

    def build_inputs_with_special_tokens(self, token_ids_0: 'list[int]', token_ids_1: 'Optional[list[int]]' = None) -> 'list[int]':
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        if token_ids_1 is None:
            return cls + token_ids_0 + sep
        return cls + token_ids_0 + sep + sep + token_ids_1 + sep

