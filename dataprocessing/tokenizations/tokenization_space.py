from dataprocessing.tokenizations.tokenization_base import BaseTokenizer
from dataprocessing.vocabs.vocab import Vocab
from dataprocessing.tokenizations.tokenization_utils import *


class SpaceTokenizer(BaseTokenizer):

    def __init__(
            self,
            vocab_path=None,
            unk_token="[UNK]"
    ):
        super().__init__(
            unk_token=unk_token
        )
        self.vocab = None
        if vocab_path:
            self.vocab = Vocab(vocab_path)
        self.model_input_names = ["input_ids"]

    def tokenize(self, text: str) -> 'list[str]':
        if self.vocab:
            return [token if token in self.vocab.w2i else self.unk_token for token in text.strip().split(" ")]
        else:
            return [token for token in text.strip().split(" ")]

