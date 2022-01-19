from dataprocessing.vocabs.vocab import VocabBase
from dataprocessing.vocabs.vocabs_utils import bytes_to_unicode
import json


class ByteBPEVocab(VocabBase):

    def __init__(self, vocab_file, merges_file):
        self.w2i = self.load_vocab(vocab_file)
        self.i2w = {v: k for k, v in self.w2i.items()}
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self.bpe_ranks = self.load_bpe_vocab(merges_file)

    def load_vocab(self, vocab_file):
        with open(vocab_file, encoding="utf-8") as vocab_handle:
            encoder = json.load(vocab_handle)
        return encoder

    def load_bpe_vocab(self, merges_file):
        with open(merges_file, encoding="utf-8") as merges_handle:
            bpe_merges = merges_handle.read().split("\n")[1:-1]
            bpe_merges = [tuple(merge.split()) for merge in bpe_merges]
            bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        return bpe_ranks

    def convert_token_to_id(self, token):
        return self.w2i[token]

    def convert_id_to_token(self, id):
        return self.i2w[id]