# -*- encoding:utf-8 -*-

from abc import ABCMeta, abstractmethod


class VocabBase(metaclass=ABCMeta):

    @abstractmethod
    def load_vocab(self, vocab_file):
        pass

    @abstractmethod
    def convert_id_to_token(self, id):
        pass

    @abstractmethod
    def convert_token_to_id(self, token):
        pass


class Vocab(VocabBase):
    def __init__(self, vocab_file=None):
        self.w2i = self.load_vocab(vocab_file)
        self.i2w = {v: k for k, v in self.w2i.items()}

    def load_vocab(self, vocab_path=None):
        w2i = {}
        with open(vocab_path, mode="r", encoding="utf-8") as reader:
            for index, line in enumerate(reader):
                w = line.strip("\n").split()[0] if line.strip() else line.strip("\n")
                w2i[w] = index
        return w2i

    def convert_token_to_id(self, token):
        return self.w2i[token]

    def convert_id_to_token(self, id):
        return self.i2w[id]






