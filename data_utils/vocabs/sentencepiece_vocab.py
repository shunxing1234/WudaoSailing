from data_utils.vocabs.vocab import VocabBase
import sentencepiece as spm


class SPVocab(VocabBase):

    def __init__(self, sp_model_file):
        self.sp_model = self.load_vocab(sp_model_file)

    def load_vocab(self, vocab_file):
        sp_model = spm.SentencePieceProcessor()
        sp_model.load(vocab_file)
        return sp_model

    def convert_token_to_id(self, token):
        return self.sp_model.PieceToId(token)

    def convert_id_to_token(self, id):
        return self.sp_model.IdToPiece(id)