from data_utils.tokenizations.tokenization_roberta import RobertaTokenizer


class BartTokenizer(RobertaTokenizer):
    def __init__(self, vocab_file, merges_file):
        super().__init__(vocab_file=vocab_file, merges_file=merges_file)