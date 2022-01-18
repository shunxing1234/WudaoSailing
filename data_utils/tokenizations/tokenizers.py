# -*- encoding:utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
from data.vocab_data.special_token import *
from data_utils.vocabs.vocab import Vocab
import unicodedata
import six
from utils.text_process import *

class Tokenizer(object):
    '''
    Basic tokenizer
    '''
    def __init__(self, args):
        """
        self.vocab
        """
        self.Vocab = Vocab(args.vocab_path)
        self.Vocab.load_vocab(args.vocab_path)
        self.vocab = self.Vocab.w2i
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

    def tokenize(self, line):
        tokens = line.replace("\n", "").replace("\t", "").split()
        return tokens

    def convert_tokens_to_ids(self, tokens: 'list[str]'):
        """
        Args：
            tokens: list of tokens
        Output：
            convert each token to its corresponding vocab index
        Example：
            >>> input_tokens = ["吃","饭","了","吗"]
            >>> tokenids = convert_tokens_to_ids(input_tokens)
            >>> tokenids
            [1391, 7649, 749, 1408]
        """
        return convert_by_vocab(self.vocab, tokens)

    def convert_ids_to_tokens(self, ids: 'list[int]'):
        """
        Args：
            ids: a list of token ids
        Output：
            convert each token id to corresponding token in vocab
        Example：
            >>> input_tokenids = [1391, 7649, 749, 1408]
            >>> tokens = convert_ids_to_tokens(input_tokenids)
            >>> tokens
            ["吃","饭","了","吗"]
        """
        return convert_by_vocab(self.inv_vocab, ids)


class CharTokenizer(Tokenizer):
    '''
    tokenizer to char base
    '''
    def __init__(self, args, is_src=True):
        super().__init__(args, is_src)

    def tokenize(self, text, use_vocab=True):
        if use_vocab:
            return [token if token in self.vocab else UNK_TOKEN for token in list(text.strip())]
        else:
            return [token for token in list(text.strip())]


class SpaceTokenizer(Tokenizer):
    '''
    tokenizer to span base with space splitting 
    '''

    def __init__(self, args, is_src=True):
        super().__init__(args, is_src)

    def tokenize(self, text, use_vocab=True):
        if use_vocab:
            return [token if token in self.vocab else UNK_TOKEN for token in text.strip().split(" ")]
        else:
            return [token for token in text.strip().split(" ")]

SPIECE_UNDERLINE = u"▁".encode("utf-8")


def convert_by_vocab(vocab, items):
    output = []
    for item in items:
        output.append(vocab[item] if item in vocab else vocab.get(UNK_TOKEN))
    return output


class BertTokenizer(Tokenizer):
    """Turn text into tokens for BERT"""

    def __init__(self, args, do_lower_case=True):
        super().__init__(args)
        
        self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case)
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab, unk_token=UNK_TOKEN,
                                                          max_input_chars_per_word=args.max_input_chars_per_word)

    def tokenize(self, text:str):
        
        split_tokens = []
        for token in self.basic_tokenizer.tokenize(text):
            for sub_token in self.wordpiece_tokenizer.tokenize(token):
                split_tokens.append(sub_token)
        return split_tokens





class BasicTokenizer(object):
    def __init__(self, do_lower_case=True):
        """Constructs a BasicTokenizer.
        Args:
            do_lower_case: Whether to lower case the input.
        """
        self.do_lower_case = do_lower_case

    def tokenize(self, text):
        """
        clean data and convert input text to list of tokens, which includes:
        1. convert text to unicode
        2. remove illegal symbols
        3. strip
        4. (optional) do lower case
        5. strip accents

        Args:
            text: input text, eg:
        Output:
            output text with sentences seperate by punctuations,eg:['报', '道', '称', '，']
        Example:
             >>> text = '报道称'
             >>> examples = tokenize(text)
             >>> examples
             ['报', '道', '称', '，']

        """
        text = convert_to_unicode(text)
        text = self._clean_text(text)

        # This was added on November 1st, 2018 for the multilingual and Chinese
        # models. This is also applied to the English models now, but it doesn't
        # matter since the English models were not trained on any Chinese data
        # and generally don't have any Chinese data in them (there are Chinese
        # characters in the vocabulary because Wikipedia does have some Chinese
        # words in the English Wikipedia.).
        text = self._tokenize_chinese_chars(text)

        orig_tokens = whitespace_tokenize(text)
        split_tokens = []
        for token in orig_tokens:
            if self.do_lower_case:
                token = token.lower()
                token = self._run_strip_accents(token)
            split_tokens.extend(self._run_split_on_punc(token))

        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        return output_tokens

    def _run_strip_accents(self, text):
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == "Mn":
                continue
            output.append(char)
        return "".join(output)

    def _run_split_on_punc(self, text):
        """split text by punctuations"""
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        return ["".join(x) for x in output]

    def _tokenize_chinese_chars(self, text):
        """add space between Chinese characters so that it can be processed as English texts"""
        output = []
        for char in text:
            
            if is_chinese_char(char):
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    

    def _clean_text(self, text):
        """Checks whether CP is the codepoint of a CJK character."""
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xfffd or is_control(char):
                continue
            if is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)


class WordpieceTokenizer(object):
    """Runs WordPiece tokenziation."""

    def __init__(self, vocab, unk_token="[UNK]", max_input_chars_per_word=200):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, text):
        """
        Tokenizes a piece of text into its word pieces. This uses a greedy longest-match-first algorithm to perform
        tokenization using the given vocabulary.

        For example, :obj:`input = "unaffable"` wil return as output :obj:`["un", "##aff", "##able"]`.

        Args:
          text: A single token or whitespace separated tokens. This should have
            already been passed through `BasicTokenizer`.

        Returns:
          A list of wordpiece tokens.
        """
        text = convert_to_unicode(text)

        output_tokens = []
        for token in whitespace_tokenize(text):
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:    # if the current word length exceeds self.max_input_chars_per_word, it will be viewed as UNKNOWN token
                output_tokens.append(self.unk_token)
                continue
            is_bad = False                    # if is_bad is True, it means current word cannot be split as valid subwords, and it will be viewed as UNKNOWN token
            sub_tokens = []

            start = 0
            while start < len(chars):
                end = len(chars)              # Start from the longest subword
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + six.ensure_str(substr)
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end

            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens



