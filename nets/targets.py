# targets

import torch
import torch.nn as nn
from nets.layer_norm import LayerNorm
from utils.setting_nav import *
 

class MlmTarget(nn.Module):
    """
    BERT exploits masked language modeling (MLM)
    and next sentence prediction (NSP) for pretraining.
    """

    def __init__(self, args, vocab_size):
        super(MlmTarget, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = args.hidden_size
        self.emb_size = args.emb_size
        self.factorized_embedding_parameterization = args.factorized_embedding_parameterization
        self.act = str2act[args.hidden_act]

        if self.factorized_embedding_parameterization:
            self.mlm_linear_1 = nn.Linear(self.hidden_size, self.emb_size)
            self.layer_norm = LayerNorm(self.emb_size)
            self.mlm_linear_2 = nn.Linear(self.emb_size, self.vocab_size)
        else:
            self.mlm_linear_1 = nn.Linear(self.hidden_size, self.hidden_size)
            self.layer_norm = LayerNorm(self.hidden_size)
            self.mlm_linear_2 = nn.Linear(self.hidden_size, self.vocab_size)

        self.softmax = nn.LogSoftmax(dim=-1)

        self.criterion = nn.NLLLoss()
        #self.criterion = nn.CrossEntropyLoss()

    def mlm(self, memory_bank, tgt_mlm):
        # Masked language modeling (MLM) with full softmax prediction.
        output_mlm = self.act(self.mlm_linear_1(memory_bank))
        output_mlm = self.layer_norm(output_mlm)
        if self.factorized_embedding_parameterization:
            output_mlm = output_mlm.contiguous().view(-1, self.emb_size)
        else:
            output_mlm = output_mlm.contiguous().view(-1, self.hidden_size)
        tgt_mlm = tgt_mlm.contiguous().view(-1)
        output_mlm = output_mlm[tgt_mlm > 0, :]
        tgt_mlm = tgt_mlm[tgt_mlm > 0]
        output_mlm = self.mlm_linear_2(output_mlm)
        output_mlm = self.softmax(output_mlm)
        denominator = torch.tensor(output_mlm.size(0) + 1e-6)
        if output_mlm.size(0) == 0:
            correct_mlm = torch.tensor(0.0)
        else:
            correct_mlm = torch.sum((output_mlm.argmax(dim=-1).eq(tgt_mlm)).float())
        loss_mlm = self.criterion(output_mlm, tgt_mlm)
        return output_mlm, loss_mlm, correct_mlm, denominator

    def forward(self, memory_bank, tgt):
        """
        Args:
            memory_bank: [batch_size x seq_length x hidden_size]
            tgt: [batch_size x seq_length]

        Returns:
            loss: Masked language modeling loss.
            correct: Number of words that are predicted correctly.
            denominator: Number of masked words.
        """

        # Masked language model (MLM).
        loss, correct, denominator = self.mlm(memory_bank, tgt)

        return loss, correct, denominator


class BertTarget(MlmTarget):
    """
    BERT exploits masked language modeling (MLM)
    and next sentence prediction (NSP) for pretraining.
    """

    def __init__(self, args, vocab_size):
        super(BertTarget, self).__init__(args, vocab_size)

        # NSP.
        self.nsp_linear_1 = nn.Linear(args.hidden_size, args.hidden_size)
        self.nsp_linear_2 = nn.Linear(args.hidden_size, 2)

    def forward(self, memory_bank, tgt):
        """
        Args:
            memory_bank: [batch_size x seq_length x hidden_size]
            tgt: tuple with tgt_mlm [batch_size x seq_length] and tgt_nsp [batch_size]

        Returns:
            loss_mlm: Masked language model loss.
            loss_nsp: Next sentence prediction loss.
            correct_mlm: Number of words that are predicted correctly.
            correct_nsp: Number of sentences that are predicted correctly.
            denominator: Number of masked words.
        """

        # Masked language model (MLM).
        assert type(tgt) == tuple
        tgt_mlm, tgt_nsp = tgt[0], tgt[1]
        output_mlm, loss_mlm, correct_mlm, denominator = self.mlm(memory_bank, tgt_mlm)

        # Next sentence prediction (NSP).
        output_nsp = torch.tanh(self.nsp_linear_1(memory_bank[:, 0, :]))
        output_nsp = self.nsp_linear_2(output_nsp)
        loss_nsp = self.criterion(self.softmax(output_nsp), tgt_nsp)
        correct_nsp = self.softmax(output_nsp).argmax(dim=-1).eq(tgt_nsp).sum()

        return output_mlm, loss_mlm, loss_nsp, correct_mlm, correct_nsp, denominator, tgt_mlm


class AlbertTarget(MlmTarget):
    """
    BERT exploits masked language modeling (MLM)
    and sentence order prediction (SOP) for pretraining.
    """

    def __init__(self, args, vocab_size):
        super(AlbertTarget, self).__init__(args, vocab_size)

        self.factorized_embedding_parameterization = True
        # SOP.
        self.sop_linear_1 = nn.Linear(args.hidden_size, args.hidden_size)
        self.sop_linear_2 = nn.Linear(args.hidden_size, 2)

    def forward(self, memory_bank, tgt):
        """
        Args:
            memory_bank: [batch_size x seq_length x hidden_size]
            tgt: tuple with tgt_mlm [batch_size x seq_length] and tgt_sop [batch_size]

        Returns:
            loss_mlm: Masked language model loss.
            loss_sop: Sentence order prediction loss.
            correct_mlm: Number of words that are predicted correctly.
            correct_sop: Number of sentences that are predicted correctly.
            denominator: Number of masked words.
        """

        # Masked language model (MLM).
        assert type(tgt) == tuple
        tgt_mlm, tgt_sop = tgt[0], tgt[1]

        output_mlm, loss_mlm, correct_mlm, denominator = self.mlm(memory_bank, tgt_mlm)

        # Sentence order prediction (SOP).
        output_sop = torch.tanh(self.sop_linear_1(memory_bank[:, 0, :]))
        output_sop = self.sop_linear_2(output_sop)
        loss_sop = self.criterion(self.softmax(output_sop), tgt_sop)
        correct_sop = self.softmax(output_sop).argmax(dim=-1).eq(tgt_sop).sum()

        return output_mlm, loss_mlm, loss_sop, correct_mlm, correct_sop, denominator, tgt_mlm