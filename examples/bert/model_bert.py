import torch
import torch.nn as nn
import torch
from nets import *
from nets.encoders import *
from nets.targets import *

class Model(nn.Module):
    """
    Pretraining models consist of three parts:
        - embedding
        - encoder
        - target
    """
    def __init__(self, args, embedding, encoder, target):
        super(Model, self).__init__()
        self.embedding = embedding
        self.encoder = encoder
        self.target = target

        if args.target in ["bert", "mlm", "albert"] and args.tie_weights:
            self.target.mlm_linear_2.weight = self.embedding.word_embedding.weight
        elif args.target in ["lm", "t5", "gsg", "bart"] and args.tie_weights:
            self.target.output_layer.weight = self.embedding.word_embedding.weight

        if args.target in ["t5", "gsg", "bart"] and args.share_embedding:
            self.target.embedding.word_embedding.weight = self.embedding.word_embedding.weight

    def forward(self, src, tgt, seg):
        emb = self.embedding(src, seg)
        output = self.encoder(emb, seg)
        loss_info = self.target(output, tgt)
        return loss_info



    # def build_model(self,args):
    #     """
    #     Build universial encoder representations models.
    #     The combinations of different embedding, encoder,
    #     and target layers yield pretrained models of different
    #     properties.
    #     We could select suitable one for downstream tasks.
    #     """
    #
    #     embedding = str2embedding[args.embedding](args, len(args.tokenizer.vocab))
    #     encoder = str2encoder[args.encoder](args)
    #     if args.target == "seq2seq":
    #         target = str2target[args.target](args, len(args.tgt_tokenizer.vocab))
    #     else:
    #         target = str2target[args.target](args, len(args.tokenizer.vocab))
    #     model = self.Model(args, embedding, encoder, target)
    #
    #     return model
