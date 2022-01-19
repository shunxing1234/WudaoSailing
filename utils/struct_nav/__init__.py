
from utils.act_fun import *
from utils.optimizers import *
from utils.adversarial import *
from dataprocessing.dataloader import *
from dataprocessing.tokenizations.tokenizers import *
from dataprocessing.BertDataset import BertDataset
from nets.BertEmbeddings import WordPosSegEmbedding
from nets.encoders import TransformerEncoder
from nets.targets import BertTarget, AlbertTarget, MlmTarget

str2tokenizer = {"char": CharTokenizer, "space": SpaceTokenizer, "bert": BertTokenizer}
str2dataset = {"bert": BertDataset}
str2dataloader = {"bert": BertDataLoader}

str2encoder = {"transformer": TransformerEncoder}

str2embedding = {"word_pos_seg": WordPosSegEmbedding}

str2target = {"bert": BertTarget, "mlm": MlmTarget, "albert": AlbertTarget}

__all__ = ["gelu", "gelu_fast", "relu", "silu", "linear",
           "AdamW", "Adafactor",
           "get_linear_schedule_with_warmup", "get_cosine_schedule_with_warmup",
           "get_cosine_with_hard_restarts_schedule_with_warmup",
           "get_polynomial_decay_schedule_with_warmup",
           "get_constant_schedule", "get_constant_schedule_with_warmup",
           "FGM", "PGD"]



