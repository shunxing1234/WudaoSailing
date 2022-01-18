
from utils.act_fun import *
from utils.optimizers import *
from utils.adversarial import *


str2act = {"gelu": gelu, "gelu_fast": gelu_fast, "relu": relu, "silu": silu, "linear": linear}

str2optimizer = {"adamw": AdamW, "adafactor": Adafactor}

str2scheduler = {"linear": get_linear_schedule_with_warmup, "cosine": get_cosine_schedule_with_warmup,
                 "cosine_with_restarts": get_cosine_with_hard_restarts_schedule_with_warmup,
                 "polynomial": get_polynomial_decay_schedule_with_warmup,
                 "constant": get_constant_schedule, "constant_with_warmup": get_constant_schedule_with_warmup}

str2adv = {"fgm": FGM, "pgd": PGD}

__all__ = ["gelu", "gelu_fast", "relu", "silu", "linear", "str2act",
           "AdamW", "Adafactor", "str2optimizer",
           "get_linear_schedule_with_warmup", "get_cosine_schedule_with_warmup",
           "get_cosine_with_hard_restarts_schedule_with_warmup",
           "get_polynomial_decay_schedule_with_warmup",
           "get_constant_schedule", "get_constant_schedule_with_warmup", "str2scheduler",
           "FGM", "PGD", "str2adv"]



