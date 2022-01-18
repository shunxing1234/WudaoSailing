import json



special_tokens_map = {
  "pad_token": "[PAD]",
  "unk_token": "[UNK]",
  "cls_token": "[CLS]",
  "sep_token": "[SEP]",
  "mask_token": "[MASK]",
  "sentinel_token": "<extra_id_0>"
}

UNK_TOKEN = special_tokens_map["unk_token"]
CLS_TOKEN = special_tokens_map["cls_token"]
SEP_TOKEN = special_tokens_map["sep_token"]
MASK_TOKEN = special_tokens_map["mask_token"]
PAD_TOKEN = special_tokens_map["pad_token"]
try:
    SENTINEL_TOKEN = special_tokens_map["sentinel_token"]  # e.g. <extra_id_0>, <extra_id_1>, ... , should have consecutive IDs.
except KeyError:
    pass
