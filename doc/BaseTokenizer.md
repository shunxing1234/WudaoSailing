# BaseTokenizer

[TOC]

## 参数

- `bos_token` 
- `eos_token`
- `sep_token`
- `cls_token`
- `unk_token`
- `pad_token`
- `mask_token`
- `additional_special_token`



## 属性

- `bos_token` 
- `eos_token`
- `sep_token`
- `cls_token`
- `unk_token`
- `pad_token`
- `mask_token`
- `special_tokens`



## 方法

### tokenize

 <span id="tokenize">`tokenize(self, text: str) -> List[str]`</span>

**\*抽象方法，所有继承的 tokenizer 类必须实现这个方法**

输入生语料的字符串，输出分词结果，分词结果是一个列表

*不同的 tokenizer 在这一步实现都是不同的，所以把这个方法定义为接口，所有继承这个类的 tokenizer 各自去实现



### encode

```
def encode(
		self,
		text0: str,
		text1: Optional[str] = None,
		add_special_tokens: bool = True,
		padding: bool = False,
		truncation: bool = False,
		max_length: Optional[int] = None
	) -> List[int]:
```

<span id="encode">将输入的字符串（或者字符串对）转化为能够被模型处理的 token id 列表</span>

参数：

- text0：需要进行编码的字符串

- text1：当模型输入是句对的情况时，需要传入第二个输入的字符串（可选参数，如果不传的话默认是None）
- add_special_tokens：是否添加特殊字符，有些模型会对输入的句子添加一些特殊字符，比如 BERT 模型会添加`[CLS]`、`[SEP]`

- padding：是否要对输入句子进行填充（需要参数 max_len）
- truncation：是否要对输入句子进行截断（需要参数 max_len）
- max_length：给输入的句子（对）设定一个最大值，小于这个值进行填充，大于这个值进行截断

调用方法：[_encode](#encode)、[build_inputs_with_special_tokens](#build_inputs_with_special_tokens)

**to do: 填充（padding）和截断（truncation）功能还没实现，打算等所有 tokenizer 子类写完看一下怎么统一实现比较好**



### encode_plus

```
def encode_plus(
        self,
        text0: str,
        text1: Optional[str] = None,
        add_special_tokens: bool = True,
        padding: bool = False,
        truncation: bool = False,
        max_length: Optional[int] = None
    ) -> Dict:
```

<span id="encode_plus">将输入的字符串（或字符串对）转化为能够被模型处理的输入信息（这个输入信息为字典形式），比如 BERT 的输入信息为 `Dict{"input_ids":List[int], "token_type_ids":List[int], "attention_mask":List[int]}`</span>

参数：

- text0：需要进行编码的字符串

- text1：当模型输入是句对的情况时，需要传入第二个输入的字符串（可选参数，如果不传的话默认是None）
- add_special_tokens：是否添加特殊字符，有些模型会对输入的句子添加一些特殊字符，比如 BERT 模型会添加`[CLS]`、`[SEP]`

- padding：是否要对输入句子进行填充（需要参数 max_len）
- truncation：是否要对输入句子进行截断（需要参数 max_len）
- max_length：给输入的句子（对）设定一个最大值，小于这个值进行填充，大于这个值进行截断

调用方法：[encode](#encode)、[_encode](#encode)、[create_token_type_ids_from_sequences](#create_token_type_ids_from_sequences)、[get_pad_attention_mask](#get_pad_attention_mask)

**to do: 填充（padding）和截断（truncation）功能还没实现，打算等所有 tokenizer 子类写完看一下怎么统一实现比较好**



### decode

```
def decode(
		self,
        token_ids	: List[int],
        spaces_between_tokens: bool = True,
        skip_special_tokens: bool = True
    ) -> str:
```

<span id="decode">将 token id 序列重新转回字符串</span>

参数：

- token_ids：需要转为字符串的 token id 序列
- spaces_between_tokens：是否在 token 之间添加空格（中文的字与字之间就不需要中文）
- skip_special_tokens：是否在还原的时候跳过特殊的 token（比如 BERT 中的 `[CLS]`、`[SEP]`）

调用方法：[convert_ids_to_tokens](convert_ids_to_tokens)、[_subword_merge](_subword_merge)、`post_process_decode`



### _encode

<span id="_encode">`_encode(self, text: str) -> List[int]`</span>

将输入的字符串转化为 id 列表

调用方法：[tokenize](#tokenize)、[convert_tokens_to_ids](#convert_tokens_to_ids)

*如果对这一步有特殊需求的 tokenizer 可以在子类中重写这个方法



### convert_tokens_to_ids

 <span id="convert_tokens_to_ids">`convert_tokens_to_ids(self, tokens: List[str]) -> List[int]`</span>

把 token 列表 转换为 id 列表

调用方法：`vocab.convert_token_to_id()`



### convert_ids_to_tokens

<span id="convert_ids_to_tokens">`convert_ids_to_tokens(self, ids: List[int]) -> List[str]`</span>

把 id 列表转换为 token 列表

调用方法：`vocab.convert_id_to_token()`



### build_inputs_with_special_tokens

<span id="build_inputs_with_special_tokens">`def build_inputs_with_special_tokens(self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None) -> List[int]:`</span>

给 token id 序列加上特殊字符

有些模型会对输入的句子添加一些特殊字符，比如 BERT 模型会添加`[CLS]`、`[SEP]`

*基类的这个函数中不添加任何特殊字符，**所有添加特殊字符的 tokenizer 都必须在子类中重写这个方法**



### create_token_type_ids_from_sequences

<span id="create_token_type_ids_from_sequences">`create_token_type_ids_from_sequences(self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None) -> List[int]`</span>

通过输入的字符串（对）序列的 token id 序列获取它的 “token_type_id”

*基类的这个函数中不会考虑任何的特殊字符，**所有添加特殊字符的 tokenizer 都必须在子类中重写这个方法**



### get_pad_attention_mask

<span id="get_pad_attention_mask">`get_pad_attention_mask(self, input_ids: List[int], pad_id: int) -> List[int]`</span>

输入 token id 序列以及填充词的 id，得到一个“attention_mask”序列，用于区分 token id 序列中哪些是填充词（填充词为1，序列中原来的词为0）



### _subword_merge

<span id="_subword_merge">`_subword_merge(self, tokens: List[str]) -> List[str]`</span>

将 token 列表中的子词（比如 WordPiece 方法输出的 “##...” ）合成一个词语

*基类的这个函数直接返回了输入序列，**所有 tokenizer 都必须根据自己的分词方式在子类中重写这个方法**



### @property 方法

这一类方法都返回相应特殊字符的 id

- sep_token_id
- cls_token_id
- pad_token_id