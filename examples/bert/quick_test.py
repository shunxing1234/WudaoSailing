# 初始化
import sys
root_dir='/data/wang/models/wudao'
sys.path.append(root_dir)
from dataprocessing.BertDataset import BertDataset
from nets.BertEmbeddings import WordPosSegEmbedding
from nets.encoders import TransformerEncoder
from nets.targets import BertTarget
from model_bert import Model
import glob
from utils.struct_nav import str2tokenizer, str2embedding, str2encoder, str2target
from arguments import *
import training.BertTrainer as trainer

isTEST = 1  # 测试模式，模型会读取./data/train_data/test.json作为输入数据
model_config_path = "./config/tiny_config.json"  # 加载预先调好的模型参数，默认为BERT基础版。如需自行设置参数请将此变量设置为空值""


print("加载config/settings.py里的超参数(如果使用预加载的模型参数，此文件里相关参数会被覆盖)")
args = build_args(root_dir=root_dir)
if model_config_path:
    print("当前模型从%s里加载模型参数" % model_config_path)
    args = load_hyperparam(args, config_path=model_config_path)
if isTEST:
    print("当前模型处于测试模式，会自动加载../data/train_data/test.json作为输入数据")
    args = load_hyperparam(args, config_path="./config/test_config.json")
else:
    print("当前模型处于正常模式")



args.file_types=['json']
if __name__=='__main__':
    print("构建数据集")
    print(args.preprocess_num)
    tokenizer = str2tokenizer[args.tokenizer](args)
    args.tokenizer = tokenizer
    dataset = BertDataset(args, tokenizer.vocab, tokenizer)

    #dataset.build_and_save(1,merge=True,file_preproces_dist=True)
    #
    print("建立模型的结构")
    embedding = str2embedding[args.embedding](args, len(args.tokenizer.vocab))
    encoder = str2encoder[args.encoder](args)
    target = str2target[args.target](args, len(args.tokenizer.vocab))
    model_struct = Model(args, embedding, encoder, target)
    # model_struct
    print("模型总参数量: ")
    get_total_params_num_from_model(model_struct)

    print("训练...")
    trainer.train_and_validate(args, model_struct)
