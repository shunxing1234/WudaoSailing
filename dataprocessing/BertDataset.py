import os
import random
import pickle
from typing import List
from multiprocessing import Pool
from data.vocab_data.special_token import *
from utils.seed import set_seed
from utils.mask import mask_seq
import json
from tqdm import tqdm
import glob
import math
from utils.text_process import count_lines, content2sentences, print_rank_0
from multiprocessing import Queue, Process


def load_file():
    with open(args.corpus_dir+'test.json', mode="r", encoding="utf-8", errors="replace") as f:
        data = json.load(f)
        lines = [e['content'] for e in data[0:len(data)]]
    return lines

def lines2documents(lines):     
    seg_documents=[ content2sentences(line,128) for line in lines]
    documents=[]
    for i in range(len(seg_documents)):
        document = [dataset.tokenizer.convert_tokens_to_ids(dataset.tokenizer.tokenize(line)) for line in seg_documents[i]]
        document = [sentence for sentence in document  if len(sentence) > 0]
        documents.append(document)
    return documents


def merge_dataset(pt_dir,temp_dir, tmp_file_prefix="dataset-tmp-", max_file_size=2**30):

# Merge datasets.

    
    files = glob.glob(temp_dir+tmp_file_prefix+"*.pt")
    size_count=0
    wfile_num=0
    wfile=pt_dir+'dataset-%d.pt' %wfile_num
    dataset_writer = open(wfile, "wb")
    for file in files:
        tmp_dataset_reader = open(file, "rb")
        while True:
            tmp_data = tmp_dataset_reader.read(2**20)
            size_count=size_count+len(tmp_data)
            if tmp_data:
                dataset_writer.write(tmp_data)
            else:
                break
        tmp_dataset_reader.close()
        if file==files[-1]:
            dataset_writer.close()
            return
        if size_count<(max_file_size):
            pass
        else:
            dataset_writer.close()
            size_count=0
            wfile_num=wfile_num+1
            wfile=pt_dir+'dataset-%d.pt' %wfile_num
            dataset_writer = open(wfile, "wb")

    dataset_writer.close()
    return


def truncate_seq_pair(tokens_a: [int], tokens_b: 'list[int]', max_num_tokens: int):
    """
    truncate sequence pair to specific length
    Args:
        tokens_a: the tokens of the first sentence
        tokens_b: the tokens of the second sentence
        max_num_tokens: the maximum number of tokens we allow truncated sequence of tokens_a and tokens_b to have
    """
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_num_tokens:
            break

        # Always delete from the longer sentence
        if len(tokens_a) > len(tokens_b):
            del tokens_a[0]
        else:
            tokens_b.pop()


class Dataset(object):
    def __init__(self, args, vocab, tokenizer):
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.file_types = args.file_types
        self.readers = args.readers
        
        corpus_files=[]
        for corpus_path in glob.glob(args.corpus_dir + "*.*"):
            postfix = corpus_path.rsplit(".", 1)[1]
            if postfix in self.readers and postfix in self.file_types :
                corpus_files.append(corpus_path)
        self.corpus_files=corpus_files
        print('avaiable training files:%d'% len(self.corpus_files))
        self.seq_length = args.seq_length
        self.seed = args.seed
        self.dynamic_masking = args.dynamic_masking
        self.whole_word_masking = args.whole_word_masking
        self.span_masking = args.span_masking
        self.span_geo_prob = args.span_geo_prob
        self.span_max_length = args.span_max_length
        self.docs_buffer_size = args.docs_buffer_size
        self.dup_factor = args.dup_factor
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.pt_dir = args.pt_dir
        self.temp_dir=args.temp_dir
        self.dataset_tmp_prefix = "dataset-tmp-"
        self.split_row = False
        self.is_json = True
        self.process_size = args.process_size
        self.file_preproces_dist=args.file_preproces_dist

    def build_and_save(self, workers_num: int,merge=False,file_preproces_dist=None ):
        
        """
        Build dataset from the given corpus.
        Start workers_num processes and each process deals with a part of data.
        Args:
            workers_num: the number of processors collected for the project
        """
        # lines_num = count_lines(self.corpus_path)
        print("开始使用%d个处理器来建立我们的数据集 ... " % workers_num)
        
        if file_preproces_dist!=None:
            self.file_preproces_dist=file_preproces_dist
        data_size = 0
        assert (workers_num >= 1)
        lines_num = 0
        pool = Pool(workers_num)
       
        count = 0  # index of the output .pt file
        
        
        
        for corpus_path in self.corpus_files:
            postfix = corpus_path.rsplit(".", 1)[1]
            try:
                data = self.readers[postfix](corpus_path)
            except:
                print("读取文件%s失败，注意文件要满足utils.io_utils里面所描述的格式才能自动读取"%corpus_path)
                continue
            process_size = len(data) // workers_num +1
            if self.file_preproces_dist==False:
                for proc_id in range(workers_num):
                    start = proc_id * process_size
                    end = ( proc_id+ 1) * process_size
                    proc_data= data[start:end]
                    pool.apply_async(func=self.worker, args=(proc_data,count,))
                    count=count+1
            else:
                pool.apply_async(func=self.worker, args=(data,count,))
                count=count+1
              
                
        pool.close()
        pool.join()
        # Merge datasets.
        
        if merge:
            merge_dataset(self.pt_dir,self.temp_dir, 
                          tmp_file_prefix=self.dataset_tmp_prefix,
                          max_file_size=2**30)
            
          
        print("预处理部分已完成")
        return int(data_size/self.batch_size*self.epochs)

    def worker(self, proc_id, start, end, ind):
        raise NotImplementedError()


class BertDataset(Dataset):
    """
    Construct dataset for MLM and NSP tasks from the given corpus.
    Each document consists of multiple sentences,
    and each sentence occupies a single line.
    Documents in corpus must be separated by empty lines.
    """

    def __init__(self, args, vocab, tokenizer):
        super(BertDataset, self).__init__(args, vocab, tokenizer)
        self.short_seq_prob = args.short_seq_prob
        
    def create_examples_from_signle_doc(self, documents: 'List[List[List[int]]]', idx: int):
    
        """Creates examples for a single document.

           Args:
            documents:
                All documents loaded from one file and are saved with index type. 
            idx :
                process the idx-th document in all documents.

           Outputs: 
            examples:
              The generated sequences have added  mask, sep, pad and was randomly exchanged for next sentence prediction.

           Example:
              documents: 
                 >>> documents = [[1,2,3],[2,2,1]],[[[1,2,3],[2,2,1]]]
                 >>> idx = 0
                 >>> examples = create_examples_from_signle_doc(documents,idx)

        """

        document=documents[idx]
        max_num_tokens = self.seq_length - 3          # max length of each sequence
        target_seq_length = max_num_tokens               
        if random.random() < self.short_seq_prob:     # random sequence length
            target_seq_length = random.randint(2, max_num_tokens)
        current_chunk = []                               # a list of continuous segments of a document,
        current_length = 0
        i = 0
        examples = []                                   \

        while i < len(document):
            segment = document[i]                        # get a segment
            if not segment:
                i += 1
                continue
            current_chunk.append(segment)                # add a segment to current chunk
            current_length += len(segment)               # overall token length

            # if current length goes to the target length or reaches the end of file, start building token a and b
            if i == len(document) - 1 or current_length >= target_seq_length:
                if current_chunk:
                    a_end = 1 
                    # `a_end` is how many segments from `current_chunk` go into the `A` (first) sentence.
                    # if current chunk has more than 2 sentences, pick part of it `A` (first) sentence
                    if len(current_chunk) >= 2:
                        a_end = random.randint(1, len(current_chunk) - 1)
                    # token a
                    tokens_a = []
                    for j in range(a_end):
                        tokens_a.extend(current_chunk[j])

                    # token b
                    tokens_b = []
                    # switch tokens_a and tokens_b randomly
                    if random.random() > 0.5:   # tokens_b is after tokens_a
                        is_next = True
                        is_random_next=False
                        for j in range(a_end, len(current_chunk)):
                            tokens_b.extend(current_chunk[j])
                    else:
                        is_next = False
                        if random.random() < 0.5:
                            # Randomly generate a sentence from one document to replace the original tokens_b
                            is_random_next = True
                            target_b_length = target_seq_length - len(tokens_a)

                            random_idx = random.randint(0,len(documents)-1)

                            random_document = documents[random_idx]
                            random_start = random.randint(0, len(random_document) - 1)
                            for j in range(random_start, len(random_document)):
                                tokens_b.extend(random_document[j])
                                if len(tokens_b) >= target_b_length:
                                    break
                        else: 
                            is_random_next=False
                            for j in range(a_end, len(current_chunk)):
                                tokens_b.extend(current_chunk[j])

                        tokens_a, tokens_b = tokens_b, tokens_a
                    if len(tokens_a) == 0 or len(tokens_b) == 0:
                        continue
                    truncate_seq_pair(tokens_a, tokens_b, max_num_tokens)

                    # add[CLS,s1,SEP,s2,SEP,PAD]
                    src = [self.vocab.get(CLS_TOKEN)]
                    src.extend(tokens_a)
                    src.append(self.vocab.get(SEP_TOKEN))
                    src.extend(tokens_b)
                    src.append(self.vocab.get(SEP_TOKEN))
                    seg_pos = [len(tokens_a)+1, len(src)]
                    while len(src) < self.seq_length:
                        src.append(self.vocab.get(PAD_TOKEN))
                        
                    # Dynamic masking,
                    if not self.dynamic_masking:       
                        masked_src, mask_pos = mask_seq(src, self.tokenizer, self.whole_word_masking, self.span_masking,
                                                       self.span_geo_prob, self.span_max_length)
                    else:
                        masked_src=src
                        mask_pos=[]

                    example = {
                            "masked_src": masked_src,
                            "mask_pos": mask_pos,
                            "seg_pos": seg_pos,
                            "sentence_order_label":  1 if is_next else 0,
                            "random_next_label": 1 if is_random_next else 0
                        }
                    examples.append(example)

                current_chunk = []  # clear current chunk
                current_length = 0  # reset current text length

                # Shorten target_seq_length for each example with a small probility
                if random.random() < self.short_seq_prob:
                    target_seq_length = random.randint(2, max_num_tokens)
                else:
                    target_seq_length = max_num_tokens
            i += 1

        return examples

    def build_examples(self, documents: 'list[list[list[int]]]'):
        examples = []
        # multi-sampling on one file
        for _ in range(self.dup_factor):
            for doc_index in range(len(documents)):
                examples.extend(self.create_examples_from_signle_doc(documents, doc_index))
        return examples

    def write_examples(self, examples, dataset_writer):
        for example in examples:
            pickle.dump(example, dataset_writer)
        return
    
    def lines2documents(self,lines):     
        seg_documents = [content2sentences(line,128) for line in lines]
        documents = []
        for i in range(len(seg_documents)):
            document = [self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(line)) for line in seg_documents[i]]
            document = [sentence for sentence in document  if len(sentence) > 0]
            documents.append(document)
        return documents

    def worker(self, lines, proc_id):
        """
        perform preprocess and save
        Args:
            proc_id: id of the current processor
            start: the start position of the input paragraphs of text
            end: the end position of the input paragraphs of text
            ind: the index of input json file
        """
        # print("Worker %d is building dataset ... " % proc_id)
        # Initialization
        set_seed(self.seed)
        docs_buffer = []   # Store documents until the number reaches docs_buffer_size
        dataset_writer = open(self.temp_dir+self.dataset_tmp_prefix+str(proc_id) + ".pt", "wb")

        # Obtain data for current worker
        # 1)convert text to list of token ids 2) construct instances which will be saved as .pt file
        for pos, content in enumerate(tqdm(lines)):

            # End of the loop, process the remaining documents and save
            if pos >= len(lines)-1:
                if len(docs_buffer) > 0:
                    self.write_examples(docs_buffer, dataset_writer)
                break
            # Find all sentences in current document, and convert then to lists of token ids
            document = self.lines2documents([content])
            examples = self.build_examples(document)

            # Append document to docs_buffer if document is not empty
            if len(examples) >= 1:
                docs_buffer.extend(examples)

            # When size of docs_buffer reached the threshold, write instances from docs_buffer and empty docs_buffer
            if len(docs_buffer) > self.docs_buffer_size:
                self.write_examples(docs_buffer, dataset_writer)
                docs_buffer = []

        dataset_writer.close()

    

class AlbertDataset(Dataset):
    """
    Construct dataset for MLM and SOP tasks from the given corpus.
    Each document consists of multiple sentences,
    and each sentence occupies a single line.
    Documents in corpus must be separated by empty lines.
    """

    def __init__(self, args, vocab, tokenizer):
        super(AlbertDataset, self).__init__(args, vocab, tokenizer)
        self.short_seq_prob = args.short_seq_prob

    def worker(self, proc_id, start, end):
        print("Worker %d is building dataset ... " % proc_id)
        set_seed(self.seed)
        document = []
        dataset_writer = open("dataset-tmp-" + str(proc_id) + ".pt", "wb")
        for _ in range(self.dup_factor):
            pos = 0
            with open(self.corpus_path, mode="r", encoding="utf-8") as f:
                while pos < start:
                    f.readline()
                    pos += 1
                while True:
                    line = f.readline()
                    pos += 1
                    if not line.strip():
                        if len(document) >= 1:
                            instances = self.build_instances(document)
                            for instance in instances:
                                pickle.dump(instance, dataset_writer)
                        document = []
                    sentence = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(line))
                    if len(sentence) > 0:
                        document.append(sentence)
                    if pos >= end:
                        if len(document) >= 1:
                            instances = self.build_instances(document)
                            for instance in instances:
                                pickle.dump(instance, dataset_writer)
                        break
        dataset_writer.close()

    def build_instances(self, document):
        instances = []
        instances.extend(self.create_ins_from_doc(document))
        return instances

    def create_ins_from_doc(self, document):
        max_num_tokens = self.seq_length - 3
        target_seq_length = max_num_tokens
        if random.random() < self.short_seq_prob:
            target_seq_length = random.randint(2, max_num_tokens)
        instances = []
        current_chunk = []
        current_length = 0
        i = 0
        while i < len(document):
            segment = document[i]
            current_chunk.append(segment)
            current_length += len(segment)
            if i == len(document) - 1 or current_length >= target_seq_length:
                if current_chunk:
                    a_end = 1
                    if len(current_chunk) >= 2:
                        a_end = random.randint(1, len(current_chunk) - 1)

                    tokens_a = []
                    for j in range(a_end):
                        tokens_a.extend(current_chunk[j])

                    tokens_b = []
                    is_wrong_order = 0
                    for j in range(a_end, len(current_chunk)):
                        tokens_b.extend(current_chunk[j])

                    if random.random() < 0.5:
                        is_wrong_order = 1
                        tmp = tokens_a
                        tokens_a = tokens_b
                        tokens_b = tmp

                    truncate_seq_pair(tokens_a, tokens_b, max_num_tokens)

                    src = []
                    src.append(self.vocab.get(CLS_TOKEN))
                    src.extend(tokens_a)
                    src.append(self.vocab.get(SEP_TOKEN))
                    seg_pos = [len(src)]
                    src.extend(tokens_b)
                    src.append(self.vocab.get(SEP_TOKEN))
                    seg_pos.append(len(src))

                    while len(src) != self.seq_length:
                        src.append(self.vocab.get(PAD_TOKEN))

                    if not self.dynamic_masking:
                        src, tgt_mlm = mask_seq(src, self.tokenizer, self.whole_word_masking, self.span_masking,                 
                                                self.span_geo_prob, self.span_max_length)
                        instance = (src, tgt_mlm, is_wrong_order, seg_pos)
                    else:
                        instance = (src, is_wrong_order, seg_pos)
                    instances.append(instance)
                current_chunk = []
                current_length = 0
            i += 1
        return instances




