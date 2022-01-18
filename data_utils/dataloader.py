import random
import pickle
import torch
from utils.mask import mask_seq
from glob import glob

class DataLoader(object):
    def __init__(self, args,  batch_size, proc_id, proc_num, shuffle=False, demo_mode=False):
        self.tokenizer = args.tokenizer
        if demo_mode:
            self.batch_size = 1
        else:
            self.batch_size = batch_size
        self.instances_buffer_size = args.instances_buffer_size
        self.proc_id = proc_id
        self.proc_num = proc_num
        self.shuffle = shuffle
       
        # self.dataset_reader = open(dataset_path, "rb")
        self.read_count = 0
        self.start = 0
        self.end = 0
        self.buffer = []
        self.vocab = args.tokenizer.vocab
        self.whole_word_masking = args.whole_word_masking
        self.span_masking = args.span_masking
        self.span_geo_prob = args.span_geo_prob
        self.span_max_length = args.span_max_length
        self.demo_mode = demo_mode

        self.paths = glob(args.pt_dir+"*.pt")
        self.file_ind = 0
        self.cur_reader = open(self.paths[0], "rb")

    def _fill_buf(self):
        '''
        Load samples into buffer
        '''
        self.buffer = []
        while True:
            try:
                instance = pickle.load(self.cur_reader)
                self.read_count += 1
                if (self.read_count - 1) % self.proc_num == self.proc_id:
                    self.buffer.append(instance)
                    if len(self.buffer) >= self.instances_buffer_size:
                        break
            except EOFError:
                # Reach file end.
                self.cur_reader.close()
                self.file_ind += 1
                if self.file_ind == len(self.paths):
                    self.file_ind = 0
                self.cur_reader = open(self.paths[self.file_ind], "rb")

        if self.shuffle:
            random.shuffle(self.buffer)
        self.start = 0
        self.end = len(self.buffer)

    def _fill_buf_store(self):
        '''
        Load samples into buffer
        '''
        try:
            self.buffer = []
            while True:
                instance = pickle.load(self.dataset_reader)

                self.read_count += 1
                if (self.read_count - 1) % self.proc_num == self.proc_id:
                    self.buffer.append(instance)
                    if len(self.buffer) >= self.instances_buffer_size:
                        break
        except EOFError:
            # Reach file end.
            self.dataset_reader.seek(0)

        if self.shuffle:
            random.shuffle(self.buffer)
        self.start = 0
        self.end = len(self.buffer)

    def _empty(self):
        return self.start >= self.end

    def __del__(self):
        self.cur_reader.close()


class BertDataLoader(DataLoader):
    def __iter__(self):
        while True:
            # Load data from pt_dir
            while self._empty():
                self._fill_buf()
                
            # Generate one batch
            if self.start + self.batch_size >= self.end:
                instances = self.buffer[self.start:]
            else:
                instances = self.buffer[self.start: self.start + self.batch_size]
            self.start += self.batch_size

            src = []
            mask_pos = []
            is_next = []
            seg = []

            masked_words_num = 0
            for ins in instances:
                if not ins["mask_pos"]:
                    # Dynamic masking
                    ins['masked_src'], ins['mask_pos'] = mask_seq(ins['masked_src'],
                                                                  self.tokenizer, 
                                                                  self.whole_word_masking, 
                                                                  self.span_masking,
                                                                  self.span_geo_prob,
                                                                  self.span_max_length)

                # ins is a dictionary with key: masked_src,mask_pos,seg_pos,sentence_order_label,random_next_label
                src.append(ins['masked_src'])
                masked_words_num += len(ins['mask_pos'])
                mask_pos.append([0] * len(ins['masked_src']))
                for mask in ins['mask_pos']:
                    mask_pos[-1][mask[0]] = mask[1]
                is_next.append(ins['sentence_order_label'])
                # seg: [1,...,1,2,...,2,0,...,0],  1: the first sentence，2: the second sentence，0: the others
                seg.append([1] * ins['seg_pos'][0] + [2] * (ins['seg_pos'][1] - ins['seg_pos'][0]) + [0] * (len(ins['masked_src']) - ins['seg_pos'][1]))

            if masked_words_num == 0:
                continue
            
            yield torch.LongTensor(src), \
                torch.LongTensor(mask_pos), \
                torch.LongTensor(is_next), \
                torch.LongTensor(seg)




