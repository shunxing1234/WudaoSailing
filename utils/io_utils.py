import json
import pandas as pd

def read_json(path, key='content'):
    """
    Read json file with a simple structure: list of dictionaries where text content are stored in key
    """
    with open(path, mode="r", encoding="utf-8", errors="replace") as f:
        json_data = [e[key] for e in json.load(f)]
    return json_data


def read_txt(path):
    """
    txt file with paragraphs separated with single empty line
    """
    with open(path, mode="r", encoding="utf-8") as f:
        text = f.readlines()
    txt_data = []
    split_inds = [-1] + [i for i in range(len(text)) if not text[i].strip()] + [len(txt_data)]      # line index for end of a paragraph
    for i in range(len(split_inds)-1):
        txt_data.extend(text[split_inds[i]+1:split_inds[i+1]])
    return txt_data


def read_csv(path, sep='\t', header=0, key='content'):
    """
    csv file with heading, tab separator and the row 'content' as data source if not specified
    """
    data = pd.read_csv(path, sep=sep, header=header)
    return list(data[key])


if __name__=='__main__':
    txt_path = "/data/train_data/corpus/test.txt"
    csv_path = "/data/train_data/corpus/test.csv"
    print(read_csv(csv_path))
    import glob
    from collections import Counter
    corpus_dir = "D://PycharmProjects/cpt-bm/data/train_data/"
    files = glob.glob(corpus_dir + "*.*")
    postfixes = list(set([file.rsplit(".")[1] for file in files]))
    print(Counter(postfixes))

