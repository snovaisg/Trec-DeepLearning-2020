import pyterrier as pt
import pandas as pd
import csv
import os
import shutil

if not pt.started():
    pt.init()

def passages_generator(filepath : str, delimiter : str, verbose : bool=False):
    """
    Generator of passages dataset. Generates 1 passage at a time.

    Parameters
    ----------
    filepath : str
        path of file that contains passages in a csv format 
        with two fields: [passage_id] and [passage_text].

    delimiter : str
        delimiter of csv file that contains the passages

    verbose: bool, default=False
        Whether or not to log progress frequently.

    Returns
    -------
    {'docno': docno, 'text': text}
    """
    csv_file = open(filepath)
    read_csv = csv.reader(csv_file, delimiter=delimiter)

    for i, (docno, text) in enumerate(read_csv):
        if i % 200000 == 0 and verbose:
            print(f'Processing passage {i}')
        yield {'docno': docno, 'text': text}



# uncomment to create index if index is not yet created.
if os.path.exists('index'):
    shutil.rmtree('index')
index_path = os.path.join(os.getcwd(),'index')
iter_indexer = pt.IterDictIndexer(index_path)

collection_file = os.path.join(os.getcwd(),'data','collection.tsv')

doc_iter = passages_generator(collection_file, '\t', verbose=True)
index_passages = iter_indexer.index(doc_iter)
print("done")