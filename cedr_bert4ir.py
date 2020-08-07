import pyterrier as pt
import pandas as pd
import csv
import os
import shutil

if not pt.started():
    pt.init()

# paths
index_path = os.path.join(os.getcwd(),'index')
qrels_path= os.path.join(os.getcwd(),'data')
qrels_eval19_path = os.path.join(os.getcwd(),'data19', '2019qrels-pass.txt')
qrels_train_path = os.path.join(qrels_path, 'qrels.train.tsv')
qrels_dev_path = os.path.join(qrels_path, 'qrels.dev.tsv')

topics_path = os.path.join(os.getcwd(),'data')
topics_train_path = os.path.join(topics_path, 'queries.train.tsv')
topics_dev_path = os.path.join(topics_path, 'queries.dev.tsv')
topics_eval_path = os.path.join(topics_path, 'queries.eval.tsv')
topics_eval19_path = os.path.join(os.getcwd(),'data19', 'queries.eval.tsv')

# read data into dataframes from paths
topics_train = pt.io.read_topics(topics_train_path, format='singleline')
topics_dev = pt.io.read_topics(topics_dev_path, format='singleline')
topics_eval = pt.io.read_topics(topics_eval_path, format='singleline')
topics_eval19 = pt.io.read_topics(topics_eval19_path, format='singleline')

qrels_train = pt.io.read_qrels(qrels_train_path)
qrels_dev = pt.io.read_qrels(qrels_dev_path)
qrels_eval19 = pt.io.read_qrels(qrels_eval19_path)

indexRef = pt.TRECCollectionIndexer(index_path)

def fill_empty_queries(df: pd.DataFrame):
    """
    fills all empty queries with some text so that pyterrier_bert does not crash.
    """
    df_copy = df.copy()
    df_copy.loc[df_copy['query'].str.len() == 0,'query'] = 'nova'
    return df_copy


topics_train = fill_empty_queries(topics_train)
topics_dev = fill_empty_queries(topics_dev)
topics_eval = fill_empty_queries(topics_eval)
topics_eval19 = fill_empty_queries(topics_eval19)

from pyterrier_bert.pyt_cedr import CEDRPipeline
from pyterrier_bert.bert4ir import *

if not os.path.exists('results'):
    os.mkdir('results')
results_path = os.path.join(os.getcwd(), 'results')

BM25_br = pt.BatchRetrieve(indexRef, controls={"wmodel" : "BM25"}, verbose=True)

#cedr
cedrpipe = BM25_br >> CEDRPipeline(max_valid_rank=20)

cedrpipe.fit(topics_train, qrels_train, topics_dev, qrels_dev)

df_baseline_retrieval_2020_cedr = cedrpipe.transform(topics_eval)
df_baseline_retrieval_2020_cedr.to_csv(
    os.path.join(results_path, 'eval20_baseline_retrieval__bm25_cedr.csv'), \
    index=False)
print('done cedr 20')

df_result_eval19_cedr = pt.pipelines.Experiment([cedrpipe],
                        topics_dev,
                        qrels_dev,
                        ['map','ndcg'],  
                        names=["BM25 + cedr"])

df_result_eval19.to_csv(
    os.path.join(results_path, 'eval19_baseline_retrieval__bm25_cedr.csv'), \
    index=False)
print('Finished cedr 19')

# bert4ir

bertpipe = BM25_br >> BERTPipeline(max_valid_rank=20)

bertpipe.fit(topics_train, qrels_train, topics_dev, qrels_dev)

df_baseline_retrieval_2020 = bertpipe.transform(topics_eval)
df_baseline_retrieval_2020.to_csv(
    os.path.join(results_path, 'eval20_baseline_retrieval__bm25_bert4ir.csv'), \
    index=False)
print('done bert4ir 20')

df_result_eval19_bert = pt.pipelines.Experiment([bertpipe],
                        topics_dev,
                        qrels_dev,
                        ['map','ndcg'],  
                        names=["BM25 + bert4ir"])

df_result_eval19_bert.to_csv(
    os.path.join(results_path, 'eval19_baseline_retrieval__bm25_bert.csv'), \
    index=False)
print('Finished bert4ir 19')
