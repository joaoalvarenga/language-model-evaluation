import argparse
import os



parser = argparse.ArgumentParser(prog='Compute TF-IDF Similary')
parser.add_argument('--input_file', type=str, required=True)
parser.add_argument('--test_file', type=str, default='data/cv-corpus-6.1-2020-12-11/pt/test.csv')
args = parser.parse_args()

import pandas as pd
import re
import numpy as np
from scipy.sparse import csr_matrix
import sparse_dot_topn.sparse_dot_topn as ct
from sklearn.feature_extraction.text import TfidfVectorizer

def awesome_cossim_top(A, B, ntop, lower_bound=0):
  
    A = A.tocsr()
    B = B.tocsr()
    M, _ = A.shape
    _, N = B.shape
 
    idx_dtype = np.int32
 
    nnz_max = M*ntop
 
    indptr = np.zeros(M+1, dtype=idx_dtype)
    indices = np.zeros(nnz_max, dtype=idx_dtype)
    data = np.zeros(nnz_max, dtype=A.dtype)

    ct.sparse_dot_topn(
        M, N, np.asarray(A.indptr, dtype=idx_dtype),
        np.asarray(A.indices, dtype=idx_dtype),
        A.data,
        np.asarray(B.indptr, dtype=idx_dtype),
        np.asarray(B.indices, dtype=idx_dtype),
        B.data,
        ntop,
        lower_bound,
        indptr, indices, data)

    return csr_matrix((data,indices,indptr),shape=(M,N))

punct = re.compile('[!.?\-,:;()@#$%*\]\[{}]+')
spaces = re.compile('[\t\r\n\s]+')

def remove_punct(text):
    text = punct.sub(' ', text)
    text = spaces.sub(' ', text)
    return text.strip()

def get_matches_df(sparse_matrix, name_vector_left, name_vector_right, top=5):
    non_zeros = sparse_matrix.nonzero()

    sparserows = non_zeros[0]
    sparsecols = non_zeros[1]

    if top:
        print (top)
        nr_matches = top
    else:
        print (sparsecols.size)
        nr_matches = sparsecols.size

    left_side = np.empty([nr_matches], dtype=object)
    right_side = np.empty([nr_matches], dtype=object)
    similairity = np.zeros(nr_matches)
    left_side_index = np.zeros(nr_matches, dtype=np.int)
    right_side_index = np.zeros(nr_matches, dtype=np.int)

    for index in range(0, nr_matches):
        left_side_index[index] = sparserows[index]
        right_side_index[index] = sparsecols[index]
        left_side[index] = name_vector_left[sparserows[index]]
        right_side[index] = name_vector_right[sparsecols[index]]
        similairity[index] = sparse_matrix.data[index]

    return pd.DataFrame({'left_side': left_side,
                         'right_side': right_side,
                         'similarity': similairity,
                        'left_side_index': left_side_index,
                        'right_side_index': right_side_index})


if __name__ == '__main__':
    test_set = pd.read_csv(args.test_file)
    clean_test = [remove_punct(s).lower() for s in test_set['sentence']]
    with open(args.input_file) as f:
        input_corpus = [l.strip() for l in f]
    corpus = clean_test + input_corpus
    
    vectorizer = TfidfVectorizer()
    vectorizer = vectorizer.fit(corpus)
    input_tfidf = vectorizer.transform(input_corpus)
    test_tfidf = vectorizer.transform(clean_test)
    
    matches = awesome_cossim_top(test_tfidf, input_tfidf.transpose(), 1)
    matches_df = get_matches_df(matches, clean_test, input_corpus, top=None)
    max_similarity = max(matches_df['similarity'])
    mean_similarity = np.mean(matches_df['similarity'])
    std_similarity = np.std(matches_df['similarity'])
    print(f'{mean_similarity}\t{std_similarity}\t{max_similarity}'.replace('.', ','))
    
    

