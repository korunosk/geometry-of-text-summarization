# These helper methods produce the different embeddings
# for the documents.
# Documents do not necessarily mean topic's documents
# but rather a textual entity.

# TODO: Create class that represents the encoding process
class Encoder():
    pass


def encode_words_glove_fasttext(documents, vocab, embs, **kwargs):
    ''' Encodes the documents provided using the fasttext embeddings
    loaded from disk. 
    '''
    def encode(sentence):
        ''' Per sentence encoder '''
        words = preprocess_string(sentence, kwargs['filters'])
        return [ list(embs[vocab[w]]) for w in words if w in vocab ]
    # Iterate over each document and encode each sentence
    document_embs = []
    for document in documents:
        sentence_embs = []
        for sentence in document:
            sentence_embs.extend(encode(sentence))
        document_embs.append(sentence_embs)
    return document_embs


def encode_bigrams_lsa(documents, vocab, embs, **kwargs):
    ''' Encodes the documents provided using the LSA method '''
    def encode(sentence):
        ''' Per sentence encoder.
        It will use pre-trained vectorizer
        '''
        X = kwargs['vectorizer'].transform(sentence)
        return [ list(embs[j]) for i,j in zip(*X.nonzero()) for c in range(X[i,j]) ]
    # Iterate over each document and encode each sentence
    document_embs = []
    for document in documents:
        sentence_embs = []
        for sentence in document:
            sentence_embs.extend(encode([sentence]))
        document_embs.append(sentence_embs)
    return document_embs
