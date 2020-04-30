# TODO
class Encoder():
    pass


def encode_words_glove_fasttext(documents, vocab, embs, **kwargs):
    def encode(sentence):
        words = preprocess_string(sentence, kwargs['filters'])
        return [ list(embs[vocab[w]]) for w in words if w in vocab ]
    document_embs = []
    for document in documents:
        sentence_embs = []
        for sentence in document:
            sentence_embs.extend(encode(sentence))
        document_embs.append(sentence_embs)
    return document_embs


def encode_bigrams_lsa(documents, vocab, embs, **kwargs):
    def encode(sentence):
        X = kwargs['vectorizer'].transform(sentence)
        return [ list(embs[j]) for i,j in zip(*X.nonzero()) for c in range(X[i,j]) ]
    document_embs = []
    for document in documents:
        sentence_embs = []
        for sentence in document:
            sentence_embs.extend(encode([sentence]))
        document_embs.append(sentence_embs)
    return document_embs
