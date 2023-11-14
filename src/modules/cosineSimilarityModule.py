from nltk import word_tokenize
from nltk.corpus import stopwords
import math
import re
from nltk.stem import WordNetLemmatizer
from nltk.probability import FreqDist

def calculate_cosine(vector_a, vector_b):
    intersection = set(vector_a.keys()) & set(vector_b.keys())
    numerator = sum([vector_a[x] * vector_b[x] for x in intersection])

    sum1 = sum([vector_a[x] ** 2 for x in list(vector_a.keys())])
    sum2 = sum([vector_b[x] ** 2 for x in list(vector_b.keys())])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator

#remove punctuations and lowercase characters
def clean(document):
    return re.sub(r'[^(a-zA-Z)\s]', '', document.lower())

def remove_stopwords(tokens):
    stop_words = list(set(stopwords.words('english')))
    cleaned = [w for w in tokens if w not in stop_words]
    return cleaned

def tokenize(document):
    return word_tokenize(document)

def lemmatize(tokens):
    lemmatizer = WordNetLemmatizer()
    for i in range(len(tokens)):
        tokens[i] = lemmatizer.lemmatize(tokens[i])
    return tokens

def calculate_frequency(tokens):
    fdist = FreqDist()
    for each in tokens:
        fdist[each] = fdist[each] + 1
    return dict(fdist)

def preprocess(document: str):
    return calculate_frequency(lemmatize(remove_stopwords(tokenize(clean(document)))))

def cosine_similarity(docA: str, docB: str):
    if not isinstance(docA, str) or not isinstance(docB, str):
        return -1
    return calculate_cosine(preprocess(docA), preprocess(docB))

if __name__ == '__main__':
    docA = "The fire in his eyes warmed her heart as he passionately spoke about his dreams. The intense heat between them grew with every shared moment, igniting a connection that would endure."
    docB = "She noticed the fire escape route, her eyes widening with fear. The blazing inferno engulfed the building, the scorching heat making escape impossible."
    print(cosine_similarity(docA, docB))