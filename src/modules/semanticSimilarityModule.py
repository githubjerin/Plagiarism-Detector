import spacy

nlp = spacy.load('en_core_web_lg')

def semantic_similarity(docA: str, docB: str) -> float:
    if not isinstance(docA, str) or not isinstance(docB, str):
        return -1
    return nlp(docA).similarity(nlp(docB))

if __name__ == '__main__':
    sentence1 = "I eat food "
    sentence2 = "I love to eat"
    print(semantic_similarity(sentence1, sentence2))
