import pandas as pd
from tqdm import tqdm

from src.handlers.dataHandler import extractColumns
from src.modules.semanticSimilarityModule import semantic_similarity
from src.modules.cosineSimilarityModule import cosine_similarity

reports = extractColumns(pd.read_csv('./data/reports.csv'), ['cord_uid', 'title', 'abstract'])
reports = reports.dropna(axis=0, how='all')
reports = list(reports.itertuples(index=False, name=None))

file = open('./data/input.txt', 'r')
new_report = file.read()
file.close()
# print(type(new_report))

(cosine, semantic, cosine_cord_uids, semantic_cord_uids) = (0, 0, [], [])

def most_similar(reports, new_report):
    for report in reports:
        new_cosine = cosine_similarity(report[2], new_report)
        new_semantic = semantic_similarity(report[2], new_report)
        if cosine < new_cosine:
            cosine = new_cosine
            cosine_cord_uids.append(report[0])
        if semantic < new_semantic:
            semantic = new_semantic
            semantic_cord_uids.append(report[0])

    return (cosine_cord_uids, semantic_cord_uids)


def similarity_range(reports, new_report):
    min_cosine, max_cosine = 1, 0
    min_semantic, max_semantic = 1, 0
    for report in tqdm(reports, desc="Calculating Range", unit="element", ncols=100):
        new_cosine = cosine_similarity(report[2], new_report)
        new_semantic = semantic_similarity(report[2], new_report)
        if new_cosine != -1 and new_semantic != -1:
            min_cosine = min(min_cosine, new_cosine)
            max_cosine = max(max_cosine, new_cosine)
            max_semantic = max(max_semantic, new_semantic)
            min_semantic = min(min_semantic, new_semantic)
    # print({'cosine': (min_cosine, max_cosine), 'semantic': (min_semantic, max_semantic)})
    return {'cosine': (min_cosine, max_cosine), 'semantic': (min_semantic, max_semantic)}

def translate_threshold(threshold, range):
    threshold_cosine = (range['cosine'][1] - range['cosine'][0]) * threshold + range['cosine'][0]
    threshold_semantic = (range['semantic'][1] - range['semantic'][0]) * threshold + range['semantic'][0]
    # print(threshold_semantic, threshold_cosine)
    return (threshold_cosine, threshold_semantic)

def similarity(reports, new_report, threshold):
    threshold_cosine, threshold_semantic = translate_threshold(threshold, similarity_range(reports, new_report))
    for report in tqdm(reports, desc="Checking         ", unit="element", ncols=100):
        new_cosine = cosine_similarity(report[2], new_report)
        new_semantic = semantic_similarity(report[2], new_report)
        if new_cosine > threshold_cosine:
            cosine_cord_uids.append((report[0], new_cosine))
        if new_semantic > threshold_semantic:
            semantic_cord_uids.append((report[0], new_semantic))
    result = {'cosine': sorted(cosine_cord_uids, key=lambda x: x[1], reverse=True), 'semantic': sorted(semantic_cord_uids, key=lambda x: x[1], reverse=True)}
    result['semantic'] = result['semantic'][:len(result['cosine']) * 2 + len(result['cosine'])]
    return result
    
print(similarity(reports, new_report, 0.85))