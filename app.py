from flask import Flask, request, jsonify
from gevent.pywsgi import WSGIServer
from collections import Counter
from kb.dbpedia import DBpedia
from learning.classifier.svmclassifier import SVMClassifier
from learning.treelstm import Constants
from learning.treelstm.vocab import Vocab
from learning.treelstm.model import DASimilarity
from learning.treelstm.model import SimilarityTreeLSTM
from learning.treelstm.trainer import Trainer
from learning.treelstm.dataset import QGDataset
from common.container.struct import Struct
from common.container.uri import Uri
from common.container.linkeditem import LinkedItem
from common.graph.graph import Graph
from common.query.querybuilder import QueryBuilder
from parser.answerparser import AnswerParser
from parser.qald import QaldParser
from parser.lc_quad import LC_QaudParser

import os
import torch
import torch.nn as nn
import torch.optim as optim
import itertools
import numpy as np
import logging
import learning.treelstm.preprocess_lcquad as preprocess_lcquad
import ujson

app = Flask(__name__)

base_dir = "./output"

question_type_classifier_path = os.path.join(base_dir, "question_type_classifier")
question_type_classifier = SVMClassifier(os.path.join(question_type_classifier_path, "svm.model"))

double_relation_classifier_path = os.path.join(base_dir, "double_relation_classifier")
double_relation_classifier = SVMClassifier(os.path.join(double_relation_classifier_path, "svm.model"))

kb = DBpedia()
answer_parser = AnswerParser(kb)
qald_parser = QaldParser()
parser = LC_QaudParser()

dep_tree_cache_file_path = './caches/dep_tree_cache_lcquadtest.json'
if os.path.exists(dep_tree_cache_file_path):
    with open(dep_tree_cache_file_path) as f:
        dep_tree_cache = ujson.load(f)
else:
    dep_tree_cache = dict()


@app.route('/api/query', methods=['POST'])
def query():
    question = request.json['question']
    raw_entities = request.json['entities']
    raw_relations = request.json['relations']

    entities = []
    for item in raw_entities:
        uris = [Uri(uri["uri"], DBpedia.parse_uri, uri["confidence"]) for uri in item["uris"]]
        entities.append(LinkedItem(item["surface"], uris))

    relations = []
    for item in raw_relations:
        uris = [Uri(uri["uri"], DBpedia.parse_uri, uri["confidence"]) for uri in item["uris"]]        
        relations.append(LinkedItem(item["surface"], uris))

    question_type, type_confidence = get_question_type(question)

    count_query = False
    ask_query = False
    if question_type == 2:
        count_query = True
    elif question_type == 1:
        ask_query = True

    # try every combination create_entity_relations_combinations()
    generated_queries = generate_query(question, question_type, entities, relations, count_query, ask_query)
    queries = postprocess(generated_queries, count_query, ask_query)

    result = {
        "queries": queries,
        "type": get_question_type_text(question_type),
        "type_confidence": type_confidence
    }

    return jsonify(result)

def get_question_type(question):
    question_type = question_type_classifier.predict([question])
    type_confidence = question_type_classifier.predict_proba([question])[0][question_type]

    if isinstance(question_type_classifier.predict_proba([question])[0][question_type], (np.ndarray, list)):
        type_confidence = type_confidence[0]
        type_confidence = float(type_confidence)

    question_type = int(question_type)

    return question_type, type_confidence

def get_question_type_text(question_type):
    question_type_text = "list"
    if question_type == 2:
        question_type_text = "count"
    elif question_type == 1:
        question_type_text = "boolean"
    return question_type_text

def generate_query(question, question_type, entities, relations, ask_query=False, count_query=False):
    sort_query = False
    h1_threshold = 9999999

    double_relation = False
    if double_relation_classifier is not None:
        double_relation = double_relation_classifier.predict([question])
        if double_relation == 1:
            double_relation = True

    graph = Graph(kb)
    graph.find_minimal_subgraph(entities, relations, double_relation=double_relation, ask_query=ask_query,
                                sort_query=sort_query, h1_threshold=h1_threshold)

    query_builder = QueryBuilder()
    valid_walks = query_builder.to_where_statement(graph, answer_parser.parse_queryresult, ask_query=ask_query,
                                                   count_query=count_query, sort_query=sort_query)

    if question_type == 0 and len(relations) == 1:
        double_relation = True
        graph = Graph(kb)
        query_builder = QueryBuilder()
        graph.find_minimal_subgraph(entities, relations, double_relation=double_relation, ask_query=ask_query,
                                    sort_query=sort_query, h1_threshold=h1_threshold)
        valid_walks_new = query_builder.to_where_statement(graph, answer_parser.parse_queryresult, ask_query=ask_query,
                                                           count_query=count_query, sort_query=sort_query)
        valid_walks.extend(valid_walks_new)

    args = Struct()
    base_path = "./learning/treelstm/"
    args.save = os.path.join(base_path, "checkpoints/")
    #args.expname = "lc_quad,epoch=5,train_loss=0.08340245485305786"
    args.expname = "lc_quad,epoch=15,train_loss=0.09691771119832993"
    args.mem_dim = 150
    args.hidden_dim = 50
    args.num_classes = 2
    args.input_dim = 300
    args.sparse = False
    args.lr = 0.01
    args.wd = 1e-4
    args.data = os.path.join(base_path, "data/lc_quad/")
    args.cuda = False
    # args.cuda = True
    try:
        scores = rank(args, question, valid_walks)
    except FileNotFoundError as error:
        print(error)
        scores = [1 for _ in valid_walks]
    for idx, item in enumerate(valid_walks):
        if idx >= len(scores):
            item["confidence"] = 0.3
        else:
            item["confidence"] = float(scores[idx] - 1)

    return valid_walks


def postprocess(generated_queries, count_query, ask_query):
    scores = []
    queries = []
    for s in generated_queries:
        scores.append(s['confidence'])

    scores = np.array(scores)
    inds = scores.argsort()[::-1]
    sorted_queries = [generated_queries[s] for s in inds]
    scores = [scores[s] for s in inds]

    used_answer = []
    uniqueid = []
    for i in range(len(sorted_queries)):
        if sorted_queries[i]['where'] not in used_answer:
            used_answer.append(sorted_queries[i]['where'])
            uniqueid.append(i)

    sorted_queries = [sorted_queries[i] for i in uniqueid]
    scores = [scores[i] for i in uniqueid]

    s_counter = Counter(sorted(scores, reverse=True))
    s_ind = []
    s_i = 0
    for k, v in s_counter.items():
        s_ind.append(range(s_i, s_i + v))
        s_i += v

    output_where = [{"query": " .".join(item["where"]), "correct": False, "target_var": "?u_0"} for item in
                    sorted_queries]
    for item in list(output_where):
        print(item["query"])
    correct = False

    wrongd = {}

    for idx in range(len(sorted_queries)):
        where = sorted_queries[idx]

        if "answer" in where:
            #answerset = where["answer"]
            target_var = where["target_var"]
        else:
            target_var = "?u_" + str(where["suggested_id"])
            #raw_answer = kb.query_where(where["where"], target_var, count_query, ask_query)
            #answerset = AnswerSet(raw_answer, answer_parser.parse_queryresult)

        output_where[idx]["target_var"] = target_var
        query = {
            "query": kb.sparql_query(where["where"], target_var, count_query, ask_query),
            "confidence": where["confidence"]
        }
        queries.append(query)

    return queries


def rank(args, question, generated_queries):
    if len(generated_queries) == 0:
        return []
    if 2 > 1:
        # Load the model
        checkpoint_filename = '%s.pt' % os.path.join(args.save, args.expname)
        dataset_vocab_file = os.path.join(args.data, 'dataset.vocab')
        # metrics = Metrics(args.num_classes)
        vocab = Vocab(filename=dataset_vocab_file,
                      data=[Constants.PAD_WORD, Constants.UNK_WORD, Constants.BOS_WORD, Constants.EOS_WORD])
        similarity = DASimilarity(args.mem_dim, args.hidden_dim, args.num_classes)
        model = SimilarityTreeLSTM(
            vocab.size(),
            args.input_dim,
            args.mem_dim,
            similarity,
            args.sparse)
        criterion = nn.KLDivLoss()
        optimizer = optim.Adagrad(model.parameters(), lr=args.lr, weight_decay=args.wd)
        emb_file = os.path.join(args.data, 'dataset_embed.pth')

        if os.path.isfile(emb_file):
            emb = torch.load(emb_file)
        model.emb.weight.data.copy_(emb)
        checkpoint = torch.load(checkpoint_filename, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['model'])
        trainer = Trainer(args, model, criterion, optimizer)

        # Prepare the dataset
        json_data = [{"id": "test", "question": question,
                      "generated_queries": [{"query": " .".join(query["where"]), "correct": False} for query in
                                            generated_queries]}]
        output_dir = "./output/qald"
        preprocess_lcquad.save_split(output_dir, *preprocess_lcquad.split(json_data, parser))
        if question in dep_tree_cache:
            preprocess_lcquad.parse(output_dir, dep_parse=False)

            cache_item = dep_tree_cache[question]
            with open(os.path.join(output_dir, 'a.parents'), 'w') as f_parent, open(
                    os.path.join(output_dir, 'a.toks'), 'w') as f_token:
                for i in range(len(generated_queries)):
                    f_token.write(cache_item[0])
                    f_parent.write(cache_item[1])
        else:
            preprocess_lcquad.parse(output_dir)
            with open(os.path.join(output_dir, 'a.parents')) as f:
                parents = f.readline()
            with open(os.path.join(output_dir, 'a.toks')) as f:
                tokens = f.readline()
            dep_tree_cache[question] = [tokens, parents]

            with open(dep_tree_cache_file_path, 'w') as f:
                ujson.dump(dep_tree_cache, f)
        test_dataset = QGDataset(output_dir, vocab, args.num_classes)
        test_loss, test_pred = trainer.test(test_dataset)
        return test_pred


def create_entity_relations_combinations():
    entity_list = []
    for L in range(1, len(entities) + 1):
        for subset in itertools.combinations(entities, L):
            entity_list.append(subset)
    entity_list = entity_list[::-1]

    relation_list = []
    for L in range(1, len(ontologies) + 1):
        for subset in itertools.combinations(ontologies, L):
            relation_list.append(subset)
    relation_list = relation_list[::-1]

    combination_list = [(x, y) for x in entity_list for y in relation_list]
    return combination_list

if __name__ == '__main__':
    logger = logging.getLogger(__name__)

    logger.info("Starting the HTTP server")
    http_server = WSGIServer(('', 9011), app)
    http_server.serve_forever()
