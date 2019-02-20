from __future__ import print_function
import json
import time

from nsm.word_embeddings import EmbeddingModel
from nsm.env_factory import QAProgrammingEnv
from nsm.computer_factory import LispInterpreter
from nsm.data_utils import Vocab
import nsm.executor_factory as executor_factory
import table.utils as utils
from table.experiment import load_jsonl


def load_environments(example_files, table_file, vocab_file, en_vocab_file, embedding_file):
    dataset = []
    for fn in example_files:
        dataset += load_jsonl(fn)
    print('{} examples in dataset.'.format(len(dataset)))

    tables = load_jsonl(table_file)
    table_dict = {table['name']: table for table in tables}
    print('{} tables.'.format(len(table_dict)))

    # Load pretrained embeddings.
    embedding_model = EmbeddingModel(vocab_file, embedding_file)

    with open(en_vocab_file, 'r') as f:
        vocab = json.load(f)

    en_vocab = Vocab([])
    en_vocab.load_vocab(vocab)
    print('{} unique tokens in encoder vocab'.format(
        len(en_vocab.vocab)))
    print('{} examples in the dataset'.format(len(dataset)))

    # Create environments.
    environments = create_environments(table_dict, dataset, en_vocab, embedding_model, executor_type='wtq')
    print('{} environments in total'.format(len(environments)))

    return environments


def create_environments(table_dict, dataset, en_vocab, embedding_model, executor_type,
                        max_n_mem=50, max_n_exp=3,
                        pretrained_embedding_size=300):
    all_envs = []

    if executor_type == 'wtq':
        score_fn = utils.wtq_score
        process_answer_fn = lambda x: x
        executor_fn = executor_factory.WikiTableExecutor
    elif executor_type == 'wikisql':
        score_fn = utils.wikisql_score
        process_answer_fn = utils.wikisql_process_answer
        executor_fn = executor_factory.WikiSQLExecutor
    else:
        raise ValueError('Unknown executor {}'.format(executor_type))

    for i, example in enumerate(dataset):
        if i % 100 == 0:
            print('creating environment #{}'.format(i))

        kg_info = table_dict[example['context']]
        executor = executor_fn(kg_info)
        api = executor.get_api()
        type_hierarchy = api['type_hierarchy']
        func_dict = api['func_dict']
        constant_dict = api['constant_dict']

        interpreter = LispInterpreter(
            type_hierarchy=type_hierarchy,
            max_mem=max_n_mem,
            max_n_exp=max_n_exp,
            assisted=True)

        for v in func_dict.values():
            interpreter.add_function(**v)

        interpreter.add_constant(
            value=kg_info['row_ents'],
            type='entity_list',
            name='all_rows')

        de_vocab = interpreter.get_vocab()

        constant_value_embedding_fn = lambda x: utils.get_embedding_for_constant(x, embedding_model,
                                                                                 embedding_size=pretrained_embedding_size)

        env = QAProgrammingEnv(en_vocab, de_vocab,
                               question_annotation=example,
                               answer=process_answer_fn(example['answer']),
                               constants=constant_dict.values(),
                               interpreter=interpreter,
                               constant_value_embedding_fn=constant_value_embedding_fn,
                               score_fn=score_fn,
                               name=example['id'])
        all_envs.append(env)

    return all_envs


def init_interpreter_for_example(example_dict, table_dict):
    executor = executor_factory.WikiTableExecutor(table_dict)
    api = executor.get_api()

    interpreter = LispInterpreter(type_hierarchy=api['type_hierarchy'],
                                  max_mem=60,
                                  max_n_exp=50,
                                  assisted=True)

    for func in api['func_dict'].values():
        interpreter.add_function(**func)

    interpreter.add_constant(value=table_dict['row_ents'],
                             type='entity_list',
                             name='all_rows')

    for constant in api['constant_dict'].values():
        interpreter.add_constant(type=constant['type'],
                                 value=constant['value'])

    for entity in example_dict['entities']:
        interpreter.add_constant(value=entity['value'],
                                 type=entity['type'])

    return interpreter


if __name__ == '__main__':
    # envs = load_environments(["/Users/yinpengcheng/Research/SemanticParsing/nsm/data/wikitable_reproduce/processed_input/train_examples.jsonl"],
    #                          "/Users/yinpengcheng/Research/SemanticParsing/nsm/data/wikitable_reproduce/processed_input/tables.jsonl",
    #                          vocab_file="/Users/yinpengcheng/Research/SemanticParsing/nsm/data/wikitable/raw_input/wikitable_glove_vocab.json",
    #                          en_vocab_file="/Users/yinpengcheng/Research/SemanticParsing/nsm/data/wikitable/processed_input/preprocess_14/en_vocab_min_count_5.json",
    #                          embedding_file="/Users/yinpengcheng/Research/SemanticParsing/nsm/data/wikitable/raw_input/wikitable_glove_embedding_mat.npy")
    #
    # env_dict = {env.name: env for env in envs}
    # env_dict['nt-3035'].interpreter.interactive(assisted=True)

    # examples = load_jsonl("/Users/yinpengcheng/Research/SemanticParsing/nsm/data/wikitable/processed_input/preprocess_14/train_examples.jsonl")
    # tables = load_jsonl("/Users/yinpengcheng/Research/SemanticParsing/nsm/data/wikitable/processed_input/preprocess_14/tables.jsonl")
    #
    # examples_dict = {e['id']: e for e in examples}
    # tables_dict = {tab['name']: tab for tab in tables}
    #
    # # q_id = 'nt-10300'
    # q_id = 'nt-13522'
    # interpreter = init_interpreter_for_example(examples_dict[q_id], tables_dict[examples_dict[q_id]['context']])
    # interpreter.interactive(assisted=True)
    print(utils.wtq_score([0.4], ['00.4', '0.4']))
