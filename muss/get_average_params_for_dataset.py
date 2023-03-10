from muss import feature_extraction
from muss.text import get_content_words
import jsonlines
from muss.simplify import ALLOWED_MODEL_NAMES, simplify_sentences
from easse.sari import corpus_sari
import numpy as np
import random
import argparse

random.seed(42)
# lev_sims_down = [0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
# depth_ratios_down = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# rank_ratios_down = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
# len_ratios_down = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
grid_params = {
    'lev_sims_down': [0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9],
    'depth_ratios_down': [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    'rank_ratios_down': [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9],
    'len_ratios_down': [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9],
    'lev_sims_up': [0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9],
    'depth_ratios_up': [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7],
    'rank_ratios_up': [1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4],
    'len_ratios_up': [1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4]
}


def get_all_feature_results(inp_sents: list, out_sents: list):
    lev_results = []
    word2rank_results = []
    depth_results = []
    length_ratio_results = []
    for i in range(len(inp_sents)):
        levsim = feature_extraction.get_replace_only_levenshtein_similarity(complex_sentence=inp_sents[i],
                                                                            simple_sentence=out_sents[i])
        lev_results.append(levsim)
        inp_depth = feature_extraction.get_dependency_tree_depth(inp_sents[i])
        out_depth = feature_extraction.get_dependency_tree_depth(out_sents[i])

        depth_results.append(out_depth / inp_depth)
        inp_len = len(' '.join(get_content_words(inp_sents[i])))
        out_len = len(' '.join(get_content_words(out_sents[i])))
        len_ratio = out_len / inp_len
        length_ratio_results.append(len_ratio)
        inp_wordrank = feature_extraction.get_lexical_complexity_score(inp_sents[i], language='en')
        out_wordrank = feature_extraction.get_lexical_complexity_score(out_sents[i], language='en')
        rank_ratio = out_wordrank / inp_wordrank
        word2rank_results.append(rank_ratio)
    av_lev = np.round(np.mean(lev_results), 2)
    av_wordrank = np.round(np.mean(word2rank_results), 2)
    av_depth = np.round(np.mean(depth_results), 2)
    av_length_ratio = np.round(np.mean(length_ratio_results), 2)
    print({
        'lev': av_lev,
        'rank_ratio': av_wordrank,
        'depth': av_depth,
        'char_len_ratio': av_length_ratio
    })


def do_param_gridsearch(all_inputs, all_outputs, direc='down', model_name='muss_en_wikilarge_mined'):
    # sample 50 pairs
    lev_options = grid_params[f'lev_sims_{direc}']
    depth_options = grid_params[f'depth_ratios_{direc}']
    rank_options = grid_params[f'rank_ratios_{direc}']
    len_options = grid_params[f'len_ratios_{direc}']
    best_sari = 0.
    best_processor_args = None
    num_files = 0
    best_sari_ind = None
    for lev in lev_options:
        for depth in depth_options:
            for rank in rank_options:
                for length in len_options:
                    num_files += 1
                    print(num_files)
                    processor_args = argparse.Namespace(len_ratio=length, lev_sim=lev,
                                                        tree_depth=depth, word_rank=rank)
                    print(processor_args)
                    pred_sentences = simplify_sentences(all_inputs, processor_args, model_name=model_name)
                    sari = corpus_sari(orig_sents=all_inputs, sys_sents=pred_sentences,
                                       refs_sents=[all_outputs], lowercase=True)
                    if sari > best_sari:
                        best_sari = sari
                        best_processor_args = processor_args
                        best_sari_ind = num_files
                        print(f'new best sari {best_sari} at ind {best_sari_ind}')
    print(mod_name, direc)
    print(best_sari)
    print(best_processor_args)
    print(best_sari_ind)







if __name__ == '__main__':
    data_path = '/home/nlplab/achi/Paraphrase_Level_Up/newsela_exps/news_manual_all_val.json'
    # data_path = '/home/nlplab/achi/Paraphrase_Level_Up/asset_test_data_with_repeats.json'
    inps = []
    outs = []
    with jsonlines.open(data_path) as reader:
        for obj in reader:
            inps.append(obj['paraphrase']['ori'])
            outs.append(obj['paraphrase']['para'])
    # get_all_feature_results(inp_sents=inps, out_sents=outs)
    mod_name = 'muss_en_wikilarge_mined'
    print(data_path)
    do_param_gridsearch(all_inputs=inps, all_outputs=outs, direc='down',
                        model_name=mod_name)