# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import shutil
import re

from muss.preprocessors import get_preprocessors
from muss.utils.helpers import write_lines, read_lines, get_temp_filepath
from muss.simplifiers import get_fairseq_simplifier, get_preprocessed_simplifier
from muss.resources.paths import MODELS_DIR
from muss.utils.resources import download_and_extract


# Models are the best of each experiment according to validation SARI score
ALLOWED_MODEL_NAMES = ['muss_en_wikilarge_mined', 'muss_en_mined', 'muss_fr_mined', 'muss_es_mined', 'muss_pt_mined']

TOKENS_RATIO_DEFAULT = {
    "LengthRatioPreprocessor": 0.9,
    "ReplaceOnlyLevenshteinPreprocessor": 0.8,
    "WordRankRatioPreprocessor": 0.8,
    "DependencyTreeDepthRatioPreprocessor": 0.4,
}

TOKENS_RATIO_EF1 = {
    "LengthRatioPreprocessor": 0.5,
    "ReplaceOnlyLevenshteinPreprocessor": 0.5,
    "WordRankRatioPreprocessor": 0.4,
    "DependencyTreeDepthRatioPreprocessor": 0.2,
}

TOKENS_RATIO = TOKENS_RATIO_DEFAULT


def is_model_using_mbart(model_name):
    return '_fr_' in model_name or '_es_' in model_name or '_pt_' in model_name


def get_mbart_languages_from_model(model_name):
    if '_pt_' in model_name:
        return 'ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN,af_ZA,az_AZ,bn_IN,fa_IR,he_IL,hr_HR,id_ID,ka_GE,km_KH,mk_MK,ml_IN,mn_MN,mr_IN,pl_PL,ps_AF,pt_XX,sv_SE,sw_KE,ta_IN,te_IN,th_TH,tl_XX,uk_UA,ur_PK,xh_ZA,gl_ES,sl_SI'  # noqa: E501
    else:
        return 'ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN'  # noqa: E501


def get_model_path(model_name):
    assert model_name in ALLOWED_MODEL_NAMES
    model_path = MODELS_DIR / model_name
    if not model_path.exists():
        url = f'https://dl.fbaipublicfiles.com/muss/{model_name}.tar.gz'
        extracted_path = download_and_extract(url)[0]
        shutil.move(extracted_path, model_path)
    return model_path


def get_language_from_model_name(model_name):
    return re.match('(..)_*', model_name).groups()[0]


def get_muss_preprocessors(model_name, processor_args):
    language = get_language_from_model_name(model_name)

    preprocessors_kwargs = {
        'LengthRatioPreprocessor': {'target_ratio': processor_args.len_ratio, 'use_short_name': False},
        # 'LengthRatioPreprocessor': {'target_ratio': 0.9, 'use_short_name': False},
        'ReplaceOnlyLevenshteinPreprocessor': {'target_ratio': processor_args.lev_sim, 'use_short_name': False},
        'WordRankRatioPreprocessor': {'target_ratio': processor_args.word_rank, 'language': language, 'use_short_name': False},
        'DependencyTreeDepthRatioPreprocessor': {'target_ratio': processor_args.tree_depth, 'language': language, 'use_short_name': False},
    }
    if is_model_using_mbart(model_name):
        preprocessors_kwargs['SentencePiecePreprocessor'] = {
            'sentencepiece_model_path': get_model_path(model_name) / 'sentencepiece.bpe.model',
            'tokenize_special_tokens': True,
        }
    else:
        preprocessors_kwargs['GPT2BPEPreprocessor'] = {}
    return get_preprocessors(preprocessors_kwargs)


def simplify_sentences(source_sentences, processor_args, model_name='muss_en_wikilarge_mined'):
    # Best ACCESS parameter values for the
    # en_bart_access_wikilarge_mined model, ideally we would
    # need to use another set of parameters for other
    # models.
    # import pdb; pdb.set_trace()
    exp_dir = get_model_path(model_name)
    preprocessors = get_muss_preprocessors(model_name, processor_args)
    generate_kwargs = {}
    if is_model_using_mbart(model_name):
        generate_kwargs['task'] = 'translation_from_pretrained_bart'
        generate_kwargs[
            'langs'
        ] = 'ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN'  # noqa: E501
    simplifier = get_fairseq_simplifier(exp_dir, **generate_kwargs)
    simplifier = get_preprocessed_simplifier(simplifier, preprocessors=preprocessors)
    source_path = get_temp_filepath()
    write_lines(source_sentences, source_path)
    pred_path = simplifier(source_path)
    return read_lines(pred_path)