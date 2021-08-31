import numpy as np
import pandas as pd
import torch

import fontTools.ttLib

import json
import os

from ..util import abspath
from ..model import SDMDDataset, TextRenderer

from typing import Sequence, Tuple


def load_fontset_config(fontset_config_path: str, base_size: int) -> Sequence[dict]:
    fontset_path = os.path.dirname(fontset_config_path)

    with open(fontset_config_path, 'r') as fd:
        fontset_config = json.load(fd)

    for font_spec in fontset_config:
        font_spec['font'] = abspath(font_spec['font'], fontset_path)
        font_spec['size'] = round(font_spec['size'] * base_size)

    return tuple(fontset_config)


def load_data(dataset_config_path: str, size: int, generator: torch.Generator) -> Tuple[SDMDDataset, Sequence[int], Sequence[int]]:
    dataset_config_basepath = os.path.dirname(dataset_config_path)

    with open(dataset_config_path, 'r') as fd:
        dataset_config = json.load(fd)
    
    data_csv_path = abspath(dataset_config['data'], dataset_config_basepath)
    data_keys = tuple(dataset_config['data_keys'])

    data_df = pd.read_csv(data_csv_path, dtype={key: str for key in data_keys}, keep_default_na=False)

    if 'characters' in dataset_config:
        characters_csv_path = abspath(dataset_config['characters'], dataset_config_basepath)
        characters_df = pd.read_csv(characters_csv_path).set_index('codepoint', verify_integrity=True)
    else:
        characters_df = data_df[['codepoint']].drop_duplicates().set_index('codepoint')
        characters_df['train_flg'] = True

    fontset_config = load_fontset_config(abspath(dataset_config['fontset'], dataset_config_basepath), size)

    cmaps = tuple(fontTools.ttLib.TTFont(font_spec['font']).getBestCmap() for font_spec in fontset_config)
    renderers = tuple(TextRenderer(font_spec) for font_spec in fontset_config)

    df_list = []

    for font_idx, cmap in enumerate(cmaps):
        cmap_flg = data_df['codepoint'].isin(cmap.keys())

        aug_df = data_df[cmap_flg].copy()
        aug_df['font'] = font_idx

        df_list.append(aug_df)

    aug_df = pd.concat(df_list, ignore_index=True)
    aug_df.sort_values(['codepoint', 'font'], inplace=True)
    aug_df.reset_index(drop=True, inplace=True)

    train_flg = aug_df[['codepoint']].merge(characters_df[['train_flg']], how='left', left_on=['codepoint'], right_index=True)['train_flg'].fillna(False).values

    train_flg &= np.any((
        aug_df['codepoint'].between(0x4E00, 0x9FFF).values,   # CJK Unified Ideographs
        aug_df['codepoint'].between(0x3400, 0x4DBF).values,   # CJK Unified Ideographs A
    ), axis=0)

    dataset = SDMDDataset(aug_df, data_keys, renderers)

    return dataset, np.nonzero(train_flg)[0], np.nonzero(~train_flg)[0]