import numpy as np
import pandas as pd
import torch
from torch import random
import torch.utils.data

import PIL.ImageFont

import collections

from typing import Dict, OrderedDict, Sequence, Tuple, Union


class TextRenderer:
    _font_params: dict
    _font: PIL.ImageFont.ImageFont

    _torch_dtype: torch.dtype

    def __init__(self, font_params: dict):
        self._font_params = font_params.copy()
        self._font = None
     
        # self._torch_dtype = torch.get_default_dtype()
        self._torch_dtype = torch.uint8

    def __getstate__(self):
        return {
            '_font_params': self._font_params,
            '_font': None,
            '_torch_dtype': self._torch_dtype,
        }

    @property
    def font(self) -> PIL.ImageFont.ImageFont:
        font = self._font

        if font is None:
            font = PIL.ImageFont.truetype(**self._font_params)
            self._font = font

        return font

    def render_text(self, text: str) -> torch.Tensor:
        mask = self.font.getmask(text, mode='1')

        result = torch.from_numpy(np.array(mask).reshape(mask.size, order='F') // 255).to(dtype=self._torch_dtype)
       
        return result


class SDMDDataset(torch.utils.data.Dataset):
    _fonts: Sequence[TextRenderer]
    _control_data: Sequence[Tuple[int, int]]
    _label_data: torch.Tensor
    _category_dict: OrderedDict[str, Sequence]
    _num_distinct: int

    def __init__(self, readings_df: pd.DataFrame, data_keys: Sequence[str], fonts: Sequence[TextRenderer]):
        control_data = tuple(readings_df[['codepoint', 'font']].itertuples(index=False, name=None))

        label_data = np.empty((len(control_data), len(data_keys)), order='F', dtype=np.int64)

        category_dict = collections.OrderedDict()

        for group_idx, key in enumerate(data_keys):
            data_col = readings_df[key].astype('category')
            label_data[:, group_idx] = data_col.cat.codes.values
            category_dict[key] = data_col.cat.categories

        label_data = torch.from_numpy(np.ascontiguousarray(label_data))

        self._fonts = tuple(fonts)
        self._control_data = control_data
        self._label_data = label_data
        self._category_dict = category_dict
        self._num_distinct = torch.unique(label_data, dim=0).shape[0]

    
    def __len__(self):
        return len(self._control_data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        codepoint, font_idx = self._control_data[idx]
        char = chr(codepoint)
        font = self._fonts[font_idx]

        glyph_tensor = font.render_text(char)

        # data = torch.zeros(self._num_categories, dtype=torch_dtype)

        # for i, (key, categories, code_offset, next_code_offset) in enumerate(self._category_groups):
        #     code = sample[i + 2]

        #     if code < 0:
        #         # missing value
        #         continue
            
        #     data[code_offset + code] = 1

        return glyph_tensor, self._label_data[idx]
    
    @property
    def category_dict(self):
        return self._category_dict

    @property
    def num_categories_dict(self):
        return collections.OrderedDict((key, len(categories)) for key, categories in self._category_dict.items())

    @property
    def num_distinct(self):
        return self._num_distinct


class AugmentingDataset(torch.utils.data.Dataset):
    _dataset: torch.utils.data.Dataset

    _torch_dtype: torch.dtype
    _size: int
    _random_translation_max: int
    _random_hole_probability: float
    _random_hole_size_max: int
    _return_original: bool

    _cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]]

    def __init__(self, dataset: torch.utils.data.Dataset, size: int, random_translation_max: int = 0, random_hole_probability: float = 0.0, random_hole_size_max: int = 0, do_memoize: bool = False, return_original: bool = False):
        self._dataset = dataset

        self._torch_dtype = torch.get_default_dtype()
        self._size = int(size)
        self._random_translation_max = int(random_translation_max)
        self._random_hole_probability = float(random_hole_probability)
        self._random_hole_size_max = int(random_hole_size_max)
        self._return_original = bool(return_original)

        if do_memoize:
            self._cache = dict()
        else:
            self._cache = None

    def __getstate__(self):
        return {
            '_dataset': self._dataset,
            '_torch_dtype': self._torch_dtype,
            '_size': self._size,
            '_random_translation_max': self._random_translation_max,
            '_return_original': self._return_original,
            '_cache': None if self._cache is None else dict(),
        }
    
    def __len__(self):
        return len(self._dataset)

    def _getitem_inner(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        cache = self._cache

        if cache is None:
            ret = None
        else:
            ret = cache.get(idx, None)
        
        if ret is None:
            ret = self._dataset[idx]
        
            if cache is not None:
                cache[idx] = ret
        
        return ret

    def _copy_to_canvas(self, glyph_tensor: torch.Tensor, left: int, top: int, canvas_width: int, canvas_height: int, dtype: torch.dtype) -> torch.Tensor:
        img_tensor = torch.zeros((1, canvas_width, canvas_height), dtype=dtype)

        glyph_shape = glyph_tensor.shape

        bottom = top + glyph_shape[1]
        right = left + glyph_shape[0]

        canvas_top = max(top, 0)
        canvas_left = max(left, 0)
        canvas_bottom = min(bottom, canvas_height)
        canvas_right = min(right, canvas_width)

        glyph_top = max(-top, 0)
        glyph_left = max(-left, 0)
        glyph_bottom = glyph_top + (canvas_bottom - canvas_top)
        glyph_right = glyph_left + (canvas_right - canvas_left)

        img_tensor[0, canvas_left:canvas_right, canvas_top:canvas_bottom] = glyph_tensor[glyph_left:glyph_right, glyph_top:glyph_bottom]
       
        # img_tensor -= 0.25

        return img_tensor

    def __getitem__(self, idx: int) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        glyph_tensor, label = self._getitem_inner(idx)

        torch_dtype = self._torch_dtype

        size = self._size

        return_original = self._return_original

        glyph_shape = glyph_tensor.shape

        random_translation_max = self._random_translation_max

        orig_top = (size - glyph_shape[1]) // 2
        orig_left = (size - glyph_shape[0]) // 2

        if random_translation_max > 0:
            top = orig_top + torch.randint(-random_translation_max, random_translation_max + 1, ()).item()
            left = orig_left + torch.randint(-random_translation_max, random_translation_max + 1, ()).item()
        else:
            top = orig_top
            left = orig_left

        img_tensor = self._copy_to_canvas(glyph_tensor, left, top, size, size, torch_dtype)

        if return_original:
            if top == orig_top and left == orig_left:
                orig_img_tensor = img_tensor.clone()
            else:
                orig_img_tensor = self._copy_to_canvas(glyph_tensor, orig_left, orig_top, size, size, torch_dtype)
        
        make_hole = (torch.rand(()).item() <= self._random_hole_probability)

        if make_hole:
            hole_size = round(torch.sqrt(torch.rand(())).item() * self._random_hole_size_max)

            hole_top = torch.randint(0, size - hole_size + 1, ()).item()
            hole_left = torch.randint(0, size - hole_size + 1, ()).item()
            hole_bottom = hole_top + hole_size
            hole_right = hole_left + hole_size

            img_tensor[0, hole_left:hole_right, hole_top:hole_bottom] = 0

        if return_original:
            return img_tensor, label, orig_img_tensor
        else:
            return img_tensor, label