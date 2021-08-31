from typing import Any, Tuple
from sdmd.phonology import mandarin
import unicodedata

from .syllable import MandarinCoda, MandarinInitial, MandarinMedial, MandarinNucleus, MandarinSyllable, MandarinTone

pinyin_tone_dict = {
    # longest keys first
    # 'm̄': ('m', 1),
    # 'm̀': ('m', 4),
    'ê̄': ('ê', 1),
    'ế': ('ê', 2),
    'ê̌': ('ê', 3),
    'ề': ('ê', 4),
    'ā': ('a', 1),
    'á': ('a', 2),
    'ǎ': ('a', 3),
    'à': ('a', 4),
    'ō': ('o', 1),
    'ó': ('o', 2),
    'ǒ': ('o', 3),
    'ò': ('o', 4),
    'ē': ('e', 1),
    'é': ('e', 2),
    'ě': ('e', 3),
    'è': ('e', 4),
    'ī': ('i', 1),
    'í': ('i', 2),
    'ǐ': ('i', 3),
    'ì': ('i', 4),
    'ū': ('u', 1),
    'ú': ('u', 2),
    'ǔ': ('u', 3),
    'ù': ('u', 4),
    'ǖ': ('ü', 1),
    'ǘ': ('ü', 2),
    'ǚ': ('ü', 3),
    'ǜ': ('ü', 4),
    # 'ń': ('n', 2),
    # 'ň': ('n', 3),
    # 'ǹ': ('n', 4),
    # 'ḿ': ('m', 2),
}

pinyin_startswith_replacement_dict = {
    # longest keys first
    'zhi': 'zh',
    'chi': 'ch',
    'shi': 'sh',
    'ri': 'r',
    'zi': 'z',
    'ci': 'c',
    'si': 's',
    'ju': 'jü',
    'qu': 'qü',
    'xu': 'xü',
    'yi': 'i',
    'wu': 'u',
    'yu': 'ü',
    'y': 'i',
    'w': 'u',
}

pinyin_endswith_replacement_dict = {
    # longest keys first
    'iong': 'üong',
    'uong': 'uong',
    'ing': 'ieng',
    'ong': 'ueng',
    'ao': 'au',
    'ie': 'iê',
    'üe': 'üê',
    'in': 'ien',
    'iu': 'iou',
    'ui': 'uei',
    'un': 'uen',
    'ün': 'üen',
}

pinyin_initial_dict = {
    # longest keys first
    'zh': MandarinInitial.ZH,
    'ch': MandarinInitial.CH,
    'sh': MandarinInitial.SH,
    'r': MandarinInitial.R,
    'b': MandarinInitial.B,
    'p': MandarinInitial.P,
    'm': MandarinInitial.M,
    'f': MandarinInitial.F,
    'd': MandarinInitial.D,
    't': MandarinInitial.T,
    'n': MandarinInitial.N,
    'l': MandarinInitial.L,
    'g': MandarinInitial.G,
    'k': MandarinInitial.K,
    'h': MandarinInitial.H,
    'j': MandarinInitial.J,
    'q': MandarinInitial.Q,
    'x': MandarinInitial.X,
    'z': MandarinInitial.Z,
    'c': MandarinInitial.C,
    's': MandarinInitial.S,
    '': MandarinInitial._
}

pinyin_medial_dict = {
    # longest keys first
    'i': MandarinMedial.I,
    'u': MandarinMedial.U,
    'ü': MandarinMedial.V,
    '': MandarinMedial._
}

pinyin_nucleus_dict = {
    # longest keys first
    'a': MandarinNucleus.A,
    'e': MandarinNucleus.E,
    'ê': MandarinNucleus.EH,
    'o': MandarinNucleus.O,
    '': MandarinNucleus._
}

pinyin_coda_dict = {
    # longest keys first
    'ng': MandarinCoda.NG,
    'n': MandarinCoda.N,
    'r': MandarinCoda.R,
    'i': MandarinCoda.I,
    'u': MandarinCoda.U,
    '': MandarinCoda._
}


def _parse_inner(input: str, token_dict: dict) -> Tuple[Any, str]:
    for key, value in token_dict.items():
        if input.startswith(key):
            return value, input[len(key):]
    

def parse_pinyin_syllable(pinyin_syllable: str, skip_normalization: bool = False) -> MandarinSyllable:
    if not skip_normalization:
        pinyin_syllable = unicodedata.normalize('NFKC', pinyin_syllable).lower()

    tone = MandarinTone._0

    for key, value in pinyin_tone_dict.items():
        if key in pinyin_syllable:
            plain, tone_number = value
            tone = MandarinTone(tone_number)
            pinyin_syllable = pinyin_syllable.replace(key, plain, 1)
            break    

    for key, value in pinyin_startswith_replacement_dict.items():
        if pinyin_syllable.startswith(key):
            pinyin_syllable = value + pinyin_syllable[len(key):]
            break
    
    for key, value in pinyin_endswith_replacement_dict.items():
        if pinyin_syllable.endswith(key):
            pinyin_syllable = pinyin_syllable[:-len(key)] + value
            break

    initial, pinyin_rest = _parse_inner(pinyin_syllable, pinyin_initial_dict)
    medial, pinyin_rest = _parse_inner(pinyin_rest, pinyin_medial_dict)
    nucleus, pinyin_rest = _parse_inner(pinyin_rest, pinyin_nucleus_dict)
    coda, pinyin_rest = _parse_inner(pinyin_rest, pinyin_coda_dict)

    if len(pinyin_rest) > 0:
        raise ValueError('invalid pinyin syllable')

    return MandarinSyllable(initial, medial, nucleus, coda, tone)


