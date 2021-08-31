import argparse

from typing import Any


multivalue_delims = {
    'kAccountingNumeric': ' ',
    'kCantonese': ' ',
    'kCCCII': ' ',
    'kCheungBauer': ' ',
    'kCheungBauerIndex': ' ',
    'kCihaiT': ' ',
    'kCowles': ' ',
    'kFenn': ' ',
    'kFennIndex': ' ',
    'kFourCornerCode': ' ',
    'kGSR': ' ',
    'kHangul': ' ',
    'kHanYu': ' ',
    'kHanyuPinlu': ' ',
    'kHanyuPinyin': ' ',
    'kHKGlyph': ' ',
    'kIBMJapan': ' ',
    'kIICore': ' ',
    'kIRGDaeJaweon': ' ',
    'kIRGDaiKanwaZiten': ' ',
    'kIRGHanyuDaZidian': ' ',
    'kIRGKangXi': ' ',
    'kJa': ' ',
    'kJapaneseKun': ' ',
    'kJapaneseOn': ' ',
    'kJinmeiyoKanji': ' ',
    'kJis0': ' ',
    'kJIS0213': ' ',
    'kJis1': ' ',
    'kJoyoKanji': ' ',
    'kKangXi': ' ',
    'kKarlgren': ' ',
    'kKorean': ' ',
    'kKoreanEducationHanja': ' ',
    'kKoreanName': ' ',
    'kKPS0': ' ',
    'kKPS1': ' ',
    'kKSC0': ' ',
    'kKSC1': ' ',
    'kLau': ' ',
    'kMainlandTelegraph': ' ',
    'kMandarin': ' ',
    'kMatthews': ' ',
    'kMeyerWempe': ' ',
    'kMorohashi': ' ',
    'kNelson': ' ',
    'kOtherNumeric': ' ',
    'kPhonetic': ' ',
    'kPrimaryNumeric': ' ',
    'kRSAdobe_Japan1_6': ' ',
    'kRSKangXi': ' ',
    'kRSUnicode': ' ',
    'kSBGY': ' ',
    'kSemanticVariant': ' ',
    'kSimplifiedVariant': ' ',
    'kSpecializedSemanticVariant': ' ',
    'kSpoofingVariant': ' ',
    'kTaiwanTelegraph': ' ',
    'kTang': ' ',
    'kTGH': ' ',
    'kTGHZ2013': ' ',
    'kTotalStrokes': ' ',
    'kTraditionalVariant': ' ',
    'kVietnamese': ' ',
    'kXerox': ' ',
    'kXHC1983': ' ',
    'kZVariant': ' ',
}


def read_unihan_data(lines, out_data: dict) -> None:
    for line_number, line in enumerate(lines):
        line = line.strip()
        if len(line) == 0:
            continue

        if line.startswith('#'):
            continue

        codepoint, key, value = line.split('\t', 2)
        
        assert(codepoint.startswith('U+'))

        codepoint = int(codepoint[2:], 16)

        if codepoint in out_data:
            entry = out_data[codepoint]
        else:
            entry = dict()
            out_data[codepoint] = entry
        
        assert(key not in entry)

        if key in multivalue_delims:
            value = tuple(value.split(multivalue_delims[key]))

        entry[key] = value
    
    return


def main(args: Any) -> None:
    data = dict()

    for input_path in args.inputs:
        with argparse.FileType('r')(input_path) as input_file:
            read_unihan_data(input_file, data)

    with argparse.FileType('w')(args.output) as output_file:
        output_format = args.output_format

        if output_format == 'pickle':
            import pickle
            if hasattr(output_file, 'encoding'):
                output_file = output_file.buffer
            pickle.dump(data, output_file)
        elif output_format == 'json':
            import json
            json.dump(data, output_file)
        elif output_format == 'yaml':
            import yaml
            yaml.dump(data, output_file)
        else:
            raise ValueError('invalid output format')

    return


def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Convert a Unihan database file.')

    parser.add_argument('-t', '--output-format', choices=('pickle', 'json', 'yaml'), default='pickle', help='output format')
    parser.add_argument('-o', '--output', metavar='OUTPUT', default='-', help='output file path')
    parser.add_argument('inputs', metavar='INPUT', default=['-'], nargs='*', help='input file path')

    return parser


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    main(args)
