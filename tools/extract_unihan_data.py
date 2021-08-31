import pandas as pd

import argparse
import sys

import sdmd.phonology.mandarin

from typing import Any


def extract_mandarin(data: dict, primary_only: bool = False) -> pd.DataFrame:
    records = []

    for codepoint, entry in data.items():
        readings = set()

        if primary_only:
            if 'kMandarin' in entry:
                readings.add(entry['kMandarin'][0])
        else:
            if 'kHanyuPinyin' in entry:
                for subentry in entry['kHanyuPinyin']:
                    readings.update(subentry.split(':', 1)[1].split(','))
        
            if 'kMandarin' in entry:
                readings.update(entry['kMandarin'])
        
        for reading in sorted(readings):
            try:
                syllable = sdmd.phonology.mandarin.parse_pinyin_syllable(reading)
            except ValueError:
                print(f'skipped: U+{codepoint:04X}\t{chr(codepoint)}\t{reading}', file=sys.stderr)
                continue
            
            records.append((codepoint, reading, syllable.initial.value, syllable.medial.value, syllable.nucleus.value, syllable.coda.value, syllable.tone.value))
    
    return pd.DataFrame.from_records(records, columns=('codepoint', 'reading', 'initial', 'medial', 'nucleus', 'coda', 'tone'))


def extract_rs(data: dict, field_name: str, primary_only: bool = False) -> pd.DataFrame:
    records = []

    for codepoint, entry in data.items():
        if field_name not in entry:
            continue

        for subentry in entry[field_name]:
            radical, stroke_count = subentry.split('.', 1)

            if radical.endswith('\''):
                radical_no = int(radical[:-1])
                radical_simpl_flg = True
            else:
                radical_no = int(radical)
                radical_simpl_flg = False
            
            stroke_count = int(stroke_count)

            records.append((codepoint, radical_no, radical_simpl_flg, stroke_count))

            if primary_only:
                break
    
    return pd.DataFrame(records, columns=('codepoint', 'radical_no', 'radical_simpl_flg', 'stroke_count'))


def main(args: Any) -> None:
    with argparse.FileType('r')(args.input) as input_file:
        input_format = args.input_format

        if input_format == 'pickle':
            import pickle
            if hasattr(input_file, 'encoding'):
                input_file = input_file.buffer
            data = pickle.load(input_file)
        elif input_format == 'json':
            import json
            data = json.load(input_file)
        elif input_format == 'yaml':
            import yaml
            data = yaml.dump(input_file)
        else:
            raise ValueError('invalid input format')
    
    select_field = args.select_field
    primary_only = args.primary_only

    if select_field == 'mandarin':
        output = extract_mandarin(data, primary_only=primary_only)
    elif select_field == 'cantonese':
        output = extract_cantonese(data, primary_only=primary_only)
    elif select_field == 'japanese':
        output = extract_japanese(data, primary_only=primary_only)
    elif select_field == 'korean':
        output = extract_korean(data, primary_only=primary_only)
    elif select_field == 'vietnamese':
        output = extract_vietnamese(data, primary_only=primary_only)
    elif select_field == 'rs-unicode':
        output = extract_rs(data, 'kRSUnicode', primary_only=primary_only)
    elif select_field == 'rs-kangxi':
        output = extract_rs(data, 'kRSKangXi', primary_only=primary_only)
    else:
        raise ValueError('invalid field selected')
    
    with argparse.FileType('w')(args.output) as output_file:
        output.to_csv(output_file, index=False)

    return


def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Extract Unihan data.')

    parser.add_argument('-l', '--select-field', default='mandarin', help='data to extract')
    parser.add_argument('-p', '--primary-only', action='store_true', help='extract primary data only')
    parser.add_argument('-t', '--input-format', choices=('pickle', 'json', 'yaml'), default='pickle', help='input format')
    parser.add_argument('-o', '--output', metavar='OUTPUT', default='-', help='output file path')
    parser.add_argument('input', metavar='INPUT', default='-', nargs='?', help='input file path')
    
    return parser


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    main(args)