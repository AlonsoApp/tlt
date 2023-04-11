import torch
from more_itertools import flatten
from transformers import BertTokenizer

SEGMENT_ID_TABLE_TOPIC = 0
SEGMENT_ID_SCHEMA = 1

TABLE_VALUE_SEP = ";"
TABLE_IDX_SEP = ":"

def encode_input(topics, tables, column_names, all_values_extra, tokenizer, max_length_model, device):
    all_input_ids = []
    all_attention_mask = []
    all_segment_ids = []

    all_topic_span_lengths = []
    all_table_token_lengths = []
    all_column_token_lengths = []
    all_values_lengths = []

    for topic, table, columns, values_extra in zip(topics, tables, column_names, all_values_extra):
        topic_tokens, topic_span_lengths, topic_segment_ids = _tokenize_topic(topic, tokenizer)
        all_topic_span_lengths.append(topic_span_lengths)

        table_tokens, table_token_lengths, table_segment_ids = _tokenize_table(table, tokenizer)
        all_table_token_lengths.append(table_token_lengths)

        column_tokens, column_token_lengths, column_segment_ids = _tokenize_column_names(columns, tokenizer)
        all_column_token_lengths.append(column_token_lengths)

        values_extra_tokens, values_extra_token_lengths, values_extra_segment_ids = _tokenize_values_extra(values_extra, tokenizer)
        all_values_lengths.append(values_extra_token_lengths)

        assert sum(topic_span_lengths) + sum(table_token_lengths) + sum(column_token_lengths) + \
               sum(values_extra_token_lengths) == len(topic_tokens) + len(table_tokens) + len(column_tokens) + len(values_extra_tokens)

        tokens = topic_tokens + table_tokens + column_tokens + values_extra_tokens
        if len(tokens) > max_length_model:
            print(
                "################### ATTENTION! Example too long ({}). Topic-len: {}, table-len:{}, columns-len: {}, "
                "values: {} ".format(
                    len(tokens),
                    len(topic_tokens),
                    len(table_tokens),
                    len(column_tokens),
                    len(values_extra_tokens)))
            print(topic)
            print(table)
            print(columns)
            print(values_extra)

        segment_ids = topic_segment_ids + table_segment_ids + column_segment_ids + values_extra_segment_ids
        # not sure here if "tokenizer.mask_token_id" or just a simple 1...
        attention_mask = [1] * len(tokens)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        all_input_ids.append(input_ids)
        all_segment_ids.append(segment_ids)
        all_attention_mask.append(attention_mask)

    max_length_data = max(map(lambda ids: len(ids), all_input_ids))

    for input_ids, segment_ids, attention_mask in zip(all_input_ids, all_segment_ids, all_attention_mask):
        _padd_input(input_ids, segment_ids, attention_mask, max_length_data, tokenizer)

    input_ids_tensor = torch.tensor(all_input_ids, dtype=torch.long).to(device)
    segment_ids_tensor = torch.tensor(all_segment_ids, dtype=torch.long).to(device)
    attention_mask_tensor = torch.tensor(all_attention_mask, dtype=torch.long).to(device)

    return input_ids_tensor, attention_mask_tensor, segment_ids_tensor, (all_topic_span_lengths, all_table_token_lengths, all_column_token_lengths, all_values_lengths)


def _padd_input(input_ids, segment_ids, attention_mask, max_length, tokenizer):

    while len(input_ids) < max_length:
        input_ids.append(tokenizer.pad_token_id)
        attention_mask.append(tokenizer.pad_token_id)
        segment_ids.append(tokenizer.pad_token_id)

    assert len(input_ids) == max_length
    assert len(attention_mask) == max_length
    assert len(segment_ids) == max_length

def _tokenize_topic(topic, tokenizer):
    topic_span_lengths = [1]  # the initial value represents the length of the CLS_TOKEN in the beginning.

    all_sub_token = list(flatten(map(lambda tok: tokenizer.tokenize(tok), topic)))

    for sub_token in all_sub_token:
        if len(sub_token) > 2 and sub_token[0:2] == '##':
            topic_span_lengths[-1] += 1
        else:
            topic_span_lengths.append(1)

    topic_tokens_with_special_chars = [tokenizer.cls_token] + all_sub_token + [tokenizer.sep_token]
    segment_ids = [SEGMENT_ID_TABLE_TOPIC] * len(topic_tokens_with_special_chars)

    # the additional 1 represents the SEP_TOKEN in the end.
    topic_span_lengths.append(1)

    assert sum(topic_span_lengths) == len(topic_tokens_with_special_chars) == len(segment_ids)

    return topic_tokens_with_special_chars, topic_span_lengths, segment_ids

def _tokenize_table(table, tokenizer):
    table_token_lengths = []
    all_table_tokens = []

    for row in table:
        for i, value in enumerate(row):
            value_sub_tokens = tokenizer.tokenize(value)

            # if row n add : if add ; to separate between values or SEP_TOKEN at the end of the row
            if i == 0:
                # row 0 :
                value_sub_tokens += [TABLE_IDX_SEP]
            elif i+1 < len(row):
                # value ;
                value_sub_tokens += [TABLE_VALUE_SEP]
            else:
                # last_value_of_row SEP
                value_sub_tokens += [tokenizer.sep_token]

            all_table_tokens.extend(value_sub_tokens)
            table_token_lengths.append(len(value_sub_tokens))

    segment_ids = [SEGMENT_ID_TABLE_TOPIC if tok == tokenizer.sep_token or tok == TABLE_VALUE_SEP else SEGMENT_ID_SCHEMA for tok in
                   all_table_tokens]

    assert sum(table_token_lengths) == len(all_table_tokens) == len(segment_ids)

    return all_table_tokens, table_token_lengths, segment_ids

def _tokenize_column_names(column_names, tokenizer):
    column_token_lengths = []
    all_column_tokens = []

    for column in column_names:
        column_sub_tokens = tokenizer.tokenize(column)

        column_sub_tokens += [tokenizer.sep_token]

        all_column_tokens.extend(column_sub_tokens)
        column_token_lengths.append(len(column_sub_tokens))

    segment_ids = [SEGMENT_ID_TABLE_TOPIC if tok == tokenizer.sep_token else SEGMENT_ID_SCHEMA for tok in all_column_tokens]

    return all_column_tokens, column_token_lengths, segment_ids

def _tokenize_values_extra(values_extra, tokenizer):
    values_extra_token_lengths = []
    all_values_extra_tokens = []

    for value in values_extra:
        values_extra_sub_tokens = tokenizer.tokenize(value)

        values_extra_sub_tokens += [tokenizer.sep_token]

        all_values_extra_tokens.extend(values_extra_sub_tokens)
        values_extra_token_lengths.append(len(values_extra_sub_tokens))

    segment_ids = [SEGMENT_ID_TABLE_TOPIC if tok == tokenizer.sep_token else SEGMENT_ID_SCHEMA for tok in all_values_extra_tokens]

    return all_values_extra_tokens, values_extra_token_lengths, segment_ids


if __name__ == '__main__':
    topic = ['2011', 'british', 'gt', 'season', 'holaquetal']
    table = [['row 0', '1', 'oulton park', '25 april', '60 mins', 'no 5 scuderia vittoria', 'no 1 trackspeed', 'no 44 abg motorsport'],
             ['row 1', '1', 'oulton park', '25 april', '60 mins', 'charles bateman michael lyons',
              'david ashburn richard westbrook', 'peter belshaw marcus clutton'],
             ['row 2', '2', 'oulton park', '25 april', '60 mins', 'no 1 trackspeed', 'no 5 scuderia vittoria',
              'no 42 century motorsport'],
             ['row 3', '2', 'oulton park', '25 april', '60 mins', 'david ashburn richard westbrook',
              'charles bateman michael lyons', 'jake rattenbury josh wakefield'],
             ['row 4', '3', 'snetterton 300', '15 may', '120 mins', 'no 10 crs racing', 'no 23 united autosports',
              'no 44 abg motorsport'], ['row 5', '3', 'snetterton 300', '15 may', '120 mins', 'glynn geddie jim geddie',
                                        'matt bell michael guasch', 'peter belshaw marcus clutton'],
             ['row 6', '4', 'brands hatch', '19 june', '120 mins', 'no 21 mtech', 'no 2 trackspeed',
              'no 44 abg motorsport']]
    columns = ['round', 'circuit', 'date', 'length', 'pole position', 'gt3 winner', 'gt4 winner']
    values = ['12']
    logic = 'eq { count { filter_eq { all_rows ; length ; 60 mins } } ; 12 } = true'
    sent = 'there were twelve occasions where the length was sixty minutes .'

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    input_ids_tensor, attention_mask_tensor, segment_ids_tensor, input_lengths = encode_input([topic], [table], [columns], [values], tokenizer, 512, torch.device("cpu"))

    print(":D")


