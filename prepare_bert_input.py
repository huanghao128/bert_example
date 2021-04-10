import numpy as np
import pandas as pd
import glob
import re
import logging
import json
import random

from tokenization import BertTokenizer

logging.basicConfig(level=logging.INFO, 
                    format='[%(asctime)s.%(msecs)03d %(levelname)s %(filename)s:%(lineno)d] %(message)s', 
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)


class InputExample(object):
    def __init__(self, guid, tokens_a, tokens_b=None, is_next=None, lm_labels=None):
        self.guid = guid
        self.tokens_a = tokens_a
        self.tokens_b = tokens_b
        self.is_next = is_next


class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self, input_ids, input_mask, segment_ids, masked_lm_positions, masked_lm_ids, is_next):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.masked_lm_positions = masked_lm_positions
        self.masked_lm_ids = masked_lm_ids
        self.is_next = is_next

    def __str__(self):
        s = self.__class__.__name__ + "(\n"
        s += "input_ids: %s\n" % str(self.input_ids)
        s += "input_mask: %s\n" % str(self.input_mask)
        s += "segment_ids: %s\n" % str(self.segment_ids)
        s += "masked_lm_positions: %s\n" % str(self.masked_lm_positions)
        s += "masked_lm_ids: %s\n" % str(self.masked_lm_ids)
        s += "is_next: %s)" % self.is_next
        return s


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def create_pair_document(all_documents, doc_index, max_seq_length, short_seq_prob):
    document = all_documents[doc_index]

    # Account for [CLS], [SEP], [SEP]
    max_num_tokens = max_seq_length - 3

    target_seq_length = max_num_tokens
    if random.random() < short_seq_prob:
        target_seq_length = random.randint(2, max_num_tokens)

    doc_pair_tokens = []
    current_chunk = []
    current_length = 0
    i = 0
    while i < len(document):
        segment = document[i]
        current_chunk.append(segment)
        current_length += len(segment)

        if i == len(document)-1 or current_length >= target_seq_length:
            if current_chunk:
                a_end = 1
                if len(current_chunk) >= 2:
                    a_end = random.randint(1, len(current_chunk) - 1)
                
                tokens_a = []
                for j in range(a_end):
                    tokens_a.append(current_chunk[j])
                
                tokens_b = []
                is_random_next = 0
                if len(current_chunk) == 1 or random.random() < 0.5:
                    is_random_next = 1
                    target_b_length = target_seq_length - len(tokens_a)
                    for _ in range(10):
                        random_doc_index = random.randint(0, len(all_documents)-1)
                        if random_doc_index != doc_index:
                            break
                    random_docment = all_documents[random_doc_index]
                    random_start = random.randint(0, len(random_docment)-1)
                    for j in range(random_start, len(random_docment)):
                        tokens_b.append(random_docment[j])
                        if len(tokens_b) >= target_b_length:
                            break
                    num_unused_segments = len(current_chunk) - a_end
                    i -= num_unused_segments
                else:
                    is_random_next = 0
                    for j in range(a_end, len(current_chunk)):
                        tokens_b.append(current_chunk[j])

                assert len(tokens_a) >= 1
                assert len(tokens_b) >= 1

                doc_pair_tokens.append([tokens_a, tokens_b, is_random_next])

            current_chunk = []
            current_length = 0
            
        i += 1
    
    return doc_pair_tokens


def random_word(tokens, tokenizer):
    output_label = []

    for i, token in enumerate(tokens):
        prob = random.random()

        if prob < 0.85:
            output_label.append(-1)
        else:
            # prob /= 0.15
            prob = (1.0 - prob) / 0.15
            if prob < 0.8:
                tokens[i] = "[MASK]"
            elif prob < 0.9:
                tokens[i] = random.choice(list(tokenizer.vocab.items()))[0]
            # 10% keep current token
            try:
                output_label.append(tokenizer.vocab[token])
            except KeyError:
                output_label.append(tokenizer.vocab["[UNK]"])

    return tokens, output_label


def create_masked_lm_labels(tokens, masked_lm_prob, max_pred_per_seq, tokenizer):
    cand_indexs = []
    for i, token in enumerate(tokens):
        if token == "[CLS]" or token == "[SEP]":
            continue
        cand_indexs.append([i])
    
    random.shuffle(cand_indexs)

    output_tokens = list(tokens)

    num_to_predict = min(max_pred_per_seq, max(1, int(round(len(tokens) * masked_lm_prob))))

    masked_lms = []
    covered_indexes = set()
    for index_set in cand_indexs:
        if len(masked_lms) >= num_to_predict:
            break
        
        is_any_index_covered = False
        for index in index_set:
            if index in covered_indexes:
                is_any_index_covered = True
                break
        if is_any_index_covered:
            continue

        for index in index_set:
            covered_indexes.add(index)

            masked_token = None
            prob = random.random()

             # 80% of the time, replace with [MASK]
            if prob < 0.8:
                masked_token = "[MASK]"
            else:
                # 10% of the time, keep original
                if prob < 0.5:
                    masked_token = tokens[index]
                else: # 10% of the time, replace with random word
                    masked_token = random.choice(list(tokenizer.vocab.items()))[0]
            
            output_tokens[index] = masked_token
            masked_lms.append((index, tokens[index]))
        
    masked_lms = sorted(masked_lms, key=lambda x: x[0])

    masked_lm_positions = []
    masked_lm_labels = []
    for (index, label) in masked_lms:
        masked_lm_positions.append(index)
        masked_lm_labels.append(label)
    
    return output_tokens, masked_lm_positions, masked_lm_labels


def convert_examples_to_features(example, seq_length, max_pred_per_seq, tokenizer):
    tokens_a = example.tokens_a
    tokens_b = example.tokens_b
    _truncate_seq_pair(tokens_a, tokens_b, seq_length - 3)

    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    assert len(tokens_b) > 0
    for token in tokens_b:
        tokens.append(token)
        segment_ids.append(1)
    tokens.append("[SEP]")
    segment_ids.append(1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)

    while len(input_ids) < seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    tokens, masked_lm_positions, masked_lm_labels = create_masked_lm_labels(
        tokens, masked_lm_prob=0.15, max_pred_per_seq=max_pred_per_seq, tokenizer=tokenizer)

    masked_lm_ids = tokenizer.convert_tokens_to_ids(masked_lm_labels)

    while len(masked_lm_positions) < max_pred_per_seq:
        masked_lm_positions.append(0)
        masked_lm_ids.append(0)
    
    assert len(input_ids) == seq_length
    assert len(input_mask) == seq_length
    assert len(segment_ids) == seq_length

    if example.guid < 5:
        logger.info("*** Example ***")
        logger.info("guid: %s" % (example.guid))
        logger.info("tokens: %s" % str(tokens))
        logger.info("input_ids: %s" % str(input_ids))
        logger.info("input_mask: %s" % str(input_mask))
        logger.info("segment_ids: %s" % str(segment_ids))
        logger.info("masked_lm_positions: %s" % str(masked_lm_positions))
        logger.info("masked_lm_ids: %s" % str(masked_lm_ids))
        logger.info("next_sentence_labels: %s " % (example.is_next))

    features = InputFeatures(input_ids=input_ids,
                             input_mask=input_mask,
                             segment_ids=segment_ids,
                             masked_lm_positions=masked_lm_positions,
                             masked_lm_ids=masked_lm_ids,
                             is_next=example.is_next)
    return features


def test2():
    sentences = ["I got this movie used and wouldn't buy it new.", 
            "I suggest you skip this movie.", 
            "all the sex is treated as being.", 
            "than at least perverse and degrading."]

    max_pred_per_seq = 10
    max_seq_length = 32

    tokenizer = BertTokenizer("data/vocab.txt")

    all_documents = [tokenizer.tokenize(sent) for sent in sentences]

    examples = []
    i  = 0
    for doc_index in range(len(all_documents)):
        doc_pair_tokens = create_pair_document(all_documents, doc_index, max_seq_length, short_seq_prob=0.1)
        for tokens_a, tokens_b, is_next in doc_pair_tokens:
            example = InputExample(i, tokens_a, tokens_b, is_next, lm_labels=None)
            examples.append(example)
            i += 1

    for example in examples:
        feature = convert_examples_to_features(example, seq_length=32, max_pred_per_seq=5, tokenizer=tokenizer)
        # logger.info(feature)


def test():
    sample1 = ["I got this movie used and wouldn't buy it new.", "I suggest you skip this movie."]
    sample2 = ["all the sex is treated as being.", "than at least perverse and degrading."]

    vocab_file = "data/vocab.txt"

    tokenizer = BertTokenizer(vocab_file)

    tokens1 = [tokenizer.tokenize(sent) for sent in sample1]
    tokens2 = [tokenizer.tokenize(sent) for sent in sample2]

    examples = [InputExample(1, tokens1[0], tokens1[1], is_next=1, lm_labels=None),
                InputExample(2, tokens2[0], tokens2[1], is_next=1, lm_labels=None)]

    for example in examples:
        feature = convert_examples_to_features(example, seq_length=32, max_pred_per_seq=5, tokenizer=tokenizer)
        logger.info(feature)


if __name__ == '__main__':
    # test()
    test2()