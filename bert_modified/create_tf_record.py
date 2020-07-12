# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Create masked LM/next sentence masked_lm TF examples for BERT."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import random
import tokenization
import tensorflow as tf
import pickle

#获取token的相关MASK概率
MASK_PROB = pickle.load(open('mask_probability.sav', 'rb'))
WRONG_COUNT = dict([(k, 0) for k in MASK_PROB])
CORRECT_COUNT = dict([(k, 0) for k in MASK_PROB])

MaskedLmInstance = collections.namedtuple("MaskedLmInstance",
                                          ["index", "label"])

flags = tf.flags

FLAGS = flags.FLAGS
#定义输入文件
flags.DEFINE_string("input_file", None,
                    "Input raw text file (or comma-separated list of files).")
#定义wrong输入文件
flags.DEFINE_string("wrong_input_file", None,
                    "same as input_file except containing wrong characters.")
#定义输出文件 一般为tf_examples.tfrecord
flags.DEFINE_string(
    "output_file", None,
    "Output TF example file (or comma-separated list of files).")
#定义词表文件 一般为bert中的vocab.txt
flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")
#定义是否转小写
flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")
#定义最大seq长度
flags.DEFINE_integer("max_seq_length", 128, "Maximum sequence length.")
#定义可预测的最大长度 也就是最多mask20个token
flags.DEFINE_integer("max_predictions_per_seq", 20,
                     "Maximum number of masked LM predictions per sequence.")
#定义随机种子
flags.DEFINE_integer("random_seed", 12345, "Random seed for data generation.")

#定义重复次数 默认10 也就是对于一个输入 进行10次重复的mask操作（每次mask的位置会不一样）
flags.DEFINE_integer(
    "dupe_factor", 10,
    "Number of times to duplicate the input data (with different masks).")
#定义mask概率
flags.DEFINE_float("masked_lm_prob", 0.15, "Masked LM probability.")
#定义创建seq时比最大长度短的概率
flags.DEFINE_float(
    "short_seq_prob", 0.1,
    "Probability of creating sequences which are shorter than the "
    "maximum length.")

#定义训练实例
class TrainingInstance(object):
    """A single training instance (sentence pair)."""

    def __init__(self, tokens, segment_ids, masked_lm_positions, masked_lm_labels,
                 is_random_next):
        self.tokens = tokens #训练实例对应的tokens ([CLS] 我 很 ...)
        self.segment_ids = segment_ids #训练实例对应的segment_ids (00001111...)
        self.is_random_next = is_random_next #训练实例对应的两个句子之间是否构成上下文关系
        self.masked_lm_positions = masked_lm_positions #训练实例对应的mask位置  (001001...)
        self.masked_lm_labels = masked_lm_labels #训练实例对应被mask掉的原token，也就是lebel

    #将训练实例字符串化 方便输出展示
    def __str__(self):
        s = ""
        s += "tokens: %s\n" % (" ".join(
            [tokenization.printable_text(x) for x in self.tokens]))
        s += "segment_ids: %s\n" % (" ".join([str(x) for x in self.segment_ids]))
        s += "is_random_next: %s\n" % self.is_random_next
        s += "masked_lm_positions: %s\n" % (" ".join(
            [str(x) for x in self.masked_lm_positions]))
        s += "masked_lm_labels: %s\n" % (" ".join(
            [tokenization.printable_text(x) for x in self.masked_lm_labels]))
        s += "\n"
        return s

    def __repr__(self):
        return self.__str__()

#将单一的训练实例转为example并写入TF
def write_instance_to_example_files(instances, tokenizer, max_seq_length,
                                    max_predictions_per_seq, output_files):
    """Create TF example files from `TrainingInstance`s."""
    #根据传入的写入TF文件列表（output_files 多个）每一个TF文件创造一个写入writer
    writers = []
    for output_file in output_files:
        writers.append(tf.python_io.TFRecordWriter(output_file))

    writer_index = 0

    total_written = 0
    for (inst_index, instance) in enumerate(instances):
        #获取每一个实例中的相关字段信息
        input_ids = tokenizer.convert_tokens_to_ids(instance.tokens)
        input_mask = [1] * len(input_ids)
        segment_ids = list(instance.segment_ids)
        assert len(input_ids) <= max_seq_length #断定input_ids是否小于等于最大输入sequence长度

        while len(input_ids) < max_seq_length:#对于小于的部分 用0补齐
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length #补齐后再一次断定
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        masked_lm_positions = list(instance.masked_lm_positions) #获取实例中要mask的位置
        masked_lm_ids = tokenizer.convert_tokens_to_ids(instance.masked_lm_labels) #获取实例中被mask掉的位置对应的label
        masked_lm_weights = [1.0] * len(masked_lm_ids) #定义被mask掉的lm_weights

        #对于每一个序列最多可以mask掉的长度 并用0进行补齐
        while len(masked_lm_positions) < max_predictions_per_seq:
            masked_lm_positions.append(0)
            masked_lm_ids.append(0)
            masked_lm_weights.append(0.0)

        #确定两个句子之间是否够长上下文关系
        next_sentence_label = 1 if instance.is_random_next else 0

        #建立feature和example的映射 实现将训练实例序列化写入tf文件中
        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(input_ids)
        features["input_mask"] = create_int_feature(input_mask)
        features["segment_ids"] = create_int_feature(segment_ids)
        features["masked_lm_positions"] = create_int_feature(masked_lm_positions)
        features["masked_lm_ids"] = create_int_feature(masked_lm_ids)
        features["masked_lm_weights"] = create_float_feature(masked_lm_weights)
        features["next_sentence_labels"] = create_int_feature([next_sentence_label])

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        #写入到相应的TF文件中
        writers[writer_index].write(tf_example.SerializeToString())
        writer_index = (writer_index + 1) % len(writers)

        #记录写入到Tf中的总example
        total_written += 1

        #输出写入的前20条信息
        if inst_index < 20:
            # pass
            tf.logging.info("*** Example ***")
            tf.logging.info("tokens: %s" % " ".join(
                [tokenization.printable_text(x) for x in instance.tokens]))

            for feature_name in features.keys():
                feature = features[feature_name]
                values = []
                if feature.int64_list.value:
                    values = feature.int64_list.value
                elif feature.float_list.value:
                    values = feature.float_list.value
                tf.logging.info(
                    "%s: %s" % (feature_name, " ".join([str(x) for x in values])))

    #输出在MASK后 在所有实例中 每个token（token来自mask_probability）被mask的次数（只输出前10个）
    for k in sorted(list(WRONG_COUNT.keys()), key=lambda x: WRONG_COUNT[x] + CORRECT_COUNT[x], reverse=True)[:10]:
        print(f'correct {k} is masked {WRONG_COUNT[k] + CORRECT_COUNT[k]} times in all instances')

    #关闭writer
    for writer in writers:
        writer.close()
    #打印总的examples数目
    tf.logging.info("Wrote %d total instances", total_written)

#创建feature和example映射是的feature定义 因为所有的字段都是（id）int64位 所以使用tf.train.Int64List
def create_int_feature(values):
    feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
    return feature

#创建feature和example映射是的feature定义 因为所有的字段都是（id）int64位 所以使用tf.train.FloatList
def create_float_feature(values):
    feature = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
    return feature


#根据输入的correct wrong 和 tokenizer 构造训练实例
def create_training_instances(input_files, tokenizer, max_seq_length,
                              dupe_factor, short_seq_prob, masked_lm_prob,
                              max_predictions_per_seq, rng, wrong_input_files):
    """Create `TrainingInstance`s from raw text."""
    all_documents = [[]] #记录所有的correct lines 采用列表中的列表 形式进行记录

    # Input file format:
    # (1) One sentence per line. These should ideally be actual sentences, not
    # entire paragraphs or arbitrary spans of text. (Because we use the
    # sentence boundaries for the "next sentence prediction" task).
    # (2) Blank lines between documents. Document boundaries are needed so
    # that the "next sentence prediction" task doesn't span between documents.
    for input_file in input_files:
        with tf.gfile.GFile(input_file, "r") as reader:
            while True:
                #逐行读取输入文件
                line = tokenization.convert_to_unicode(reader.readline())
                if not line:#稳当读取完毕 退出循环
                    break
                line = line.strip()#去除末尾的换行符

                # Empty lines are used as document delimiters
                if not line:#读取到空行
                    all_documents.append([]) #之所以在构建correct和wrong时候需要有空行做间隔 就是为了在这里进行数据处理 也就是所说的文档间隔
                tokens = tokenizer.tokenize(line) #对line进行tokenize
                if tokens:#将tokenize后的line加入到all_docuents最后一个子元素中
                    all_documents[-1].append(tokens)
                #result：all_documents = [[[t1,t2..]],[[t1,...]],[[t1,t2,t3...]],[[t1,...]],...,[[t1,t2...]]]
    # Remove empty documents
    all_documents = [x for x in all_documents if x]#去掉空的子元素（list）

    # rng.shuffle(all_documents)

    all_wrong_documents = [[]]#记录所有的wrong lines 采用列表中的列表 形式进行记录

    # Input file format:
    # (1) One sentence per line. These should ideally be actual sentences, not
    # entire paragraphs or arbitrary spans of text. (Because we use the
    # sentence boundaries for the "next sentence prediction" task).
    # (2) Blank lines between documents. Document boundaries are needed so
    # that the "next sentence prediction" task doesn't span between documents.
    for wrong_input_file in wrong_input_files:
        with tf.gfile.GFile(wrong_input_file, "r") as reader:
            while True:
                line = tokenization.convert_to_unicode(reader.readline())
                if not line:
                    break
                line = line.strip()

                # Empty lines are used as document delimiters
                if not line:
                    all_wrong_documents.append([])
                tokens = tokenizer.tokenize(line)
                if tokens:
                    all_wrong_documents[-1].append(tokens)

    # Remove empty documents
    all_wrong_documents = [x for x in all_wrong_documents if x]


    #对根据correct 和 wrong 构建的数据进行断定 二者的长度应该相等
    assert len(all_documents) == len(all_wrong_documents)

    # rng.shuffle(all_documents)

    vocab_words = list(tokenizer.vocab.keys())
    instances = []
    for _ in range(dupe_factor):
        for document_index in range(len(all_documents)):
            instances.extend(
                create_instances_from_document(
                    all_documents, document_index, max_seq_length, short_seq_prob,
                    masked_lm_prob, max_predictions_per_seq, vocab_words, rng, all_wrong_documents))

    rng.shuffle(instances)#数据打乱
    return instances



#构建训练实例
def create_instances_from_document(
        all_documents, document_index, max_seq_length, short_seq_prob,
        masked_lm_prob, max_predictions_per_seq, vocab_words, rng, all_wrong_documents, original=True):
    """Creates `TrainingInstance`s for a single document."""
    #根据指定下标document_index获取指定的correct和wrong document
    document = all_documents[document_index]
    wrong_document = all_wrong_documents[document_index]

    # Account for [CLS], [SEP], [SEP]
    #因为要在tokenize后 插入三个特殊符号[CLS], [SEP], [SEP]所以能允许的最大tokens的长度应该在max_seq_length基础上-3
    max_num_tokens = max_seq_length - 3

    # We *usually* want to fill up the entire sequence since we are padding
    # to `max_seq_length` anyways, so short sequences are generally wasted
    # computation. However, we *sometimes*
    # (i.e., short_seq_prob == 0.1 == 10% of the time) want to use shorter
    # sequences to minimize the mismatch between pre-training and fine-tuning.
    # The `target_seq_length` is just a rough target however, whereas
    # `max_seq_length` is a hard limit.
    target_seq_length = max_num_tokens
    if rng.random() < short_seq_prob:#生成一个01间的float 如果小于short_seq_prob（比最大sequence长度小的概率） 就重新设定target_seq_length的长度
        target_seq_length = rng.randint(2, max_num_tokens)#生成2到max_num_tokens之间的一个int数

    # We DON'T just concatenate all of the tokens from a document into a long
    # sequence and choose an arbitrary split point because this would make the
    # next sentence prediction task too easy. Instead, we split the input into
    # segments "A" and "B" based on the actual "sentences" provided by the user
    # input.
    instances = []
    current_chunk = []
    current_wrong_chunck = []
    current_length = 0
    i = 0
    while i < len(document):
        segment = document[i]
        wrong_segment = wrong_document[i]
        current_chunk.append(segment)
        current_wrong_chunck.append(wrong_segment)

        # assert len(segment) == len(wrong_segment)
        current_length += len(segment)
        try:
            assert len(segment) == len(wrong_segment)
        except:
            print(segment)
            print(wrong_segment)
            exit()
        # assert segment == wrong_segment

        if i == len(document) - 1 or current_length >= target_seq_length:
            if current_chunk:
                # `a_end` is how many segments from `current_chunk` go into the `A`
                # (first) sentence.
                a_end = 1
                if len(current_chunk) >= 2:
                    a_end = rng.randint(1, len(current_chunk) - 1)

                tokens_a = []
                wrong_tokens_a = []
                for j in range(a_end):
                    tokens_a.extend(current_chunk[j])
                    wrong_tokens_a.extend(current_wrong_chunck[j])

                assert len(tokens_a) == len(wrong_tokens_a)

                tokens_b = []
                wrong_tokens_b = []
                # Random next 随机的下句
                is_random_next = False
                if len(current_chunk) == 1 or rng.random() < 0.5:
                    is_random_next = True
                    target_b_length = target_seq_length - len(tokens_a)

                    # This should rarely go for more than one iteration for large
                    # corpora. However, just to be careful, we try to make sure that
                    # the random document is not the same as the document
                    # we're processing.
                    for _ in range(10):
                        random_document_index = rng.randint(0, len(all_documents) - 1)
                        if random_document_index != document_index:
                            break

                    random_document = all_documents[random_document_index]
                    random_wrong_document = all_wrong_documents[random_document_index]
                    random_start = rng.randint(0, len(random_document) - 1)
                    for j in range(random_start, len(random_document)):
                        tokens_b.extend(random_document[j])
                        wrong_tokens_b.extend(random_wrong_document[j])
                        if len(tokens_b) >= target_b_length:
                            break
                    # We didn't actually use these segments so we "put them back" so
                    # they don't go to waste.
                    num_unused_segments = len(current_chunk) - a_end
                    i -= num_unused_segments
                # Actual next 实际的下句
                else:
                    is_random_next = False
                    for j in range(a_end, len(current_chunk)):
                        tokens_b.extend(current_chunk[j])
                        wrong_tokens_b.extend(current_wrong_chunck[j])
                truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng, wrong_tokens_a, wrong_tokens_b)
                # truncate_seq_pair(wrong_tokens_a, wrong_tokens_b, max_num_tokens, rng)

                assert len(tokens_a) >= 1
                assert len(tokens_b) >= 1

                tokens = []
                wrong_tokens = []
                segment_ids = []
                tokens.append("[CLS]")
                wrong_tokens.append("[CLS]")
                segment_ids.append(0)
                for token in tokens_a:
                    tokens.append(token)
                    segment_ids.append(0)

                for w_token in wrong_tokens_a:
                    wrong_tokens.append(w_token)
                    # segment_ids.append(0)

                tokens.append("[SEP]")
                wrong_tokens.append("[SEP]")
                segment_ids.append(0)

                for token in tokens_b:
                    tokens.append(token)
                    segment_ids.append(1)

                for w_token in wrong_tokens_b:
                    wrong_tokens.append(w_token)

                tokens.append("[SEP]")
                wrong_tokens.append("[SEP]")
                segment_ids.append(1)

                if tokens == wrong_tokens:#对于正确句子的MASK策略
                    (tokens, masked_lm_positions,
                     masked_lm_labels) = create_masked_lm_predictions(
                        tokens, masked_lm_prob, max_predictions_per_seq, vocab_words, rng)
                else:#对于错误句子的MASK策略
                    (tokens, masked_lm_positions,
                     masked_lm_labels) = create_masked_lm_predictions_for_wrong_sentences(
                        tokens, masked_lm_prob, max_predictions_per_seq, vocab_words, rng, wrong_tokens)
                    # print(tokens)
                    # print(wrong_tokens)
                    # print(masked_lm_positions)
                    # print(masked_lm_labels)
                    # print('\n')
                instance = TrainingInstance(
                    tokens=tokens,
                    segment_ids=segment_ids,
                    is_random_next=is_random_next,
                    masked_lm_positions=masked_lm_positions,
                    masked_lm_labels=masked_lm_labels)
                instances.append(instance)

            current_chunk = []
            current_wrong_chunck = []
            current_length = 0
        i += 1
    '''
    在每一次for循环中 创建一个instances实例 上述代码很难弄懂得部分是关于下一句预测 is_random_next = 1(不构成上下文关系)0(构成上下文关系)
    tokens = [CLS]s1[SEP]s2[SEP]
    segment_ids = [0]+[0]*len(s1)+[1]*len(s2)+[1]
    is_random_next = 0/1
    masked_lm_positions = 29, 582, 120,...
    masked_lm_labels = 213,415,312,...
    assert len(tokens) = len(segment_ids) = len(masked_lm_positions) = len(masked_lm_labels)
    '''
    return instances

#对于正确句子的MASK策略
def create_masked_lm_predictions(tokens, masked_lm_prob,
                                 max_predictions_per_seq, vocab_words, rng):
    """Creates the predictions for the masked LM objective."""
    #根据传入的tokens和相关信息进行mask
    cand_indexes = []
    for (i, token) in enumerate(tokens):
        if token == "[CLS]" or token == "[SEP]":
            continue
        cand_indexes.append(i)

    rng.shuffle(cand_indexes)#打乱index 具有鲁棒性

    output_tokens = list(tokens)
    #max_predictions_per_seq 每一个seq最多mask的数目 masked_lm_prob 每一个seq掩盖的mask比例（0.15）
    #确定目标掩盖数目
    num_to_predict = min(max_predictions_per_seq,
                         max(1, int(round(len(tokens) * masked_lm_prob))))


    masked_lms = [] #存储被MASK位置和相应的label
    covered_indexes = set()#存储对一个seq实际掩盖的index
    for index in cand_indexes:
        if len(masked_lms) >= num_to_predict:#超出了准许掩盖或者目标掩盖数目
            break
        if index in covered_indexes:#避免重复掩盖位置
            continue
        covered_indexes.add(index)

        masked_token = None

        #MASK策略 80%替换为[MASK] 10%保持自身不变 10%替换为一个随机的token
        # 80% of the time, replace with [MASK]
        if rng.random() < 0.8:
            masked_token = "[MASK]"
        else:
            # 10% of the time, keep original
            if rng.random() < 0.5:
                masked_token = tokens[index]
            # 10% of the time, replace with random word
            else:
                masked_token = vocab_words[rng.randint(0, len(vocab_words) - 1)]

        # overwrite the above assignment with mask_prob
        if tokens[index] in MASK_PROB:#确定待MASK的token在MASK_PROB中且以1-MASK_PROB[tokens[index]]的概率进行替换
            if rng.random() < MASK_PROB[tokens[index]]:
                masked_token = tokens[index]
                # print(f'cover {tokens[index]} in correct instance.')
                #未被替换时统计+1
                CORRECT_COUNT[tokens[index]] += 1

        #进行替换操作
        output_tokens[index] = masked_token

        masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))

    masked_lms = sorted(masked_lms, key=lambda x: x.index)
    #统计一个sequenc中被imask掉的位置和对应的label
    masked_lm_positions = []
    masked_lm_labels = []
    for p in masked_lms:
        masked_lm_positions.append(p.index)
        masked_lm_labels.append(p.label)

    return output_tokens, masked_lm_positions, masked_lm_labels

#对于错误句子的MASK策略
def create_masked_lm_predictions_for_wrong_sentences(tokens, masked_lm_prob,
                                                     max_predictions_per_seq, vocab_words, rng, wrong_tokens):
    """Creates the predictions for the masked LM objective."""

    cand_indexes = []
    #对输入的token和wrong_token的长度进行判断
    if not len(tokens) == len(wrong_tokens):
        print(tokens)
        print(wrong_tokens)
    assert len(tokens) == len(wrong_tokens)
    for (i, token) in enumerate(tokens):
        if token == "[CLS]" or token == "[SEP]":
            assert wrong_tokens[i] == token
            continue
        elif token != wrong_tokens[i]:#将错误位置添加到待替换的选项中
            cand_indexes.append(i)
        else:  # when a token is not confused, add it to candidates according to its mask probability
            if token in MASK_PROB:
                if rng.random() < MASK_PROB[token]:
                    #替换统计+1
                    WRONG_COUNT[token] += 1
                    # print(f'cover {token} in wrong instance.')
                    cand_indexes.append(i)#将位置添加到待替换的选项中

    rng.shuffle(cand_indexes)#index打乱

    output_tokens = list(tokens)

    # num_to_predict = min(max_predictions_per_seq,
    #                      max(1, int(round(len(tokens) * masked_lm_prob ))))
    num_to_predict = min(max_predictions_per_seq,
                         max(1, int(round(len(tokens)))))  # we set 100% masking rate to allow all errors and corresponding non-errors to be masked

    masked_lms = []
    covered_indexes = set()
    for index in cand_indexes:
        if len(masked_lms) >= num_to_predict:
            break
        if index in covered_indexes:
            continue
        covered_indexes.add(index)

        #用自身进行MASK
        masked_token = wrong_tokens[index]
        output_tokens[index] = masked_token

        masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))

    masked_lms = sorted(masked_lms, key=lambda x: x.index)

    masked_lm_positions = []
    masked_lm_labels = []
    for p in masked_lms:
        masked_lm_positions.append(p.index)
        masked_lm_labels.append(p.label)

    return output_tokens, masked_lm_positions, masked_lm_labels

#对字段进行截断 以满足max_num_tokens
def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng, wrong_tokens_a, wrong_tokens_b):
    """Truncates a pair of sequences to a maximum sequence length."""
    assert len(tokens_a) == len(wrong_tokens_a)
    try:
        assert len(tokens_b) == len(wrong_tokens_b)
    except:
        print(tokens_b)
        print(wrong_tokens_b)
        exit()

    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_num_tokens:
            break

        trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
        wrong_trunc_tokens = wrong_tokens_a if len(wrong_tokens_a) > len(wrong_tokens_b) else wrong_tokens_b
        assert len(trunc_tokens) >= 1

        # We want to sometimes truncate from the front and sometimes from the
        # back to add more randomness and avoid biases.
        if rng.random() < 0.5:
            del trunc_tokens[0]
            del wrong_trunc_tokens[0]
        else:
            trunc_tokens.pop()
            wrong_trunc_tokens.pop()


def main(_):
    #设置log的可见度
    tf.logging.set_verbosity(tf.logging.INFO)
    #实例化分词器tokenizer 采用bert预训练的词表
    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file)
    #输入文件（correct）
    input_files = []
    for input_pattern in FLAGS.input_file.split(","):
        input_files.extend(tf.gfile.Glob(input_pattern))
    #输入文件（wrong）
    wrong_input_files = []
    for wrong_input_pattern in FLAGS.wrong_input_file.split(","):
        wrong_input_files.extend(tf.gfile.Glob(wrong_input_pattern))

    print(input_files)

    tf.logging.info("*** Reading from input files ***")
    '''
    #我认为correct和wrong都属于输入文件 所以修改了这部分代码
    old version:
        for input_file in input_files:
            tf.logging.info("  %s", input_file)
    '''
    for input_file, wrong_input_file in zip(input_files, wrong_input_files):
        tf.logging.info("  %s", input_file)
        tf.logging.info("  %s", wrong_input_file)#我认为correct和wrong都属于输入文件 所以修改了这部分代码
    #固定随机生成的种子
    rng = random.Random(FLAGS.random_seed)
    #根据输出入输出创建训练实例
    instances = create_training_instances(
        input_files, tokenizer, FLAGS.max_seq_length, FLAGS.dupe_factor,
        FLAGS.short_seq_prob, FLAGS.masked_lm_prob, FLAGS.max_predictions_per_seq,
        rng, wrong_input_files)

    output_files = FLAGS.output_file.split(",")
    tf.logging.info("*** Writing to output files ***")
    for output_file in output_files:
        tf.logging.info("  %s", output_file)

    write_instance_to_example_files(instances, tokenizer, FLAGS.max_seq_length,
                                    FLAGS.max_predictions_per_seq, output_files)


if __name__ == "__main__":
    flags.mark_flag_as_required("input_file")
    flags.mark_flag_as_required("wrong_input_file")
    flags.mark_flag_as_required("output_file")
    flags.mark_flag_as_required("vocab_file")
    tf.app.run()



