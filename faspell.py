from char_sim import CharFuncs
from masked_lm import MaskedLM
from bert_modified import modeling
import re
import json
import pickle
import argparse
import numpy
import logging
import plot
import tqdm
import time





####################################################################################################

__author__ = 'Yuzhong Hong <hongyuzhong@qiyi.com / eugene.h.git@gmail.com>'
__date__ = '10/09/2019'
__description__ = 'The main script for FASPell - Fast, Adaptable, Simple, Powerful Chinese Spell Checker'

#加载faspell的配置文件
CONFIGS = json.loads(open('faspell_configs.json', 'r', encoding='utf-8').read())
#从配置信息中获取 形似 和音似的权重
WEIGHTS = (CONFIGS["general_configs"]["weights"]["visual"], CONFIGS["general_configs"]["weights"]["phonological"], 0.0)

#从配置信息中获取通用配置信息
CHAR = CharFuncs(CONFIGS["general_configs"]["char_meta"])


#从配置信息中获取MLM的相关配置信息
class LM_Config(object):
    max_seq_length = CONFIGS["general_configs"]["lm"]["max_seq"]
    vocab_file = CONFIGS["general_configs"]["lm"]["vocab"]
    bert_config_file = CONFIGS["general_configs"]["lm"]["bert_configs"]
    #选择微调后的模型还是选择原生BERT
    if CONFIGS["general_configs"]["lm"]["fine_tuning_is_on"]:
        init_checkpoint = CONFIGS["general_configs"]["lm"]["fine-tuned"]
    else:
        init_checkpoint = CONFIGS["general_configs"]["lm"]["pre-trained"]
    bert_config = modeling.BertConfig.from_json_file(bert_config_file)
    topn = CONFIGS["general_configs"]["lm"]["top_n"]


#构造过滤器
class Filter(object):
    def __init__(self):
        self.curve_idx_sound = {0: {0: Curves.curve_null,#Curves.curve_full,  # 0 for non-difference
                              1: Curves.curve_null,#Curves.curve_d0r1p,
                              2: Curves.curve_null,#Curves.curve_d0r2p,
                              3: Curves.curve_null,
                              4: Curves.curve_null,
                              5: Curves.curve_null,
                              6: Curves.curve_null,
                              7: Curves.curve_null,
                              },
                          1: {0: Curves.curve_null,#Curves.curve_d1r0p,  # 1 for difference
                              1: Curves.curve_null,#Curves.curve_d1r1p,
                              2: Curves.curve_null,#Curves.curve_d1r2p,
                              3: Curves.curve_null,
                              4: Curves.curve_null,
                              5: Curves.curve_null,
                              6: Curves.curve_null,
                              7: Curves.curve_null,
                              }}

        self.curve_idx_shape = {0: {0: Curves.curve_null,#Curves.curve_full,  # 0 for non-difference
                                    1: Curves.curve_null,#Curves.curve_d0r1s,
                                    2: Curves.curve_null,#Curves.curve_d0r2s,
                                    3: Curves.curve_null,
                                    4: Curves.curve_null,
                                    5: Curves.curve_null,
                                    6: Curves.curve_null,
                                    7: Curves.curve_null,
                                    },
                                1: {0: Curves.curve_null,#Curves.curve_d1r0s,  # 1 for difference y1 = (7.64960918 * x1 -7) / - 2.87156076
                                    1: Curves.curve_null,#Curves.curve_d1r1s,
                                    2: Curves.curve_null,#Curves.curve_d1r2s,
                                    3: Curves.curve_null,
                                    4: Curves.curve_null,
                                    5: Curves.curve_null,
                                    6: Curves.curve_null,
                                    7: Curves.curve_null,
                                    }}

    def filter(self, rank, difference, error, filter_is_on=True, sim_type='shape'):
        if filter_is_on:#开启过滤条件下 根据候选字进行过滤类别选择 是否属于top-difference -> 属于哪一个rank -> 属于sound还是shape
            if sim_type == 'sound':
                curve = self.curve_idx_sound[int(difference)][rank]
            else:
                # print(int(difference))
                curve = self.curve_idx_shape[int(difference)][rank]
        else:
            curve = Curves.curve_null

        if curve(error["confidence"], error["similarity"]) and self.special_filters(error):
            return True

        return False

    #加入规则进行过滤
    @staticmethod
    def special_filters(error):
        """
        Special filters for, essentially, grammatical errors. The following is some examples.
        """
        # if error["original"] in {'他': 0, '她': 0, '你': 0, '妳': 0}:
        #     if error["confidence"] < 0.95:
        #         return False
        #
        # if error["original"] in {'的': 0, '得': 0, '地': 0}:
        #     if error["confidence"] < 0.6:
        #         return False
        #
        # if error["original"] in {'在': 0, '再': 0}:
        #     if error["confidence"] < 0.6:
        #         return False

        return True


#过滤器中应用的过滤曲线
class Curves(object):
    def __init__(self):
        pass

    #不进行过滤
    @staticmethod
    def curve_null(confidence, similarity):
        """This curve is used when no filter is applied"""
        return True
    #过滤所有
    @staticmethod
    def curve_full(confidence, similarity):
        """This curve is used to filter out everything"""
        return False

    #这就是论文中所选择的人工确定的曲线过滤方式，可以采用多根直线进行联合过滤 模拟曲线效果
    @staticmethod
    def curve_01(confidence, similarity):
        """
        we provide an example of how to write a curve. Empirically, curves are all convex upwards.
        Thus we can approximate the filtering effect of a curve using its tangent lines.
        """
        #使用两根直线进行过滤
        flag1 = 20 / 3 * confidence + similarity - 21.2 / 3 > 0
        flag2 = 0.1 * confidence + similarity - 0.6 > 0
        if flag1 or flag2:
            return True

        return False
    #========================================================================#
    def curve_d1r0s(confidence, similarity):
        flag1 = 1 * confidence + similarity * 1 -1> 0
        flag2 = similarity > 0.4
        if flag1 or flag2:
            return True
        return False

    def curve_d1r1s(confidence, similarity):
        flag1 = 0.6 * confidence + similarity*0.5 -0.3 > 0
        if flag1:
            return True

        return False

    def curve_d1r2s(confidence, similarity):
        flag1 = 0.6 * confidence + similarity*0.4 -0.24 > 0
        if flag1:
            return True

        return False

    def curve_d1r0p(confidence, similarity):
        flag1 = 0.6 * confidence + similarity*0.8 -0.48 > 0
        flag2 = similarity > 0.3
        if flag1 and flag2:
            return True

        return False

    def curve_d1r1p(confidence, similarity):
        flag1 = 0.6 * confidence + similarity*0.6 -0.36 > 0
        flag2 = similarity > 0.3
        if flag1 and flag2:
            return True

        return False

    def curve_d1r2p(confidence, similarity):
        flag1 = 0.8 * confidence + similarity*0.1 -0.08 > 0
        flag2 = similarity > 0.6
        if flag1 and flag2:
            return True

        return False

        return False

    def curve_d0r1s(confidence, similarity):
        flag1 = 0.8 * confidence + similarity*0.4 -0.32 > 0
        flag2 = similarity > 0.4
        flag3 = similarity < 0.8
        if flag1 and flag2 and flag3:
            return True

        return False

    def curve_d0r1p(confidence, similarity):
        flag1 = 0.9* confidence + similarity*0.7 -0.63 > 0
        flag2 = similarity > 0.7

        if flag1 and flag2:
            return True

        return False

    def curve_d0r2p(confidence, similarity):
        flag1 = 1* confidence + similarity*0.4 -0.4 > 0
        flag2 = similarity > 0.7

        if flag1 and flag2:
            return True

        return False

    def curve_d0r2s(confidence, similarity):
        flag1 = 1 * confidence + similarity*0.4 -0.4 > 0
        flag2 = similarity > 0.4

        if flag1 and flag2:
            return True

        return False
    # ========================================================================#
#构造CSCChecker 由两部分组成 MLM 和 过滤器
class SpellChecker(object):
    def __init__(self):
        self.masked_lm = MaskedLM(LM_Config()) #MLM
        self.filter = Filter() #过滤器

    #添加部分规则 确定哪些纠正被视为无效
    @staticmethod
    def pass_ad_hoc_filter(corrected_to, original): #original原始字符（输入） corrected_to纠正后的字符
        if corrected_to == '[UNK]':
            return False
        if '#' in corrected_to:
            return False
        if len(corrected_to) != len(original):
            return False
        if re.findall(r'[a-zA-ZＡ-Ｚａ-ｚ]+', corrected_to):
            return False
        if re.findall(r'[a-zA-ZＡ-Ｚａ-ｚ]+', original):
            return False
        return True

    #错误字符的检查
    def get_error(self, sentence, j, cand_tokens, rank=0, difference=True, filter_is_on=True, weights=WEIGHTS, sim_type='shape'):
        """
        PARAMS
        ------------------------------------------------
        sentence: sentence to be checked
        j: position of the character to be checked
        cand_tokens: all candidates
        rank: the rank of the candidate in question
        filters_on: only used in ablation experiment to remove CSD
        weights: weights for different types of similarity
        sim_type: type of similarity

        """

        cand_token, cand_token_prob = cand_tokens[rank]

        if cand_token != sentence[j]:#候选和原始不一样 判断是否为有效的错误检测
            #对于char的error具体结构
            error = {"error_position": j,
                     "original": sentence[j],
                     "corrected_to": cand_token,
                     "candidates": dict(cand_tokens),
                     "confidence": cand_token_prob,
                     "similarity": CHAR.similarity(sentence[j], cand_token, weights=weights),#weights表示在计算相似度时 不同部分所占比例 （shape, sound, freq）
                     "sentence_len": len(sentence)}

            if not self.pass_ad_hoc_filter(error["corrected_to"], error["original"]):
                logging.info(f'{error["original"]}'
                             f' --> <PASS-{error["corrected_to"]}>'
                             f' (con={error["confidence"]}, sim={error["similarity"]}, on_top_difference={difference})')
                return None #无效的错误

            else:
                if self.filter.filter(rank, difference, error, filter_is_on, sim_type=sim_type):
                    logging.info(f'{error["original"]}'
                                 f'--> {error["corrected_to"]}'
                                 f' (con={error["confidence"]}, sim={error["similarity"]}, on_top_difference={difference})')
                    return error #有效的错误

                logging.info(f'{error["original"]}'
                             f' --> <PASS-{error["corrected_to"]}>'
                             f' (con={error["confidence"]}, sim={error["similarity"]}, on_top_difference={difference})')
                return None #无效的错误

        logging.info(f'{sentence[j]}'
                     f' --> <PASS-{sentence[j]}>'
                     f' (con={cand_token_prob}, sim=null, on_top_difference={difference})')
        return None #没有错误 候选的和原始的字符一样

    #错误的纠正
    def make_corrections(self,
                         sentences,
                         tackle_n_gram_bias=CONFIGS["exp_configs"]["tackle_n_gram_bias"],
                         rank_in_question=CONFIGS["general_configs"]["rank"],
                         dump_candidates=CONFIGS["exp_configs"]["dump_candidates"],
                         read_from_dump=CONFIGS["exp_configs"]["read_from_dump"],
                         is_train=False,
                         train_on_difference=True,
                         filter_is_on=CONFIGS["exp_configs"]["filter_is_on"],
                         sim_union=CONFIGS["exp_configs"]["union_of_sims"]
                         ):
        """
        PARAMS:
        ------------------------------
        sentences: sentences with potential errors
        tackle_n_gram_bias: whether the hack to tackle n gram bias is used
        rank_in_question: rank of the group of candidates in question
        dump_candidates: whether save candidates to a specific path
        read_from_dump: read candidates from dump
        is_train: if the script is in the training mode
        train_on_difference: choose the between two sub groups
        filter_is_on: used in ablation experiments to decide whether to remove CSD
        sim_union: whether to take the union of the filtering results given by using two types of similarities

        RETURN:
        ------------------------------
        correction results of all sentences
        """
        #对待纠错句子做预处理 前后添加。
        processed_sentences = self.process_sentences(sentences)
        generation_time = 0
        if read_from_dump:#从dump加载candidate
            assert dump_candidates
            topn_candidates = pickle.load(open(dump_candidates, 'rb'))
        else:#使用MLM进行candidate的生成
            start_generation = time.time()
            topn_candidates = self.masked_lm.find_topn_candidates(processed_sentences,
                                                                  batch_size=CONFIGS["general_configs"]["lm"][
                                                                      "batch_size"])
            end_generation = time.time()
            generation_time += end_generation - start_generation #生成condidate所用的时间
            if dump_candidates:#写入dump文件 防止在训练CSD的时候重复调用模型进行生成
                pickle.dump(topn_candidates, open(dump_candidates, 'wb'))

        # main workflow  CSC工作的主要流程
        filter_time = 0 #filter所用时间
        skipped_count = 0 #skip次数
        results = []    #记录对sentences的CSC结果
        print('making corrections ...')
        if logging.getLogger().getEffectiveLevel() != logging.INFO:  # show progress bar if not in verbose mode
            progess_bar = tqdm.tqdm(enumerate(topn_candidates))
        else:
            progess_bar = enumerate(topn_candidates)

        for i, cand in progess_bar:#循环每一个句子的topn_candidate
            logging.info("*" * 50)
            logging.info(f"spell checking sentence {sentences[i]}")
            sentence = '' #存储CSC后的句子
            res = [] #存储错误信息 每一个错误信息的结构是一个char的error

            # can't cope with sentences containing Latin letters yet.
            # 过滤那些含有拉丁文字母的句子
            if re.findall(r'[a-zA-ZＡ-Ｚａ-ｚ]+', sentences[i]):#对待check 句子 进行逐一check
                skipped_count += 1
                results.append({"original_sentence": sentences[i],
                                "corrected_sentence": sentences[i],
                                "num_errors": 0,
                                "errors": []
                                })
                logging.info("containing Latin letters; pass current sentence.")

            else:

                # when testing on SIGHAN13,14,15, we recommend using `extension()` to solve
                # issues caused by full-width humbers;
                # when testing on OCR data, we recommend using `extended_cand = cand`
                #extended_cand = extension(cand)
                extended_cand = cand
                for j, cand_tokens in enumerate(extended_cand):  # spell check for each characters 对于每一个待检查的句子逐一检查每个char
                    if 0 < j < len(extended_cand) - 1:  # skip the head and the end placeholders -- `。` 因为在句子的首尾加了。
                        # print(j)

                        char = sentences[i][j - 1]

                        # detect and correct errors
                        error = None

                        # spell check rank by rank 按照rank的排行逐一检查  rank_in_question实在faspellconfig.json中指定的
                        start_filter = time.time()
                        for rank in range(rank_in_question + 1):
                            logging.info(f"spell checking on rank={rank}")

                            if not sim_union:#对类型的不同选择进行分开检查 计算sim的时候weight为[1,0,0] 或者[0,1,0]
                                if WEIGHTS[0] > WEIGHTS[1]:#形状相似
                                    sim_type = 'shape'
                                else:#发音相似
                                    sim_type = 'sound'
                                error = self.get_error(sentences[i],#对第i个句子进行检查
                                                       j - 1,#检察第i句子中的第j-1个字符
                                                       cand_tokens,#第i个句子第j-1个char的candidate
                                                       rank=rank,#逐一选择rank
                                                       difference=cand_tokens[0][0] != sentences[i][j - 1],#判断排名第一的候选字符与原始字符是否不同
                                                       filter_is_on=filter_is_on, sim_type=sim_type)

                            else:#对两种类型均进行检查

                                logging.info("using shape similarity:")
                                error_shape = self.get_error(sentences[i],
                                                             j - 1,
                                                             cand_tokens,
                                                             rank=rank,
                                                             difference=cand_tokens[0][0] != sentences[i][j - 1],
                                                             filter_is_on=filter_is_on,
                                                             weights=(1, 0, 0), sim_type='shape')
                                logging.info("using sound similarity:")
                                error_sound = self.get_error(sentences[i],
                                                             j - 1,
                                                             cand_tokens,
                                                             rank=rank,
                                                             difference=cand_tokens[0][0] != sentences[i][j - 1],
                                                             filter_is_on=filter_is_on,
                                                             weights=(0, 1, 0), sim_type='sound')
  

                                if error_shape:
                                    error = error_shape
                                    if is_train:
                                        error = None  # to train shape similarity, we do not want any error that has already detected by sound similarity
                                                        #我认为这里的逻辑写的有点问题 所以避免这个问题影响结果 在训练CSD的时候 保持union_of_sims为false
                                else:
                                    error = error_sound

                            #当在该轮rank中找到了错误的时候
                            if error:
                                if is_train:#训练CSD模式下
                                    if rank != rank_in_question:  # not include candidate that has a predecessor already
                                        # taken as error
                                        error = None
                                        # break
                                    else:
                                        # do not include candidates produced by different candidate generation strategy#在两个子group中选择 但是不明白为什么要用！=逻辑 不直接用==
                                        if train_on_difference != (cand_tokens[0][0] != sentences[i][j - 1]):
                                            error = None
                                else:#非训练模式下
                                    break


                        end_filter = time.time()
                        # 检查每个char在不同rank中所用的时间之和 为所有CSC句子所用时间
                        filter_time += end_filter - start_filter

                        #对于找到了错误的char将其替换为纠正后的token
                        if error:
                            res.append(error)
                            char = error["corrected_to"]
                            sentence += char
                            continue
                        #对于不认为是错误的char 则仍然是原始的字符
                        sentence += char

                # a small hack: tackle the n-gram bias problem: when n adjacent characters are erroneous,
                # pick only the one with the greatest confidence.
                #对于完整csc后的一个句子继续做以下处理  这其实是一个小的技巧 对于连续错的char 选择其中具有最高confidence的
                #错误char作为是错误的判定和纠正 之所以要这样做的原因还有待探究
                error_delete_positions = []
                if tackle_n_gram_bias:
                    error_delete_positions = []
                    for idx, error in enumerate(res):
                        delta = 1
                        n_gram_errors = [error]
                        try:
                            while res[idx + delta]["error_position"] == error["error_position"] + delta:
                                n_gram_errors.append(res[idx + delta])
                                delta += 1
                        except IndexError:
                            pass
                        n_gram_errors.sort(key=lambda e: e["confidence"], reverse=True)
                        error_delete_positions.extend([(e["error_position"], e["original"]) for e in n_gram_errors[1:]])

                    error_delete_positions = dict(error_delete_positions)

                    res = [e for e in res if e["error_position"] not in error_delete_positions]

                    def process(pos, c):
                        if pos not in error_delete_positions:
                            return c
                        else:
                            return error_delete_positions[pos]

                    sentence = ''.join([process(pos, c) for pos, c in enumerate(sentence)])

                # add the result for current sentence
                results.append({"original_sentence": sentences[i],
                                "corrected_sentence": sentence,
                                "num_errors": len(res),
                                "errors": res
                                })
                logging.info(f"current sentence is corrected to {sentence}")
                logging.info(f" {len(error_delete_positions)} errors are deleted to prevent n-gram bias problem")
                logging.info("*" * 50 + '\n')
        try:#CSC在generate上的时间和filter的时间
            print(
                f"Elapsed time: {generation_time // 60} min {generation_time % 60} s in generating candidates for {len(sentences)} sentences;\n"
                f"              {filter_time} s in filtering candidates for {len(sentences) - skipped_count} sentences;\n"
                f"Speed: {generation_time / len(sentences) * 1000} ms/sentence in generating and {filter_time / (len(sentences) - skipped_count) * 1000} ms/sentence in filtering ")
        except ZeroDivisionError:
            print(
                f"Elapsed time: {generation_time // 60} min {generation_time % 60} s in generating candidates for {len(sentences)} sentences;\n"
                f"              {filter_time} s in filtering candidates for {len(sentences) - skipped_count} sentences;\n"
                f"Speed: {generation_time / len(sentences) * 1000} ms/sentence in generating and NaN ms/sentence in filtering ")
        return results

    #重复进行错误纠正 也就是对待CSC的句子进行重复CSC
    def repeat_make_corrections(self, sentences, num=3, is_train=False, train_on_difference=True):
        all_results = [] #存储历史CSC记录 [[round1],[round2],...,[roundn]]
        sentences_to_be_corrected = sentences

        for _ in range(num):
            results = self.make_corrections(sentences_to_be_corrected,
                                            is_train=is_train,
                                            train_on_difference=train_on_difference)
            sentences_to_be_corrected = [res["corrected_sentence"] for res in results]
            all_results.append(results)
        #重复CSC的历史记录
        correction_history = []
        for i, sentence in enumerate(sentences):
            r = {"original_sentence": sentence, "correction_history": []}
            for item in all_results:#在每一个round中找第i个句子的csc结果
                r["correction_history"].append(item[i]["corrected_sentence"])
            correction_history.append(r)

        return all_results, correction_history


    #对于输入句子前后加上。

    #因为MLM模型都是在许多连续句子上进行训练的，所以句子开始后结束出极有可能被
    #预测为‘。’（做实验可以观察，确实是这样）。为了避免这个问题没在每个句子前后加上‘。’
    @staticmethod
    def process_sentences(sentences):
        """Because masked language model is trained on concatenated sentences,
         the start and the end of a sentence in question is very likely to be
         corrected to the period symbol (。) of Chinese. Hence, we add two period
        symbols as placeholders to prevent this from harming FASPell's performance."""

        return ['。' + sent + '。' for sent in sentences]


def extension(candidates):#两个邻近的full-width 数字或字母被当做一个token输入了mlm
    """this function is to resolve the bug that when two adjacent full-width numbers/letters are fed to mlm,
       the output will be merged as one output, thus lead to wrong alignments."""
    new_candidates = []
    for j, cand_tokens in enumerate(candidates): #candidates：[[('token1',Pro),('token1',Pro),('token1',Pro)...('token1',Pro)],[('token2',Pro),('token2',Pro),('token2',Pro)...('token2',Pro)]...]
        real_cand_tokens = cand_tokens[0][0]#每一个token的第一个候选
        if '##' in real_cand_tokens:  # sometimes the result contains '##', so we need to get rid of them first
            real_cand_tokens = real_cand_tokens[2:]
        # 正常情况 每一个产生的候选长度应该为1  用于将长度为2的一个candidate拆分成 两个长度为1的candidate
        if len(real_cand_tokens) == 2 and not re.findall(r'[a-zA-ZＡ-Ｚａ-ｚ]+', real_cand_tokens):
            a = []
            b = []
            for cand, score in cand_tokens:#依次处理每一个token的候选字
                real_cand = cand
                if '##' in real_cand:
                    real_cand = real_cand[2:]
                a.append((real_cand[0], score))
                b.append((real_cand[-1], score))
            new_candidates.append(a)
            new_candidates.append(b)
            continue

        new_candidates.append(cand_tokens)

    return new_candidates

#在测试文件上进行重复纠错测试
def repeat_test(test_path, spell_checker, repeat_num, is_train, train_on_difference=True):
    sentences = []
    #根据指定路径获取相关信息
    for line in open(test_path, 'r', encoding='utf-8'):
        num, wrong, correct = line.strip().split('\t')
        sentences.append(wrong)
    #获取所有的结果
    all_results, correction_history = spell_checker.repeat_make_corrections(sentences, num=repeat_num,
                                                                            is_train=is_train,
                                                                            train_on_difference=train_on_difference)
    if is_train:
        for i, res in enumerate(all_results):
            print(f'performance of round {i}:')
            test_unit(res, test_path,
                      f'difference_{int(train_on_difference)}-rank_{CONFIGS["general_configs"]["rank"]}-results_{i}')
    else:
        for i, res in enumerate(all_results):
            print(f'performance of round {i}:')
            test_unit(res, test_path, f'test-results_{i}')

    #将纠正历史写入history.json文件中
    w = open(f'history.json', 'w', encoding='utf-8')
    w.write(json.dumps(correction_history, ensure_ascii=False, indent=4, sort_keys=False))
    w.close()

#纠错测试
def repeat_non_test(sentences, spell_checker, repeat_num):
    all_results, correction_history = spell_checker.repeat_make_corrections(sentences, num=repeat_num,
                                                                            is_train=False,
                                                                            train_on_difference=True)

    w = open(f'history.json', 'w', encoding='utf-8')
    w.write(json.dumps(correction_history, ensure_ascii=False, indent=4, sort_keys=False))
    w.close()

    # 对于命令行执行模式 将结果反馈到屏幕
    args = parse_args()
    if args.mode == 's':
        for i in range(len(correction_history)):
            print('纠正前：', all_results[i]["original_sentence"])
            print('纠正后：', all_results[i]["corrected_sentence"])

    #将每一个句子的纠错结果写入到result_{i}.josn中
    for i, res in enumerate(all_results):
        w = open(f'results_{i}.json', 'w', encoding='utf-8')
        w.write(json.dumps(res, ensure_ascii=False, indent=4, sort_keys=False))
        w.close()



#测试单元 用验证CSC的效果
def test_unit(res, test_path, out_name, strict=True):
    out = open(f'{out_name}.txt', 'w', encoding='utf-8') #存入对每一个待CSC句子的性能检查结果

    corrected_char = 0 #统计在CSC句子中 所有纠正过的char数目
    wrong_char = 0 #统计在CSC句子中 所有存在错误的char数目
    corrected_sent = 0
    wrong_sent = 0
    true_corrected_char = 0 #正确纠正的char数目
    true_corrected_sent = 0
    true_detected_char = 0
    true_detected_sent = 0 #句子级别 检错正确的句子数目
    accurate_detected_sent = 0
    accurate_corrected_sent = 0
    all_sent = 0


    for idx, line in enumerate(open(test_path, 'r', encoding='utf-8')):
        all_sent += 1
        #以下3个变量 没有使用 可以注释掉
        falsely_corrected_char_in_sentence = 0
        falsely_detected_char_in_sentence = 0
        true_corrected_char_in_sentence = 0

        num, wrong, correct = line.strip().split('\t')
        predict = res[idx]["corrected_sentence"]
        
        wrong_num = 0 #记录一个句子中 错误纠正的char数目
        corrected_num = 0 #记录一个句子中 纠正的char数目
        original_wrong_num = 0 #记录一个句子中 存在错误的char数目
        true_detected_char_in_sentence = 0 #记录一个句子中正确检查的char数目

        #通过循环的方式 确定各个指标的具体值
        for c, w, p in zip(correct, wrong, predict):
            if c != p:#正确的char不等于模型输出的char
                wrong_num += 1 #错误纠正的char数目+1
            if w != p:#原句的char不等于模型输出的char
                corrected_num += 1 #纠正过的char次数+1
                if c == p:#正确纠正char数目+1
                    true_corrected_char += 1
                if w != c:#正确检查char+1 正确检查句子中的char+1
                    true_detected_char += 1
                    true_detected_char_in_sentence += 1
            if c != w:#正确的char不等于原始的char
                original_wrong_num += 1#原句中char错误的数目+1
        #写入对该句子的性能检查结果
        out.write('\t'.join([str(original_wrong_num), wrong, correct, predict, str(wrong_num)]) + '\n')
        corrected_char += corrected_num  #统计所有纠正过的char数目
        wrong_char += original_wrong_num #统计所有存在错误的char数目
        if original_wrong_num != 0:#检查该句子中是否有错误的char
            wrong_sent += 1  #错误句子数目+1 统计所有存在错误的句子数目
        if corrected_num != 0 and wrong_num == 0:#二者均满足的时候 句子中纠正过且纠正全部正确
            true_corrected_sent += 1 #正确纠正的句子数目+1
        if corrected_num != 0:#只满足纠正了句子
            corrected_sent += 1#纠正的句子数目+1
        if strict:#是否严格检查句子级别检错结果 严格：所有检查出的位置是存在错误的位置
            true_detected_flag = (true_detected_char_in_sentence == original_wrong_num and original_wrong_num != 0 and corrected_num == true_detected_char_in_sentence)
        else:#不严格检查： 句子存在错误 且有检查痕迹
            true_detected_flag = (corrected_num != 0 and original_wrong_num != 0)
        # if corrected_num != 0 and original_wrong_num != 0:
        if true_detected_flag:#检错结果  句子级别检错 只要句子中存在错误 且判断模型有改动 则认为是正确的检查出了错误的句子
            #字词级别检错 句子中存在错误 且所有错误的地方均被正确的检查出来
            true_detected_sent += 1
        if correct == predict:#满足模型输出句子 等于 正确句子
            accurate_corrected_sent += 1 #输出句子等于correct的数目+1（就是正确的句子不做改变+错误的句子纠正正确 Tp + Tn）
        if correct == predict or true_detected_flag: #满足模型输出句子 等于 正确句子 或者检错结果正确
            accurate_detected_sent += 1 #正确检查出的句子数目+1

    print("corretion:")
    print(f'char_p={true_corrected_char}/{corrected_char}') #正确纠正的char数目/所有纠正的char数目
    print(f'char_r={true_corrected_char}/{wrong_char}')#正确纠正的char数目/所有存在错误的char数目
    print(f'sent_p={true_corrected_sent}/{corrected_sent}')#正确纠正的句子数目/所有纠正的句子数目
    print(f'sent_r={true_corrected_sent}/{wrong_sent}') #正确纠正的句子数目/所有存在错误的句子数目
    print(f'sent_a={accurate_corrected_sent}/{all_sent}') #（Tp +Tn）/ all_sents
    print("detection:")
    print(f'char_p={true_detected_char}/{corrected_char}')#正确检查出的char数目 / 所有检查出的char数目
    print(f'char_r={true_detected_char}/{wrong_char}')#正确检查出的char数目 / 所有存在错误的char数目
    print(f'sent_p={true_detected_sent}/{corrected_sent}')#正确检查出错误的句子数目 / 所有纠正的句子数目
    print(f'sent_r={true_detected_sent}/{wrong_sent}')#正确检查出错误的句子数目 / 所有存在错误的句子数目
    print(f'sent_a={accurate_detected_sent}/{all_sent}')#（Tp + Tn )/ all_sentences

    #将模型CSC后的结果又写入到out_name.json文件中 方便后续试验观察
    w = open(f'{out_name}.json', 'w', encoding='utf-8')
    w.write(json.dumps(res, ensure_ascii=False, indent=4, sort_keys=False))
    w.close()


def parse_args():
    usage = '\n1. You can spell check several sentences by:\n' \
            'python faspell.py 扫吗关注么众号 受奇艺全网首播 -m s\n' \
            '\n' \
            '2. You can spell check a file by:\n' \
            'python faspell.py -m f -f /path/to/your/file\n' \
            '\n' \
            '3. If you want to do experiments, use mode e:\n' \
            ' (Note that experiments will be done as configured in faspell_configs.json)\n' \
            'python faspell.py -m e\n' \
            '\n' \
            '4. You can train filters under mode e by:\n' \
            'python faspell.py -m e -t\n' \
            '\n' \
            '5. to train filters on difference under mode e by:\n' \
            'python faspell.py -m e -t -d\n' \
            '\n'
    parser = argparse.ArgumentParser(
        description='A script for FASPell - Fast, Adaptable, Simple, Powerful Chinese Spell Checker', usage=usage)

    parser.add_argument('multiargs', nargs='*', type=str, default=None,
                        help='sentences to be spell checked')
    parser.add_argument('--mode', '-m', type=str, choices=['s', 'f', 'e'], default='s',
                        help='select the mode of using FASPell:\n'
                             ' s for spell checking sentences as args in command line,\n'
                             ' f for spell checking sentences in a file,\n'
                             ' e for doing experiments on FASPell')
    parser.add_argument('--file', '-f', type=str, default=None,
                        help='under mode f, a file to be spell checked should be provided here.')
    parser.add_argument('--difference', '-d', action="store_true", default=False,
                        help='train on difference')
    parser.add_argument('--train', '-t', action="store_true", default=False,
                        help='True=to train FASPell with confidence-similarity graphs, etc.'
                             'False=to use FASPell in production')

    args = parser.parse_args()
    return args


def main():
    spell_checker = SpellChecker()
    args = parse_args()
    if args.mode == 's':  # command line mode
        try:

            assert args.multiargs is not None
            assert not args.train

            logging.basicConfig(level=logging.INFO)
            repeat_non_test(args.multiargs, spell_checker, CONFIGS["general_configs"]["round"])

        except AssertionError:
            print("Sentences to be spell checked cannot be none.")

    elif args.mode == 'f':  # file mode
        try:
            assert args.file is not None
            sentences = []
            for sentence in open(args.file, 'r', encoding='utf-8'):
                sentences.append(sentence.strip())
            repeat_non_test(sentences, spell_checker, CONFIGS["general_configs"]["round"])

        except AssertionError:
            print("Path to a txt file cannot be none.")

    elif args.mode == 'e':  # experiment mode
        
        if args.train:
            repeat_test(CONFIGS["exp_configs"]["training_set"], spell_checker, CONFIGS["general_configs"]["round"],
                    args.train, train_on_difference=args.difference)
            # assert not CONFIGS["exp_configs"]["union_of_sims"]  # union of sims is a strategy used only in testing
            name = f'difference_{int(args.difference)}-rank_{CONFIGS["general_configs"]["rank"]}-results_0' #绘制结果图以及输出相应字段信息，这里也提醒注意：在训练时round只能为1，因为此处只是能对第一轮round的结果进行绘制
            plot.plot(f'{name}.json',
                      f'{name}.txt',
                      store_plots=CONFIGS["exp_configs"]["store_plots"],
                      plots_to_latex=CONFIGS["exp_configs"]["store_latex"])
        else:
            repeat_test(CONFIGS["exp_configs"]["testing_set"], spell_checker, CONFIGS["general_configs"]["round"],
                    args.train, train_on_difference=args.difference)


if __name__ == '__main__':
    main()

    '''
    Faspell的纠错策：给定一个句子 abcdef 指定纠正的rank（候选集大小） round（重复纠正次数）
    
    对于a进入MLM后 选择前rank个token（按概率）作为候选集 （a1,a2,a3）
    在该rank内循环，依次比较a和a1，a2，a3的值 如果a = a1继续比较后面的候选，如果a！=a1 则对a和a1进行判断
    根据a1与a的similarity和confidence 进入之前训练好的分类器，如果判定为是一个错误，则将a改为a1 否则继续比较后面
    的候选。当比较完候选集仍没找到错误时，则判定a为正确，不做修正。
    所以对于FASPell而言：
    1.MLM能将越多的错误修正确定在前rank中以及不将正确的改为错误的能力很重要，这直接决定了模型的上线；
    2.对于CSD的分类器也异常重要，如何确保模型在rank中找到正确的字，过滤不必要的修改以及不正确的修改，
    对于提高魔性的性能起到了关键作用。
    
    '''