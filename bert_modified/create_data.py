import numpy as np
import pickle
import os
import argparse

def cut_line(sentence):
    # 对句子进行切割，如果句子中间存在为['。', '；', '？', '！']而不是['」', '”', '’']的句子
    # 则将该标点符号及之前的内容做为一个单独的句子。
    # eg S1= '我真的很喜欢你！你知道吗？' cut -> '我真的很喜欢你！'  &  '你知道吗？'
    # eg S2= '我真的很喜欢“天使”你知道吗？' cut -> '我真的很喜欢“天使”你知道吗？'
    #不过需要注意的是，得益于yield作为关键字，调用该函数的结果应该使用for循环遍历


    sent = ''
    delimiter = ['。', '；', '？', '！']
    for i, c in enumerate(sentence):
        sent += c
        if ((c in delimiter) and (sentence[min(len(sentence)-1, i + 1)] not in ['」', '”', '’'])) or i == len(sentence)-1:
            yield sent
            sent = ''
def cut_line2(sentence):
    #对含有标点符号[,]的句子进行过滤，主要是针对那些长度过长的句子进行cut，只有满足以下条件的句子才能被选择
    #从左往右的每一个[,] 只要存在一个是句子末尾或者满足该[,]后面的6个字符没有[,]且
    #到句子头的长度大于20，则返回该[,]前的内容，结尾符号为[。]
    sent = ''
    for i, c in enumerate(sentence):
        sent += c
        if c == '，':
            flag = True
            for j in range(i+1, min(len(sentence)-1, i+6)):
                if sentence[j] == '，' or j == len(sentence)-1:
                    # 实际上，这个地方无论怎么样 都不能达到j == len(sentence)-1的条件
                    flag = False

            if (flag and len(sent) > 20) or i == len(sentence)-1:
                yield sent[:-1] + '。'
                sent = ''



def make_docs(wrong, correct):
    #按照上述两个句子的cut方法将训练数据的wrong和correct句子进行处理，并写入到对应的文件中
    #我觉得这个方法的逻辑写的有点混乱
    w_res = []
    #对满足下面条件的句子进行切分
    if ('。' in wrong[:-1]) or ('；' in wrong[:-1]) or ('？' in wrong[:-1]) or ('！' in wrong[:-1]):
        for w_sent in cut_line(wrong):
            w_res.append(w_sent + '\n')
            # wrong_file.write(w_sent + '\n')
    elif len(wrong) > 100:#对句子长度大于100的句子切分
        for w_sent in cut_line2(wrong):
            w_res.append(w_sent + '\n')
            # wrong_file.write(w_sent + '\n')
    else:#不做切分处理
        w_res.append(wrong + '\n')
        # wrong_file.write(wrong + '\n')

    # wrong_file.write('\n')
    c_res = []
    if ('。' in correct[:-1]) or ('；' in correct[:-1]) or ('？' in correct[:-1]) or ('！' in correct[:-1]):
        for c_sent in cut_line(correct):
            c_res.append(c_sent + '\n')
            # correct_file.write(c_sent + '\n')
    elif len(wrong) > 100:
        for c_sent in cut_line2(correct):
            c_res.append(c_sent + '\n')
            # correct_file.write(c_sent + '\n')
    else:
        c_res.append(correct + '\n')
        # correct_file.write(correct + '\n')

    #按照上述方法处理后的句子结果检查，如果两个list的长度不相等，则不做上述处理，直接将原句放入list
    if len(w_res) != len(c_res):
        w_res = [wrong + '\n']
        c_res = [correct + '\n']

    #如果wrong和correct对应的句子长度不相等，exit()
    for w_r, c_r in zip(w_res, c_res):
        if not len(w_r.strip()) == len(c_r.strip()):
            print(w_r)
            print(len(w_r.strip()))
            print(c_r)
            print(len(c_r.strip()))
            exit()
    #检查过后将处理的数据写入到文件中
    for l in w_res:
        wrong_file.write(l)
    wrong_file.write('\n')

    for l in c_res:
        correct_file.write(l)
    correct_file.write('\n')


def main(fname, output_dir):
    confusions = {}

    for line in open(fname, 'r', encoding='utf-8'):
        #读入预训练模型使用的数据 eg： 1	把这个橘子瓣辨	把这个橘子瓣瓣  对应的是：错误数目 错误句子 正确句子
        #print(line)
        #print(line.strip().split('\t'))
        num, wrong, correct = line.strip().split('\t')
        wrong = wrong.strip()
        correct = correct.strip()
        for w, c in zip(wrong, correct):
            #如果两个句子对应对应位置的字符不一致
            if w!=c:
                #将w+c添加到confusion字典中， 构造困惑替换集
                #eg：wrong: abcd correct:cbcd    confusion={'ac':0} ac是一个困惑替换对 a被替换成了c
                if w + c not in confusions:
                    confusions[w + c] = 0
                confusions[w + c] += 1
        # if len(wrong) != len(correct):
        #     print(wrong)
        #     print(correct)
        #     exit()

        #判断wrong 和 correct 句子的长度是否对等
        assert len(wrong) == len(correct)
        #在该组wrong和correct对中存在的错误次数
        num = int(num)
        #处理wrong和correct句子对
        make_docs(wrong, correct)
        #对于wrong和correct存在不一致的句子对，使用correct-correct构建负样本
        if wrong != correct:
            make_docs(correct, correct)

        #统计wrong和correct中 对应位置不一致的字符所对应的index
        #wrong: abcd correct:cbcd         poses=[0]  对应不一致的下标为0
        poses = [pos for pos, (w, c) in enumerate(zip(wrong, correct)) if w != c]
        num = len(poses) #此处这一条语句没有意义（下面的判断：if len(poses) != num就不应该存在了） 所以我在运行的时候是注释掉该条语句的



        #对于wrong correct存在不一致的字符数大于等于2的，以此为基础继续构建错误
        #eg: wrong: abcde correct:fghde there num=3
        # -> wrong1 corect wrong1为上述三个错误中的任意一个
        # -> wrong2 corect wrong2为上述三个错误中的任意二个
        if num >= 2:
            #判断 如果计算出wrong和correct不一致的字符数 不等于 num 退出
            if len(poses) != num:
                print(wrong)
                print(correct)
                exit()
            assert len(poses) == num
            for i in range(1, num):
                selected_poses = [poses[k] for k in np.random.choice(num, i, replace=False)]
                fake_wrong = list(wrong)
                for p in selected_poses:
                    fake_wrong[p] = correct[p]

                fake_wrong = ''.join(fake_wrong)
                assert len(fake_wrong) == len(correct)
                assert fake_wrong != correct
                make_docs(fake_wrong, correct)

    # take the top frequency of confusions about the each character.

    #在训练集中 一个字被替换为哪一个字的频率最高，则选择这个替换对出现的次数作为该字的被替换频率
    top_confusions = {}
    for k in confusions:
        if k[0] not in top_confusions:
            top_confusions[k[0]] = confusions[k]
        elif top_confusions[k[0]] < confusions[k]:
            top_confusions[k[0]] = confusions[k]

    #按照频率进行排序，并只取key值
    confusions_top = sorted(list(top_confusions.keys()), key=lambda x: top_confusions[x], reverse=True)

    #对于那些在某些句子中被认为是错字的字 在另外句子中却又是正确字 的字的按频率统计
    correct_count = {}
    for line_c, line_w in zip(open(os.path.join(args.output, 'correct.txt'), 'r', encoding='utf-8'), open(os.path.join(args.output, 'wrong.txt'), 'r', encoding='utf-8')):
        if line_c.strip():
            wrong, correct = line_w.strip(), line_c.strip()#strip()默认移除空格和换行 所以不需要显示指定strip('\n')，所以下面紧跟的两条语句实际也不需要
            wrong = wrong.strip()
            correct = correct.strip()
            for w, c in zip(wrong, correct):
                if w==c and w in top_confusions:
                    if w not in correct_count:
                        correct_count[w] = 0
                    correct_count[w] += 1
    #对于一个字 有时会是错字 有时会是正确字 的比例计算 p = min(被替换的次数 / 未被替换的次数,1.0) 也是后面的MASK概率
    proportions = {}
    for k in correct_count:
        assert correct_count[k] != 0
        proportions[k] = min(top_confusions[k] / correct_count[k], 1.0)

    print('confusion statistics:')

    for i in range(min(len(confusions_top), 20)):
        if confusions_top[i] in correct_count:
            correct_occurs = correct_count[confusions_top[i]]
            proportions_num = proportions[confusions_top[i]]
        else:
            correct_occurs = 0
            proportions_num = 'NaN'
        print(f'most frequent confusion pair for {confusions_top[i]} occurs {top_confusions[confusions_top[i]]} times,'
              f' correct ones occur {correct_occurs} times, mask probability should be {proportions_num}')

    pickle.dump(proportions, open(os.path.join(args.output, 'mask_probability.sav'), 'wb'))
    # print('top confusions:')
    # for i in range(20):
    #     print(f'{top_confusions[i]} occurs {confusions[confusions_top[i]]} times')

'''
对于出现的几个字典的详细解释：
confusions 用来记录困惑替换集  key：错误句子中的一个字被替换成正确句子中的另外一个字（相当于被纠正的意思） value：出现频率
top_confusions 一个字自己对应的困惑替换对中 选取频率最高的作为自己被替换（出错）的频率
            key：被替换的字 value：替换频率
confusions_top 对top_confusions的排序（从高到低） 并只保留key值 是一个list
correct_count 对于那些在某些句子中是错字 但在另外句子中又是正确的字的统计
            key:既会被认为是错字又会被认为是正确字的字 value：被认为是正确的字的频率

proportions key:既会被认为是错字又会被认为是正确字的字 value：p = min(被替换的次数 / 未被替换的次数,1.0)

eg:
对于字    ‘天’ 在训练集wrong中出现了4次 对应correct中分别是 ‘田’‘田’‘填’ ‘天’
 confusions = {'天田':2, '天填':1}    
 top_confusions = {'天':2}   
 confusions_top = ['天']
 correct_count = {'天':1}
 proportions = {'天':1}
'''

# main()
def parse_args():
    #使用命令介绍 该py文件是用来根据原始文件创建爱对应的与训练集合mask概率值
    usage = '\n1. create wrong.txt, correct.txt and mask_probability.sav by:\n' \
            'python create_data.py -f /path/to/train.txt\n' \ 
            '\n' \
            '\n2. specify output dir by:\n' \
            'python create_data.py -f /path/to/train.txt -o /path/to/dir/\n' \
            '\n' 
    parser = argparse.ArgumentParser(
        description='A module for FASPell - Fast, Adaptable, Simple, Powerful Chinese Spell Checker', usage=usage)
    #原始输入文件
    parser.add_argument('--file', '-f', type=str, default=None,
                        help='original training data.')
    #指定输出文件的路径 默认为当前路径
    parser.add_argument('--output', '-o', type=str, default='',
                        help='output a file a dir; default is current directory.')
    # parser.add_argument('--verbose', '-v', action="store_true", default=False,
    #                     help='to show details of spell checking sentences under mode s')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    correct_file = open(os.path.join(args.output,'correct.txt'), 'w', encoding='utf-8')
    wrong_file = open(os.path.join(args.output,'wrong.txt'), 'w', encoding='utf-8')
    main(args.file, args.output)
