import json
import matplotlib.pyplot as plt
import os


def plot(json_fname, results_fname, store_plots='', plots_to_latex=''):
    name = '.'.join(json_fname.split('.')[:-1])

    data = json.loads(open(json_fname, 'r', encoding='utf-8').read())#对所有句子cSC后错误的结果 results = [[句子1的result],[句子2的result]，...]
    fi = open(results_fname, 'r', encoding='utf-8')#对所有句子csc后的结果 line = origin_num, wrong_sent, correct_sent, predict_sent, num

    # data for a confidence-similarity graph
    truely_detected_and_truely_corrected = [[], []]
    truely_detected_and_falsely_corrected = [[], []]
    falsely_detected = [[], []]

    count_of_absence_of_correct_chars = [0, 0]

    w3 = open(f'{name}_falsely_detected.txt', 'w', encoding='utf-8')
    w4 = open(f'{name}_falsely_corrected.txt', 'w', encoding='utf-8')

    for line, entry in zip(fi, data):
        origin_num, wrong_sent, correct_sent, predict_sent, num = line.strip().split('\t')
        pos_to_error = dict([(e["error_position"], e) for e in entry["errors"]])
        for pos, (w, c, p) in enumerate(zip(wrong_sent, correct_sent, predict_sent)):
            if w != c and w != p:#正确检查
                e = pos_to_error[pos]
                assert e["corrected_to"] == p
                if c != p:#正确检查错误纠正
                    candidatas = dict(sorted(list(e["candidates"].items()), reverse=True, key=lambda it: it[1])[:5])
                    absent = 'no'
                    if c not in candidatas:#查看正确字符是否出现在了candidatas中
                        count_of_absence_of_correct_chars[0] += 1
                        absent = 'yes'
                    truely_detected_and_falsely_corrected[0].append(e["confidence"])
                    truely_detected_and_falsely_corrected[1].append(e["similarity"])

                    w4.write('\t'.join([wrong_sent,
                                        f'pos={pos}',
                                        f'w={w}',
                                        f'c={c}',
                                        f'p={p}',
                                        f'sim={e["similarity"]}',
                                        f'absent={absent}']) + '\n')
                else:#正确检查且正确纠正
                    truely_detected_and_truely_corrected[0].append(e["confidence"])
                    truely_detected_and_truely_corrected[1].append(e["similarity"])

            elif w == c and w != p:#错误检查 本身正确却被判定为错误
                e = pos_to_error[pos]
                candidates = dict(sorted(list(e["candidates"].items()), reverse=True, key=lambda it: it[1])[:5])
                absent = 'no'
                if c not in candidates:#查看正确字符是否出现在了candidatas中
                    count_of_absence_of_correct_chars[1] += 1
                    absent = 'yes'

                falsely_detected[0].append(e["confidence"])
                falsely_detected[1].append(e["similarity"])

                w3.write('\t'.join([wrong_sent,
                                    f'pos={pos}',
                                    f'w={w}',
                                    f'c={c}',
                                    f'p={p}',
                                    f'sim={e["similarity"]}',
                                    f'absent={absent}']) + '\n')

            elif w!=c and w==p:#本身是错误的 模型没有检查出来 CSC在画图的时候 未考虑这一类点
                pass
                #print('==' * 20)
                #print(wrong_sent)
                #(w)
                #print(correct_sent)
                #print(predict_sent)
                #print('==' * 20)
    # print statistics
    print(f'In {len(truely_detected_and_falsely_corrected[0])} falsely corrected characters,'
          f' {count_of_absence_of_correct_chars[0]} are because of absent correct candidates.')
    print(f'In {len(falsely_detected[0])} falsely detected characters,'
          f' {count_of_absence_of_correct_chars[1]} are because of absent correct candidates.')

    #绘图
    plt.plot(truely_detected_and_truely_corrected[0], truely_detected_and_truely_corrected[1], 'ro',
             truely_detected_and_falsely_corrected[0], truely_detected_and_falsely_corrected[1], 'bo',
             falsely_detected[0], falsely_detected[1], 'x')
    plt.axis([0.0, 1.0, 0.0, 1.0])
    plt.show()

    #是否保存图中每个点的详细信息
    if plots_to_latex:
        produce_latex(truely_detected_and_truely_corrected,
                      truely_detected_and_falsely_corrected,
                      falsely_detected, os.path.join(plots_to_latex, f'{name}_latex.txt'))
    if store_plots:#保存图片 两张 一张是基础图 另一张是放大基础图中的右上角部分
        # plt.savefig(os.path.join(store_plots, f'{name}.png'))
        axes = plt.gca()
        # axes.set_xlim([0.95,1])
        # axes.set_ylim([0.0,0.3])
        plt.savefig(os.path.join(store_plots, f'{name}.png'))
        axes.set_xlim([0.95,1])
        axes.set_ylim([0.0,0.6])
        plt.savefig(os.path.join(store_plots, f'{name}2.png'))
        # plt.pause(0.0001)
        # plt.clf()


def produce_latex(truely_detected_and_truely_corrected, truely_detected_and_falsely_corrected, falsely_detected, path):
    #将绘制点的信息进行保存
    f = open(path, 'w')
    for a_x, a_y in zip(truely_detected_and_truely_corrected[0], truely_detected_and_truely_corrected[1]):
        f.write(f'({a_x},{a_y})[a]')
    for b_x, b_y in zip(truely_detected_and_falsely_corrected[0], truely_detected_and_falsely_corrected[1]):
        f.write(f'({b_x},{b_y})[b]')
    for c_x, c_y in zip(falsely_detected[0], falsely_detected[1]):
        f.write(f'({c_x},{c_y})[c]')

    f.close()
