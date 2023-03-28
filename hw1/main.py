# @Author: 谭天一
# @Description: 根据参考文献Entropy_of_English_PeterBrown和给定语料库计算中文的一元、二元、三元信息熵
# @Date: 2023-3-27

import jieba
import math
import os


def load_data(path):
    # 这个程序项目由Mac构建，文件路径采用正斜杠'/'，在Windows系统运行请改成'\\'
    content_mix = ''
    novels = os.listdir(path)
    for novel in novels:
        if 'txt' in novel:  # Mac文件夹中含有.Ds_store，应该滤除
            novel_path = path + '/' + novel
            with open(novel_path, 'r', encoding='gb18030') as file:
                content = file.read()
                content_mix += content
                content_mix += '\n'
            file.close()
    return content_mix


def data_process(content, stopwords_path):
    with open(stopwords_path, 'r', encoding='utf-8') as file:
        stopwords = file.read().split('\n')
    file.close()
    stopwords.append('\n')
    for stopword in stopwords:
        content = content.replace(stopword, '')
    return content


def freq_count_one(content):
    """
    同时适用于以字为单位和以词为单位，
    当以字为单位时，输入应为<str>类型的文本
    当以词为单位时，输入应为<list>类型的已完成分词的文本
    本注释也适用于下方的freq_count_two(content)、freq_count_three(content)
    """
    freq_one = {}
    for word in content:
        freq_one[word] = freq_one.get(word, 0) + 1
    freq_one = dict(sorted(freq_one.items(), key=lambda x: x[1], reverse=True))
    return freq_one


def freq_count_two(content):
    freq_two = {}
    for i in range(len(content) - 1):
        freq_two[content[i], content[i + 1]] = freq_two.get((content[i], content[i + 1]), 0) + 1
    freq_two = dict(sorted(freq_two.items(), key=lambda x: x[1], reverse=True))
    return freq_two


def freq_count_three(content):
    freq_three = {}
    for i in range(len(content) - 2):
        freq_three[content[i], content[i + 1], content[i + 2]] = freq_three.get(
            (content[i], content[i + 1], content[i + 2]), 0) + 1
    freq_three = dict(sorted(freq_three.items(), key=lambda x: x[1], reverse=True))
    return freq_three


def entropy_calculate_one(frequency1):
    entropy = 0
    content_length = sum(frequency1.values())
    for word, freq in frequency1.items():
        prob = freq / content_length
        entropy -= prob * math.log(prob, 2)
    return entropy


def entropy_calculate_two(frequency2, frequency1):
    entropy = 0
    content_length = sum(frequency2.values())
    for word1, word2 in frequency2:
        prob = frequency2[(word1, word2)] / content_length
        prob_conditional = frequency2[(word1, word2)] / frequency1[word1]
        # P(x|y)可近似看作每个二元词组在语料库中出现的频数与以该二元词组的第一个词为词首的二元词组的频数的比值
        # 分母等价于该二元词组的第一个词在一元词组统计中的频数
        entropy -= prob * math.log(prob_conditional, 2)
    return entropy


def entropy_calculate_three(frequency3, frequency2):
    entropy = 0
    content_length = sum(frequency3.values())
    for word1, word2, word3 in frequency3:
        prob = frequency3[(word1, word2, word3)] / content_length
        prob_conditional = frequency3[(word1, word2, word3)] / frequency2[(word1, word2)]
        # P(x|y,z)可近似看作每个三元词组在语料库中出现的频数与以该三元词组的前两个词为词首的三元词组的频数的比值
        # 分母等价于该三元词组的前两个词在二元词组统计中的频数
        entropy -= prob * math.log(prob_conditional, 2)
    return entropy


if __name__ == '__main__':
    content = load_data('./jyxstxtqj_downcc')
    content = data_process(content, './cn_stopwords.txt')

    # 以字为单位求一元模型的信息熵
    frequency1 = freq_count_one(content)
    entropy1 = entropy_calculate_one(frequency1)
    print('以字为单位，一元模型：', entropy1)

    # 以字为单位求二元模型的信息熵
    frequency2 = freq_count_two(content)
    entropy2 = entropy_calculate_two(frequency2, frequency1)
    print('以字为单位，二元模型：', entropy2)

    # 以字为单位求三元模型的信息熵
    frequency3 = freq_count_three(content)
    entropy3 = entropy_calculate_three(frequency3, frequency2)
    print('以字为单位，三元模型：', entropy3)

    # jieba分词
    content_cut = jieba.lcut(content)

    # 以词为单位求一元模型的信息熵
    frequency1 = freq_count_one(content_cut)
    entropy1 = entropy_calculate_one(frequency1)
    print('以词为单位，一元模型：', entropy1)

    # 以词为单位求二元模型的信息熵
    frequency2 = freq_count_two(content_cut)
    entropy2 = entropy_calculate_two(frequency2, frequency1)
    print('以词为单位，二元模型：', entropy2)

    # 以词为单位求三元模型的信息熵
    frequency3 = freq_count_three(content_cut)
    entropy3 = entropy_calculate_three(frequency3, frequency2)
    print('以词为单位，三元模型：', entropy3)
