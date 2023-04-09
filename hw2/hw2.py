# @Author: 谭天一
# @Date: 2023-4-9

import math
import numpy as np


def load_data(filepath):
    df = np.loadtxt(filepath, skiprows=1)
    return df


def gauss_N(x, mu, sigma):
    pdf = 1 / np.sqrt(2 * np.pi) / sigma * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    return pdf


def cal_EM(x):
    # Initialize parameters
    mu0 = np.min(x)
    mu1 = np.max(x)
    sigma0 = np.std(x)
    sigma1 = sigma0
    p0 = 0.5
    p1 = 0.5
    n = len(x)
    # Set Precision
    accuracy = 0.0000001
    # Step count
    step_count = 0
    while True:
        step_count += 1
        print(step_count, ': ', mu0, mu1, sigma0, sigma1, p0, p1)
        # Previous Memory
        p0_pre = p0
        p1_pre = p1
        mu0_pre = mu0
        mu1_pre = mu1
        sigma0_pre = sigma0
        sigma1_pre = sigma1
        # E-STEP
        gamma0 = p0 * gauss_N(x, mu0, sigma0) / (p0 * gauss_N(x, mu0, sigma0) + p1 * gauss_N(x, mu1, sigma1))
        gamma1 = 1 - gamma0
        # M-STEP
        n0 = np.sum(gamma0)
        n1 = n - n0
        mu0 = gamma0.dot(x) / n0
        mu1 = gamma1.dot(x) / n1
        sigma0 = np.sqrt(gamma0.dot((x - mu0) ** 2) / n0)
        sigma1 = np.sqrt(gamma1.dot((x - mu1) ** 2) / n1)
        p0 = n0 / n
        p1 = n1 / n
        # End loop judgment
        if abs(sigma1_pre - sigma1) < accuracy \
                and abs(sigma0_pre - sigma0) < accuracy \
                and abs(p0_pre - p0) < accuracy \
                and abs(p1_pre - p1) < accuracy \
                and abs(mu0_pre - mu0) < accuracy \
                and abs(mu1_pre - mu1) < accuracy:
            print("结束迭代：达到预定精度")
            break
        if step_count > 100000:
            print("结束迭代：达到最大步数")
            break
        if math.isnan(mu0) or math.isnan(mu1) or math.isnan(sigma0) or math.isnan(sigma1) or math.isnan(
                p0) or math.isnan(p1):
            print("结束迭代：出现NaN")
            break
    return mu0, mu1, sigma0, sigma1, p0, p1


def test(test_data, mu0, mu1, sigma0, sigma1):
    probability_girl = gauss_N(test_data, mu0, sigma0)
    probability_boy = gauss_N(test_data, mu1, sigma1)
    considered_boy = 0
    considered_girl = 0
    for i in range(len(probability_girl)):
        if probability_girl[i] >= probability_boy[i]:
            considered_girl += 1
        else:
            considered_boy += 1
    return considered_boy, considered_girl


path = './height_data.csv'
data = load_data(path)
# 划分测试集和训练集，比例7:3
girl = data[:500]
boy = data[500:]
girl_train = girl[:350]
girl_test = girl[350:]
boy_train = boy[:1050]
boy_test = boy[1050:]
data_train = np.concatenate([girl_train, boy_train])
# 用训练集进行训练
mu0, mu1, sigma0, sigma1, p0, p1 = cal_EM(data_train)
print("女生：均值=", mu0, "; 标准差=", sigma0, "; 权重=", p0)
print("男生：均值=", mu1, "; 标准差=", sigma1, "; 权重=", p1)
# 用测试集进行测试
girl_incorrect_identify, girl_correct_identify = test(girl_test, mu0, mu1, sigma0, sigma1)
boy_correct_identify, boy_incorrect_identify = test(boy_test, mu0, mu1, sigma0, sigma1)
girl_classification_accuracy = ('%.2f' % (girl_correct_identify / len(girl_test) * 100))
boy_classification_accuracy = ('%.2f' % (boy_correct_identify / len(boy_test) * 100))
print("女生测试集，识别正确：", girl_correct_identify, '/', len(girl_test), "正确率：", girl_classification_accuracy, '%')
print("男生测试集，识别正确：", boy_correct_identify, '/', len(boy_test), "正确率：", boy_classification_accuracy, '%')
