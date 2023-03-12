import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# 选择一列带入方法进行灰色预测
def GM11(data):
    data = np.array(data)
    n = len(data)
    # 数据检验，计算极比
    lambds = []
    for i in range(1, n):
        lambds.append(data[i - 1] / data[i])
    X = (np.exp(-2 / (n + 1)), np.exp(2 / (n + 1)))
    is_access = True
    for lambd in lambds:
        if lambd < X[0] or lambd > X[1]:
            is_access = False
    if is_access == False:
        print("数据检验未通过")
        return
    else:
        print("数据检验通过")
    # 构建灰色预测模型
    data_cumsum = data.cumsum()
    # 灰导数及临值生成数列
    ds = []
    zs = []
    for i in range(1, n):
        ds.append(data[i])
        zs.append(-1 / 2 * (data_cumsum[i - 1] + data_cumsum[i]))
    # 求a和b
    B = np.array(zs).reshape(n - 1, 1)
    one = np.ones(n - 1)
    B = np.c_[B, one]  # 加上一列
    Y = np.array(ds).reshape(n - 1, 1)
    a, b = np.dot(np.dot(np.linalg.inv(np.dot(B.T, B)), B.T), Y)
    print("a=" + str(a))
    print("b=" + str(b))

    c = b / a
    data_1_hat = []  # 累加预测值
    data_0_hat = []  # 原始预测值
    data_1_hat.append((data[0] - c) * (np.exp(-a * 0)) + c)
    data_0_hat.append((data_1_hat[0]))
    for i in range(1, n + 5):
        data_1_hat.append((data[0] - c) * (np.exp(-a * i)) + c)
        data_0_hat.append(data_1_hat[i] - data_1_hat[i - 1])
    print("预测值")
    for i in data_0_hat:
        print(i)
    # 模型检验 预测结果方差
    data_h = np.array(data_0_hat[0:n]).T
    sum_h = data_h.sum()
    mean_h = sum_h / n
    S1 = np.sum((data_h - mean_h) ** 2) / n
    ## 残差方差
    e = data - data_h
    sum_e = e.sum()
    mean_e = sum_e / n
    S2 = np.sum((e - mean_e) ** 2) / n
    ## 后验差比
    C = S2 / S1
    ## 结果
    if (C <= 0.35):
        print('1级，效果好')
    elif (C <= 0.5 and C >= 0.35):
        print('2级，效果合格')
    elif (C <= 0.65 and C >= 0.5):
        print('3级，效果勉强')
    else:
        print('4级，效果不合格')
    # 画图
    plt.figure(figsize=(9, 4), dpi=100)
    x1 = np.linspace(1, n, n)
    x2 = np.linspace(1, n + 5, n + 5)
    plt.subplot(121)
    plt.title('x^0')
    plt.plot(x2, data_0_hat, 'r--', marker='*')
    plt.scatter(x1, data, marker='^')
    plt.subplot(122)
    plt.title('x^1')
    plt.plot(x2, data_1_hat, 'r--', marker='*')
    plt.scatter(x1, data_cumsum, marker='^')
    plt.show()


if __name__ == '__main__':
    df = pd.DataFrame({"sale": [2, 4, 8, 17, 31, 72, 144], "population": [122, 144, 155, 177, 188, 199, 250]})
    print(GM11(df["sale"]))
