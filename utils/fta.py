
def fta(freq, y):
    L = 2
    frequency_length = len(freq)
    FP = [0] * frequency_length

    for i in range(L, frequency_length - L):
        tmp1 = 0
        tmp2 = 0
        for j in range(L - 2):
            tmp1 = tmp1 + y[i - j - 1]
            tmp2 = tmp2 + y[i + j + 1]
        FP[i + 1] = 1 / (L - 1) * tmp1 + 1 / (L - 1) * tmp2

    val = max(FP)
    num = FP.index(val)
    # Значение несущей
    ans = freq[num]

    # Вычисление "левой" частоты
    Fc1 = None
    for i in range(frequency_length):
        if (y[num - i] < y[num - i - 1]) and (y[num - i - 1] > y[num - i - 2]):
            Fc1 = freq[num - i - 1]
            break

    # Вычисление "правой" частоты
    Fc2 = None
    for i in range(frequency_length):
        if (y[num + i] < y[num + i + 1]) and (y[num + i + 1] > y[num + i + 2]):
            Fc2 = freq[num + i + 1]
            break

    # вычисление частоты сигнала
    Fc = (ans - Fc1 + Fc2 - ans) / 2
    return ans, Fc1, Fc2, Fc
