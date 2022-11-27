import math

from utils.signal_receiver import SignalReceiver
from utils.fft import custom_fft
from utils.fta import fta
import matplotlib.pyplot as plt


# Variant 61: Fn = 383 Fm = 14 K = 128
def main():
    signal_receiver = SignalReceiver(
        freq_n=383, # несущая частота (Fn)
        freq_m=14, # частота модуляции (Fm)
        phase_n=0, # фаза несущей частоты (произвольная)
        phase_m=0, # фаза частоты модуляции (произвольная)
        modulation_factor=1, # глубина модуляции (1)
        quantization=128, # число уровней квантования (K)
        k_disc=8, # отношение частоты дискретизации к частоте несущей
        t_end=2.0 # время окончания модуляции сигнала
    )
    # Генерация сигнала
    t, gen_signal = signal_receiver.generate_signal()
    duration = len(t)

    # Построение сгенерированного сигнала
    plt.plot(t, gen_signal)
    plt.title('Модулированный сигнал')
    plt.ylabel('А, В')
    plt.xlabel('t, с.')
    plt.show()

    # Дискретизация сигнала
    disc_signal = signal_receiver.discretize(duration, gen_signal)

    # Построение результата дискретизации сигнала
    plt.plot(t, disc_signal)
    plt.title('Импульсный сигнал АЦП')
    plt.ylabel('Квантованные значения сигнала')
    plt.xlabel('t, с.')
    plt.show()

    # Перенос частоты несущей, получение исходного сигнала
    signal2 = signal_receiver.butter_filter(duration, disc_signal, freq=383, coeff=1, filter_n=2)
    signal4 = signal_receiver.butter_filter(duration, disc_signal, freq=383, coeff=1, filter_n=4)
    signal6 = signal_receiver.butter_filter(duration, disc_signal, freq=383, coeff=1, filter_n=6)


    # Построение результата первой фильтрации, исходный сигнал
    plt.plot(t, signal2, label='Второй порядок', color='cornflowerblue')
    plt.plot(t, signal4, label='Четвертый порядок', color='darkorange')
    plt.plot(t, signal6, label='Шестой порядок', color='darkorange')
    plt.title('Результат первого переноса частоты')
    plt.ylabel('А, В')
    plt.xlabel('t, с.')
    plt.legend(loc='lower right')
    plt.show()

    # Второе пропускание сигнала через фильтр Баттерворта
    signal_detection2 = signal_receiver.butter_filter(duration, signal2, freq=14, coeff=8, filter_n=2)
    signal_detection4 = signal_receiver.butter_filter(duration, signal4, freq=14, coeff=8, filter_n=4)
    signal_detection6 = signal_receiver.butter_filter(duration, signal6, freq=14, coeff=8, filter_n=6)


    N2 = len(signal_detection2[2080:])
    lower_bound2 = ([sum(signal_detection2[2080:]) / N2 * 0.708] * duration)[0]
    lower_bound4 = ([sum(signal_detection4[2080:]) / N2 * 0.708] * duration)[0]
    lower_bound6 = ([sum(signal_detection6[2080:]) / N2 * 0.708] * duration)[0]

    print(sum_detection2[0])
    print(sum_detection4[0])
    print(sum_detection6[0])
    # Построение результата второй фильтрации
    plt.plot(t, signal_detection2, label='Фильтр 2-го порядка', color='cornflowerblue')
    plt.plot(t, signal_detection4, label='Фильтр 4-го порядка', color='darkorange')
    plt.plot(t, signal_detection6, label='Фильтр 6-го порядка', color='forestgreen')
    plt.plot(t, sum_detection2, label='2-ой порог', color='cornflowerblue')
    plt.plot(t, sum_detection4, label='4-ый порог', color='darkorange')
    plt.plot(t, sum_detection6, label='6-ой порог', color='forestgreen')
    plt.title('Результаты второго переноса частоты')
    plt.ylabel('A, В')
    plt.xlabel('t, с.')
    plt.legend(loc='lower right')
    plt.show()

    # Определение наличия сигнала
    signal_presence2 = signal_receiver.determine_signal_presence(duration, signal_detection2, lower_bound2)
    signal_presence4 = [1] * duration
    signal_presence6 = [1] * duration
    for i in range(duration):
        if signal_detection2[i] <= lower_bound2:
            signal_presence2[i] = 0
        else:
            break

    for i in range(duration):
        if signal_detection4[i] <= lower_bound4:
            signal_presence4[i] = 0
        else:
            break

    for i in range(duration):
        if signal_detection6[i] <= lower_bound6:
            signal_presence6[i] = 0
        else:
            break

    # Построение графика присутствия сигнала
    plt.plot(t, signal_presence2, '*', color='cornflowerblue')
    plt.title('Результаты определения наличия сигнала 2-го порядка')
    plt.ylabel('Наличие сигнала (0 - нет, 1 - есть')
    plt.xlabel('t, с.')
    plt.show()

    plt.plot(t, signal_presence4, '*', color='darkorange')
    plt.title('Результаты определения наличия сигнала 4-го порядка')
    plt.ylabel('Наличие сигнала (0 - нет, 1 - есть')
    plt.xlabel('t, с.')
    plt.show()

    plt.plot(t, signal_presence6, '*', color='forestgreen')
    plt.title('Результаты определения наличия сигнала 6-го порядка')
    plt.ylabel('Наличие сигнала (0 - нет, 1 - есть')
    plt.xlabel('t, с.')
    plt.show()

    # Определение задержки определителя
    i = 0
    while signal_presence2[i] == 0:
        i += 1
    print('Задержка усилителя: ' + str(t[i]))

    i = 0
    while signal_presence4[i] == 0:
        i += 1
    print('Задержка усилителя: ' + str(t[i]))

    i = 0
    while signal_presence6[i] == 0:
        i += 1
    print('Задержка усилителя: ' + str(t[i]))

if __name__ == '__main__':
    main()