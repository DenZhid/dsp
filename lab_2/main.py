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
    signal2 = signal_receiver.butter_filter(duration, disc_signal, freq=360, coeff=1, filter_n=2)
    signal4 = signal_receiver.butter_filter(duration, disc_signal, freq=360, coeff=1, filter_n=4)
    signal6 = signal_receiver.butter_filter(duration, disc_signal, freq=360, coeff=1, filter_n=6)


    # Построение результата первой фильтрации, исходный сигнал
    plt.plot(t, signal2, label='Второй порядок', color='cornflowerblue')
    plt.plot(t, signal4, label='Четвертый порядок', color='darkorange')
    plt.plot(t, signal6, label='Шестой порядок', color='darkorange')
    plt.title('Результат первого переноса частоты')
    plt.ylabel('А, В')
    plt.xlabel('t, с.')
    plt.show()

    # Второе пропускание сигнала через фильтр Баттерворта
    signal_detection2 = signal_receiver.butter_filter(duration, signal2, freq=10, coeff=8, filter_n=2)
    signal_detection4 = signal_receiver.butter_filter(duration, signal4, freq=10, coeff=8, filter_n=4)
    signal_detection6 = signal_receiver.butter_filter(duration, signal6, freq=10, coeff=8, filter_n=6)

    N2 = len(signal_detection2[1743:])
    sum_detection2 = [sum(signal_detection2[1743:]) / N2 * 0.708] * len(t)
    sum_detection4 = [sum(signal_detection4[1743:]) / N2 * 0.708] * len(t)
    sum_detection6 = [sum(signal_detection6[1743:]) / N2 * 0.708] * len(t)

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
    lower_bound2 = sum_detection2[0]
    lower_bound4 = sum_detection4[0]
    lower_bound6 = sum_detection6[0]
    signal_presence2 = [1] * duration
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

    # Зашумление сигнала
    fn_mod = 510
    phin_mod = 0

    mod_signal = []
    for i in range(duration):
        mod_signal.append(gen_signal[i] + 0.5 * math.sin(2 * math.pi * fn_mod * t[i] + phin_mod))

    # Построение зашумленного сигнала
    plt.plot(t, mod_signal)
    plt.title('Зашумленный сигнал')
    plt.xlabel('t, сек')
    plt.ylabel('A, В')
    plt.axis([0, 1.6, -1.5, 1.5])
    plt.show()

    # Анализ спектра сигнала
    freq_d = signal_receiver.freq_d
    fourier_amp_gen_signal, freq_array = custom_fft(duration, gen_signal, freq_d)
    fourier_amp_mod_signal, _ = custom_fft(duration, mod_signal, freq_d)

    # Построение спектра исходного сигнала
    plt.plot(freq_array, fourier_amp_gen_signal[0:len(freq_array)])
    plt.title('Спектр исходного сигнала')
    plt.xlabel('f, Гц')
    plt.ylabel('|Y(f)|')
    plt.axis([210, 560, 0, 1])
    plt.show()

    # Построение спектра зашумленного сигнала
    plt.plot(freq_array, fourier_amp_mod_signal[0:len(freq_array)])
    plt.title('Спектр зашумленного сигнала')
    plt.xlabel('f, Гц')
    plt.ylabel('|Y(f)|')
    plt.axis([210, 560, 0, 1])
    plt.show()

    ans_gen_signal, fc1_gen_signal, fc2_gen_signal, fc_gen_signal = fta(freq_array, fourier_amp_gen_signal)
    ans_mod_signal, fc1_mod_signal, fc2_mod_signal, fc_mod_signal = fta(freq_array, fourier_amp_mod_signal)

    print(
        f"Для исходного сигнала: \n"
        f"ans={ans_gen_signal}\n"
        f"Fc1={fc1_gen_signal}\n"
        f"Fc2={fc2_gen_signal}\n"
        f"Fc={fc_gen_signal}\n"
    )
    print(
        f"Для зашумленного  сигнала: \n"
        f"ans={ans_mod_signal}\n"
        f"Fc1={fc1_mod_signal}\n"
        f"Fc2={fc2_mod_signal}\n"
        f"Fc={fc_mod_signal}\n"
    )


if __name__ == '__main__':
    main()