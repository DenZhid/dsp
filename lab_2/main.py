import math

import numpy as np

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
    plt.ylabel('Амплитуда, В')
    plt.xlabel('Время, с.')
    plt.show()

    # Дискретизация сигнала
    disc_signal = signal_receiver.discretize(duration, gen_signal)

    # Построение результата дискретизации сигнала
    plt.plot(t, disc_signal)
    plt.title('Дискретизированный сигнал')
    plt.ylabel('Уровень сигнала')
    plt.xlabel('Время, с.')
    plt.show()

    # Перенос частоты несущей, получение исходного сигнала
    signal_2 = signal_receiver.butter_filter(duration, disc_signal, freq=383, coeff=1, filter_n=2)
    signal_4 = signal_receiver.butter_filter(duration, disc_signal, freq=383, coeff=1, filter_n=4)
    signal_6 = signal_receiver.butter_filter(duration, disc_signal, freq=383, coeff=1, filter_n=6)


    # Построение результата первой фильтрации, исходный сигнал
    plt.plot(t, signal_2, label='Фильтр 2-го порядка', color='cornflowerblue')
    plt.plot(t, signal_4, label='Фильтр 4-го порядка', color='darkorange')
    plt.plot(t, signal_6, label='Фильтр 6-го порядка', color='forestgreen')
    plt.title('Результат первого переноса частоты')
    plt.ylabel('Уровень сигнала')
    plt.xlabel('Время, с.')
    plt.legend()
    plt.show()

    # Второе пропускание сигнала через фильтр Баттерворта
    signal_detection_2 = signal_receiver.butter_filter(duration, signal_2, freq=14, coeff=8, filter_n=2)
    signal_detection_4 = signal_receiver.butter_filter(duration, signal_4, freq=14, coeff=8, filter_n=4)
    signal_detection_6 = signal_receiver.butter_filter(duration, signal_6, freq=14, coeff=8, filter_n=6)


    N2 = len(signal_detection_2[2080:])
    lower_bound_2 = ([sum(signal_detection_2[2080:]) / N2 * 0.708] * duration)[0]
    lower_bound_4 = ([sum(signal_detection_4[2080:]) / N2 * 0.708] * duration)[0]
    lower_bound_6 = ([sum(signal_detection_6[2080:]) / N2 * 0.708] * duration)[0]

    # Построение результата второй фильтрации
    plt.plot(t, signal_detection_2, label='Фильтр 2-го порядка', color='cornflowerblue')
    plt.plot(t, signal_detection_4, label='Фильтр 4-го порядка', color='darkorange')
    plt.plot(t, signal_detection_6, label='Фильтр 6-го порядка', color='forestgreen')
    plt.plot(t, np.full((duration, 1), lower_bound_2), label='Порог 2-го порядка', color='cornflowerblue')
    plt.plot(t, np.full((duration, 1), lower_bound_4), label='Порог 4-го порядка', color='darkorange')
    plt.plot(t, np.full((duration, 1), lower_bound_6), label='Порог 6-го порядка', color='forestgreen')
    plt.title('Результаты второго переноса частоты')
    plt.ylabel('Уровень сигнала')
    plt.xlabel('Время, с.')
    plt.legend()
    plt.show()

    # Определение наличия сигнала
    signal_presence_2 = signal_receiver.determine_signal_presence(duration, signal_detection_2, lower_bound_2)
    signal_presence_4 = signal_receiver.determine_signal_presence(duration, signal_detection_4, lower_bound_4)
    signal_presence_6 = signal_receiver.determine_signal_presence(duration, signal_detection_6, lower_bound_6)

    # Построение графика присутствия сигнала
    plt.plot(t, signal_presence_2, '*', color='cornflowerblue')
    plt.title('Результаты определения наличия сигнала для фильтра 2-го порядка')
    plt.ylabel('Наличие сигнала (0 - нет, 1 - есть')
    plt.xlabel('Время, с.')
    plt.show()

    plt.plot(t, signal_presence_4, '*', color='darkorange')
    plt.title('Результаты определения наличия сигнала для фильтра 4-го порядка')
    plt.ylabel('Наличие сигнала (0 - нет, 1 - есть')
    plt.xlabel('Время, с.')
    plt.show()

    plt.plot(t, signal_presence_6, '*', color='forestgreen')
    plt.title('Результаты определения наличия сигнала для фильтра 6-го порядка')
    plt.ylabel('Наличие сигнала (0 - нет, 1 - есть')
    plt.xlabel('Время, с.')
    plt.show()

    # Определение задержки определителя
    signal_receiver.determine_delay(signal_presence_2, t)
    signal_receiver.determine_delay(signal_presence_4, t)
    signal_receiver.determine_delay(signal_presence_6, t)

    # Зашумление сигнала
    fn_mod = 510
    phin_mod = 0

    mod_signal = []
    for i in range(duration):
        mod_signal.append(gen_signal[i] + 0.5 * math.sin(2 * math.pi * fn_mod * t[i] + phin_mod))

    # Построение зашумленного сигнала
    plt.plot(t, mod_signal)
    plt.title('Зашумленный сигнал')
    plt.xlabel('Время, с.')
    plt.ylabel('Амплитуда, В')
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