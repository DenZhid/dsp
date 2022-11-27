import random

import numpy as np

from utils.signal_receiver import SignalReceiver
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
    plt.title('Импульсный сигнал АЦП')
    plt.ylabel('Квантованные значения сигнала')
    plt.xlabel('Время, с.')
    plt.show()

    # Перенос частоты несущей, получение исходного сигнала
    signal = signal_receiver.butter_filter(duration, disc_signal, freq=383, coeff=1, filter_n=6)

    # Построение результата первой фильтрации, исходный сигнал
    plt.plot(t, signal)
    plt.title('Результат первого переноса частоты')
    plt.ylabel('Амплитуда, В')
    plt.xlabel('Время, с.')
    plt.show()

    # Второе пропускание сигнала через фильтр Баттерворта
    signal_detection = signal_receiver.butter_filter(duration, signal, freq=14, coeff=8, filter_n=6)

    N2 = len(signal_detection[2080:])
    lower_bound = ([sum(signal_detection[2080:]) / N2 * 0.708] * duration)[0]

    # Построение результата второй фильтрации
    plt.plot(t, signal_detection, label='Фильтр 6-го порядка')
    plt.plot(t, np.full((duration, 1), lower_bound), label='Порог фильтра 6-го порядка')
    plt.title('Результаты второго переноса частоты')
    plt.ylabel('Амплитуда, В')
    plt.xlabel('Время, с.')
    plt.legend()
    plt.show()

    # Определение наличия сигнала
    signal_presence = signal_receiver.determine_signal_presence(duration, signal_detection, lower_bound)

    # Построение графика присутствия сигнала
    plt.plot(t, signal_presence, '*', color='forestgreen')
    plt.title('Результаты определения наличия сигнала')
    plt.ylabel('Наличие сигнала (0 - нет, 1 - есть')
    plt.xlabel('t, с.')
    plt.show()

    # Определение задержки определителя
    signal_receiver.determine_delay(signal_presence, t)

    # Внесение шума в исходный сигнал
    noise_signal_10 = []
    noise_signal_35 = []
    noise_signal_65 = []
    for i in range(duration):
        noise_signal_10.append(gen_signal[i] * (1 + 0.10 * (random.random() * 2 - 1)))
        noise_signal_35.append(gen_signal[i] * (1 + 0.35 * (random.random() * 2 - 1)))
        noise_signal_65.append(gen_signal[i] * (1 + 0.65 * (random.random() * 2 - 1)))

    abs_list_noised_10 = list(map(lambda x: abs(x), noise_signal_10))
    abs_list_noised_35 = list(map(lambda x: abs(x), noise_signal_35))
    abs_list_noised_65 = list(map(lambda x: abs(x), noise_signal_65))

    noise_signal_10 = list(map(lambda x: x / max(abs_list_noised_10), noise_signal_10))
    noise_signal_35 = list(map(lambda x: x / max(abs_list_noised_35), noise_signal_35))
    noise_signal_65 = list(map(lambda x: x / max(abs_list_noised_65), noise_signal_65))

    # Построение сгенерированных сигналов
    plt.plot(t, noise_signal_10, color='cornflowerblue')
    plt.title('Зашумленный сигнал, 10%')
    plt.ylabel('Амплитуда, В')
    plt.xlabel('Время, с.')
    plt.show()

    plt.plot(t, noise_signal_35, color='darkorange')
    plt.title('Зашумленный сигнал, 35%')
    plt.ylabel('Амплитуда, В')
    plt.xlabel('Время, с.')
    plt.show()

    plt.plot(t, noise_signal_65, color='forestgreen')
    plt.title('Зашумленный сигнал, 65%')
    plt.ylabel('Амплитуда, В')
    plt.xlabel('Время, с.')
    plt.show()

    # Дискретизация сигналов
    disc_noise_signal_10 = signal_receiver.discretize(duration, noise_signal_10)
    disc_noise_signal_35 = signal_receiver.discretize(duration, noise_signal_35)
    disc_noise_signal_65 = signal_receiver.discretize(duration, noise_signal_65)

    # Построение результатов дискретизации сигналов
    plt.plot(t, disc_noise_signal_10, color='cornflowerblue')
    plt.title('Импульсный сигнал АЦП, зашумление 10%')
    plt.ylabel('Квантованные значения сигнала')
    plt.xlabel('Время, с.')
    plt.show()

    plt.plot(t, disc_noise_signal_35, color='darkorange')
    plt.title('Импульсный сигнал АЦП, зашумление 35%')
    plt.ylabel('Квантованные значения сигнала')
    plt.xlabel('Время, с.')
    plt.show()

    plt.plot(t, disc_noise_signal_65, color='forestgreen')
    plt.title('Импульсный сигнал АЦП, зашумление 65%')
    plt.ylabel('Квантованные значения сигнала')
    plt.xlabel('Время, с.')
    plt.show()

    # Перенос частоты несущей, получение исходного сигнала
    signal_10 = signal_receiver.butter_filter(duration, disc_noise_signal_10, freq=383, coeff=1, filter_n=6)
    signal_35 = signal_receiver.butter_filter(duration, disc_noise_signal_35, freq=383, coeff=1, filter_n=6)
    signal_65 = signal_receiver.butter_filter(duration, disc_noise_signal_65, freq=383, coeff=1, filter_n=6)

    # Построение результата первой фильтрации, исходный сигнал
    plt.plot(t, signal_10, label='Зашумление 10%', color='cornflowerblue')
    plt.plot(t, signal_35, label='Зашумление 35%', color='darkorange')
    plt.plot(t, signal_65, label='Зашумление 65%', color='forestgreen')
    plt.title('Результат первого переноса частоты')
    plt.ylabel('Амплитуда, В')
    plt.xlabel('Время, с.')
    plt.show()

    # Второе пропускание сигнала через фильтр Баттерворта
    signal_detection_10 = signal_receiver.butter_filter(duration, signal_10, freq=14, coeff=8, filter_n=6)
    signal_detection_35 = signal_receiver.butter_filter(duration, signal_35, freq=14, coeff=8, filter_n=6)
    signal_detection_65 = signal_receiver.butter_filter(duration, signal_65, freq=14, coeff=8, filter_n=6)

    # Построение результата второй фильтрации
    plt.plot(t, signal_detection_10, label='Зашумление 10%', color='cornflowerblue')
    plt.plot(t, signal_detection_35, label='Зашумление 35%', color='darkorange')
    plt.plot(t, signal_detection_65, label='Зашумление 65%',  color='forestgreen')
    plt.plot(t, np.full((duration, 1), lower_bound), label='Порог фильтра 6-го порядка')
    plt.title('Результаты второго переноса частоты')
    plt.ylabel('Амплитуда, В')
    plt.xlabel('Время, с.')
    plt.legend()
    plt.show()

    # Определение наличия сигнала
    signal_presence_10 = signal_receiver.determine_signal_presence(duration, signal_detection_10, lower_bound)
    signal_presence_35 = signal_receiver.determine_signal_presence(duration, signal_detection_35, lower_bound)
    signal_presence_65 = signal_receiver.determine_signal_presence(duration, signal_detection_65, lower_bound)

    # Построение графика присутствия сигнала
    plt.plot(t, signal_presence_10, '*', color='cornflowerblue')
    plt.title('Результаты определения наличия сигнала для зашумления в 10%')
    plt.ylabel('Наличие сигнала (0 - нет, 1 - есть')
    plt.xlabel('Время, с.')
    plt.show()

    plt.plot(t, signal_presence_35, '*', color='darkorange')
    plt.title('Результаты определения наличия сигнала для зашумления в 35%')
    plt.ylabel('Наличие сигнала (0 - нет, 1 - есть')
    plt.xlabel('Время, с.')
    plt.show()

    plt.plot(t, signal_presence_65, '*', color='forestgreen')
    plt.title('Результаты определения наличия сигнала для зашумления в 35%')
    plt.ylabel('Наличие сигнала (0 - нет, 1 - есть')
    plt.xlabel('Время, с.')
    plt.show()

    # Определение задержки определителя
    signal_receiver.determine_delay(signal_presence_10, t)
    signal_receiver.determine_delay(signal_presence_35, t)
    signal_receiver.determine_delay(signal_presence_65, t)

if __name__ == '__main__':
    main()
