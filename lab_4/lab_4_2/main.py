import numpy as np

from utils.signal_receiver import SignalReceiver
import matplotlib.pyplot as plt


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
    t, gen_signal = signal_receiver.generate_special_signal()
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
    signal = signal_receiver.butter_filter(duration, disc_signal, freq=383, coeff=1, filter_n=6)

    # Построение результата первой фильтрации, исходный сигнал
    plt.plot(t, signal)
    plt.title('Результат первого переноса частоты')
    plt.ylabel('Уровень сигнала')
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
    plt.ylabel('Уровень сигнала')
    plt.xlabel('Время, с.')
    plt.legend()
    plt.show()

    # Определение наличия сигнала
    signal_presence = signal_receiver.determine_signal_presence(duration, signal_detection, lower_bound)

    # Построение графика присутствия сигнала
    plt.plot(t, signal_presence, '*')
    plt.title('Результаты определения наличия сигнала 6-го порядка')
    plt.ylabel('Наличие сигнала (0 - нет, 1 - есть')
    plt.xlabel('Время, с.')
    plt.show()

    # Определение задержки определителя
    signal_receiver.determine_delay(signal_presence, t)

if __name__ == '__main__':
    main()
