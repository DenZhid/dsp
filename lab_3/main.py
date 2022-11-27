import numpy as np

from utils.signal_receiver import SignalReceiver
import matplotlib.pyplot as plt


# Variant 61: Fn = 383 Fm = 14 K = 128
def main():
    N_LIM = 500 # ограничение количества точек на графиках

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
    t, gen_signal_2 = signal_receiver.generate_signal_with_duty_cycle(2) # Скважность Q = 2
    _, gen_signal_4 = signal_receiver.generate_signal_with_duty_cycle(4) # Скважность Q = 4
    _, gen_signal_6 = signal_receiver.generate_signal_with_duty_cycle(6) # Скважность Q = 6
    duration = len(t)

    # Настройка разрешения графиков
    plt.rcParams['figure.figsize'] = [12, 8]
    plt.rcParams['figure.dpi'] = 100

    # Построение сгенерированных сигналов
    plt.plot(t[:N_LIM], gen_signal_2[:N_LIM], label='Скважность Q = 2', color='cornflowerblue')
    plt.title('Модулированный сигнал')
    plt.ylabel('Амплитуда, В')
    plt.xlabel('Время, с.')
    plt.legend()
    plt.show()

    plt.plot(t[:N_LIM], gen_signal_4[:N_LIM], label='Скважность Q = 4', color='darkorange')
    plt.title('Модулированный сигнал')
    plt.ylabel('Амплитуда, В')
    plt.xlabel('Время, с.')
    plt.legend()
    plt.show()

    plt.plot(t[:N_LIM], gen_signal_6[:N_LIM], label='Скважность Q = 6', color='forestgreen')
    plt.title('Модулированный сигнал')
    plt.ylabel('Амплитуда, В')
    plt.xlabel('Время, с.')
    plt.legend()
    plt.show()

    # Дискретизация сигнала
    disc_signal_2 = signal_receiver.discretize(duration, gen_signal_2)
    disc_signal_4 = signal_receiver.discretize(duration, gen_signal_4)
    disc_signal_6 = signal_receiver.discretize(duration, gen_signal_6)

    # Построение результата дискретизации сигнала
    plt.plot(t[:N_LIM], disc_signal_2[:N_LIM], label='Скважность Q = 2', color='cornflowerblue')
    plt.title("Дискретизированный сигнал")
    plt.ylabel("Уровень сигнала")
    plt.xlabel("Время, с")
    plt.legend()
    plt.show()

    plt.plot(t[:N_LIM], disc_signal_4[:N_LIM], label='Скважность Q = 4', color='darkorange')
    plt.title("Дискретизированный сигнал")
    plt.ylabel("Уровень сигнала")
    plt.xlabel("Время, с")
    plt.legend()
    plt.show()

    plt.plot(t[:N_LIM], disc_signal_6[:N_LIM], label='Скважность Q = 6', color='forestgreen')
    plt.title("Дискретизированный сигнал")
    plt.ylabel("Уровень сигнала")
    plt.xlabel("Время, с")
    plt.legend()
    plt.show()

    # Перенос частоты несущей, получение исходного сигнала
    signal_2 = signal_receiver.butter_filter(duration, disc_signal_2, freq=360, coeff=1, filter_n=6)
    signal_4 = signal_receiver.butter_filter(duration, disc_signal_4, freq=360, coeff=1, filter_n=6)
    signal_6 = signal_receiver.butter_filter(duration, disc_signal_6, freq=360, coeff=1, filter_n=6)


    # Построение результата первой фильтрации, исходный сигнал
    plt.plot(t, signal_2, label='Скважность Q = 2', color='cornflowerblue')
    plt.plot(t, signal_4, label='Скважность Q = 4', color='darkorange')
    plt.plot(t, signal_6, label='Скважность Q = 6', color='forestgreen')
    plt.title('Результат первого переноса частоты')
    plt.ylabel('Уровень сигнала')
    plt.xlabel('Время, с.')
    plt.legend()
    plt.show()

    # Второе пропускание сигнала через фильтр Баттерворта
    signal_detection_2 = signal_receiver.butter_filter(duration, signal_2, freq=10, coeff=8, filter_n=6)
    signal_detection_4 = signal_receiver.butter_filter(duration, signal_4, freq=10, coeff=8, filter_n=6)
    signal_detection_6 = signal_receiver.butter_filter(duration, signal_6, freq=10, coeff=8, filter_n=6)

    N2 = len(signal_detection_2[1743:])
    lower_bound_2 = ([sum(signal_detection_2[1743:]) / N2 * 0.708] * len(t))[0]
    lower_bound_4 = ([sum(signal_detection_4[1743:]) / N2 * 0.708] * len(t))[0]
    lower_bound_6 = ([sum(signal_detection_6[1743:]) / N2 * 0.708] * len(t))[0]

    print(lower_bound_2)
    print(lower_bound_4)
    print(lower_bound_6)

    # Построение результата второй фильтрации
    plt.plot(t, np.full((duration, 1), lower_bound_2), label='Пороговое значение для скважности Q = 2')
    plt.plot(t, signal_detection_2, label='Скважность Q = 2', color='cornflowerblue')
    plt.title("Результат второй фильтрации")
    plt.ylabel("Уровень сигнала")
    plt.xlabel("Время, с")
    plt.legend()
    plt.show()

    plt.plot(t, np.full((duration, 1), lower_bound_4), label='Пороговое значение для скважности Q = 4')
    plt.plot(t, signal_detection_4, label='Скважность Q = 4', color='darkorange')
    plt.title("Результат второй фильтрации")
    plt.ylabel("Уровень сигнала")
    plt.xlabel("Время, с")
    plt.legend()
    plt.show()

    plt.plot(t, np.full((duration, 1), lower_bound_6), label='Пороговое значение для скважности Q = 6')
    plt.plot(t, signal_detection_6, label='Скважность Q = 6', color='forestgreen')
    plt.title("Результат второй фильтрации")
    plt.ylabel("Уровень сигнала")
    plt.xlabel("Время, с")
    plt.legend()
    plt.show()

    # Определение наличия сигнала
    signal_presence_2 = signal_receiver.determine_signal_presence(duration, signal_detection_2, lower_bound_2)
    signal_presence_4 = signal_receiver.determine_signal_presence(duration, signal_detection_4, lower_bound_4)
    signal_presence_6 = signal_receiver.determine_signal_presence(duration, signal_detection_2, lower_bound_6)

    # Построение графика присутствия сигнала
    plt.plot(t, signal_presence_2, '*', color='cornflowerblue')
    plt.title('Результаты определения наличия сигнала для скважности Q = 2')
    plt.ylabel('Наличие сигнала (0 - нет, 1 - есть')
    plt.xlabel('Время, с.')
    plt.legend()
    plt.show()

    plt.plot(t, signal_presence_4, '*', color='darkorange')
    plt.title('Результаты определения наличия сигнала для скважности Q = 4')
    plt.ylabel('Наличие сигнала (0 - нет, 1 - есть')
    plt.xlabel('t, с.')
    plt.legend()
    plt.show()

    plt.plot(t, signal_presence_6, '*', color='forestgreen')
    plt.title('Результаты определения наличия сигнала для скважности Q = 6')
    plt.ylabel('Наличие сигнала (0 - нет, 1 - есть')
    plt.xlabel('t, с.')
    plt.legend()
    plt.show()

    # Определение задержки определителя
    signal_receiver.determine_delay(signal_presence_2, t)
    signal_receiver.determine_delay(signal_presence_4, t)
    signal_receiver.determine_delay(signal_presence_6, t)

if __name__ == '__main__':
    main()
