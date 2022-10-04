from matplotlib import pyplot as plt

from utils.signal_receiver import SignalReceiver


# Variant 61: Fn = 383 Fm = 14 K = 128
def main():
    signal_receiver = SignalReceiver(
        freq_n=383, # несущая частота (Fn)
        freq_m=10, # частота модуляции (Fm)
        phase_n=0, # фаза несущей частоты (произвольная)
        phase_m=0, # фаза частоты модуляции (произвольная)
        modulation_factor=1, # глубина модуляции (1)
        quantization=512, # число уровней квантования (K)
        k_disc=8, # отношение частоты дискретизации к частоте несущей
        t_end=2.0 # время окончания модуляции сигнала
    )
    # Генерация сигнала
    t, mod_sig = signal_receiver.generate_signal()
    length = len(t)

    # Дискретизация сигнала
    disc_mod_sig = signal_receiver.discretize(length, mod_sig)

    # Перенос частоты несущей, получение исходного сигнала
    detection = signal_receiver.butter_filter(length, disc_mod_sig, 360, 1)

    # Второе пропускание сигнала через фильтр Баттерворта
    signal_detection = signal_receiver.butter_filter(length, detection, 10, 8)

    # Определение наличия сигнала
    lower_bound = 10
    signal_presence = []
    for i in range(length):
        signal_presence.append(0 if signal_detection[i] <= lower_bound else 1)

    # Построение сигнала
    plt.plot(t, mod_sig)
    plt.title('Модулированный сигнал')
    plt.ylabel('А, В')
    plt.xlabel('t, с.')
    plt.show()

    # Построение результата дискретизации сигнала
    plt.plot(t, disc_mod_sig)
    plt.title('Импульсный сигнал АЦП')
    plt.ylabel('Квантованные значения сигнала')
    plt.xlabel('t, с.')
    plt.show()

    # Построение результата первой фильтрации, исходный сигнал
    plt.plot(t, detection)
    plt.title('Результат первого переноса частоты')
    plt.ylabel('А, В')
    plt.xlabel('t, с.')
    plt.show()

    # Построение результата второй фильтрации
    plt.plot(t, signal_detection)
    plt.title('Результаты второго переноса частоты')
    plt.ylabel('A, В')
    plt.xlabel('t, с.')
    plt.show()

    # Построение графика присутствия сигнала
    plt.plot(t, signal_presence)
    plt.title('Результаты определения наличия сигнала')
    plt.ylabel('Наличие сигнала (0 - нет, 1 - есть')
    plt.xlabel('t, с.')
    plt.show()



if __name__ == '__main__':
    main()