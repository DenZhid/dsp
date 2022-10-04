from utils.signal_receiver import SignalReceiver
import matplotlib.pyplot as plt


def main():
    signal_receiver = SignalReceiver(
        freq_n=360,
        freq_m=10,
        phase_n=0,
        phase_m=0,
        modulation_factor=1,
        quantization=512,
        k_disc=8,
        t_end=1.6
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
