import math

from utils.signal_receiver import SignalReceiver
from utils.fft import custom_fft
from utils.fta import fta
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
    signal = signal_receiver.butter_filter(duration, disc_signal, freq=360, coeff=1)

    # Построение результата первой фильтрации, исходный сигнал
    plt.plot(t, signal)
    plt.title('Результат первого переноса частоты')
    plt.ylabel('А, В')
    plt.xlabel('t, с.')
    plt.show()

    # Второе пропускание сигнала через фильтр Баттерворта
    signal_detection = signal_receiver.butter_filter(duration, signal, freq=10, coeff=8)

    # Построение результата второй фильтрации
    plt.plot(t, signal_detection)
    plt.title('Результаты второго переноса частоты')
    plt.ylabel('A, В')
    plt.xlabel('t, с.')
    plt.show()

    # Определение наличия сигнала
    lower_bound = 10
    signal_presence = [1] * duration
    for i in range(duration):
        if signal_detection[i] <= lower_bound:
            signal_presence[i] = 0
        else:
            break

    # Построение графика присутствия сигнала
    plt.plot(t, signal_presence, '*')
    plt.title('Результаты определения наличия сигнала')
    plt.ylabel('Наличие сигнала (0 - нет, 1 - есть')
    plt.xlabel('t, с.')
    plt.show()

    # Определение задержки определителя
    i = 1
    while signal_presence[i] == 0:
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
