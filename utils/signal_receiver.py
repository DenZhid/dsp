import math

import numpy as np
from scipy.signal import butter, lfilter


class SignalReceiver:

    def __init__(self, freq_n, freq_m, phase_n, phase_m, modulation_factor, quantization, k_disc, t_end):
        self.freq_n = freq_n
        self.freq_m = freq_m
        self.phase_n = phase_n
        self.phase_m = phase_m
        self.modulation_factor = modulation_factor
        self.quantization = quantization
        self.k_disc = k_disc
        self.t_end = t_end
        self.freq_d = freq_n * k_disc
        self.t_d = 1 / self.freq_d

    def generate_signal(self):
        mod_sig = []
        t = np.arange(0, self.t_end, self.t_d).tolist()
        n = len(t)
        for i in range(n):
            sin_m_i = math.sin(2 * math.pi * self.freq_m * t[i] + self.phase_m)
            sin_n_i = math.sin(2 * math.pi * self.freq_n * t[i] + self.phase_n)
            mod_sig.append(0.5 * (1 + self.modulation_factor * sin_m_i) * sin_n_i)
        return t, mod_sig

    def discretize(self, n, mod_sig):
        disc_mod_sig = []
        for i in range(n):
            disc_mod_sig_i = math.floor(
                (mod_sig[i] + self.modulation_factor) * self.quantization / (2 * self.modulation_factor)
            )
            disc_mod_sig.append(disc_mod_sig_i)
        return disc_mod_sig

    def butter_filter(self, n, disc_mod_sig, freq, coeff):
        detection = []
        sin_out = []
        cos_out = []
        for i in range(n):
            sin_out.append(disc_mod_sig[i] * math.sin((i - 1) * 2 * math.pi * freq / self.freq_d))
            cos_out.append(disc_mod_sig[i] * math.cos((i - 1) * 2 * math.pi * freq / self.freq_d))
        [b, a] = butter(2, self.freq_m * self.k_disc / (2 * coeff * self.freq_d))
        sin_out_butter = lfilter(b, a, sin_out)
        cos_out_butter = lfilter(b, a, cos_out)
        for i in range(n):
            detection_i = math.sqrt(math.pow(sin_out_butter[i], 2) + math.pow(cos_out_butter[i], 2))
            detection.append(detection_i)
        return detection
