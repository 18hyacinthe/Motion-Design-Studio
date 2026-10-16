# Small synthesis kit shared by the Colibri soundtracks: Kokoro French voice,
# voice chain, drums, pads, plucks and sound effects, all generated in numpy.
import os
import numpy as np
from scipy import signal

SR = 48000
HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(7)
_kokoro = None


def note(m):
    return 440.0 * 2 ** ((m - 69) / 12)


def tts(text, speed):
    global _kokoro
    if _kokoro is None:
        from kokoro_onnx import Kokoro
        _kokoro = Kokoro(os.path.join(HERE, 'kokoro-v1.0.onnx'), os.path.join(HERE, 'voices-v1.0.bin'))
    s, sr = _kokoro.create(text, voice='ff_siwis', speed=speed, lang='fr-fr')
    s = signal.resample_poly(s.astype(np.float64), SR, sr)
    env = np.abs(s)
    idx = np.where(env > 10 ** (-42 / 20) * env.max())[0]
    s = s[max(0, idx[0] - int(0.02 * SR)):min(len(s), idx[-1] + int(0.06 * SR))]
    f = int(0.01 * SR)
    s[:f] *= np.linspace(0, 1, f)
    s[-f:] *= np.linspace(1, 0, f)
    return s


def biquad_peak(f0, gain_db, q):
    A = 10 ** (gain_db / 40)
    w = 2 * np.pi * f0 / SR
    al = np.sin(w) / (2 * q)
    b = np.array([1 + al * A, -2 * np.cos(w), 1 - al * A])
    a = np.array([1 + al / A, -2 * np.cos(w), 1 - al / A])
    return b / a[0], a / a[0]


def compress(x, thr_db=-18, ratio=3.0, att=0.004, rel=0.12):
    ga, gr = np.exp(-1 / (att * SR)), np.exp(-1 / (rel * SR))
    env = np.empty_like(x)
    e = 0.0
    ax = np.abs(x)
    for i in range(len(x)):
        c = ga if ax[i] > e else gr
        e = c * e + (1 - c) * ax[i]
        env[i] = e
    over = np.maximum(20 * np.log10(env + 1e-9) - thr_db, 0)
    return x * 10 ** (-over * (1 - 1 / ratio) / 20)


def voice_chain(vo):
    vo = signal.sosfilt(signal.butter(2, 90, 'hp', fs=SR, output='sos'), vo)
    for f0, g, q in ((3500, 2.5, 0.8), (250, -1.5, 1.0)):
        b, a = biquad_peak(f0, g, q)
        vo = signal.lfilter(b, a, vo)
    vo = vo / (np.abs(vo).max() + 1e-9) * 0.7
    vo = compress(vo, -18, 3.0)
    return vo / (np.abs(vo).max() + 1e-9) * 0.8


def place_lines(lines, n):
    """lines: (start | 'after', text, speed). Returns voice, mask, timing."""
    vo, mask, timing, cursor = np.zeros(n), np.zeros(n), [], 0.0
    for start, text, speed in lines:
        s = tts(text, speed)
        t0 = cursor + 0.15 if start == 'after' else start
        if t0 < cursor + 0.08:
            raise SystemExit(f'line overlaps the previous one: {text} at {t0:.2f}s, previous ends {cursor:.2f}s')
        i0 = int(t0 * SR)
        if i0 + len(s) > n:
            raise SystemExit(f'line overruns: {text} ends at {(i0 + len(s)) / SR:.2f}s')
        vo[i0:i0 + len(s)] += s
        mask[i0:i0 + len(s)] = 1
        cursor = (i0 + len(s)) / SR
        timing.append((round(t0, 3), round(cursor, 3), text))
    return vo, mask, timing


class Bus:
    def __init__(self, dur):
        self.n = int(SR * dur)
        self.L = np.zeros(self.n)
        self.R = np.zeros(self.n)

    def add(self, sig, at, gain=1.0, pan=0.0):
        i0 = int(at * SR)
        k = min(len(sig), self.n - i0)
        if k <= 0 or i0 < 0:
            return
        l, r = np.cos((pan + 1) * np.pi / 4), np.sin((pan + 1) * np.pi / 4)
        self.L[i0:i0 + k] += sig[:k] * gain * l * 1.4142
        self.R[i0:i0 + k] += sig[:k] * gain * r * 1.4142


def saw(f, n, detune=0.0, harm=10):
    tt = np.arange(n) / SR
    out = np.zeros(n)
    ff = f * 2 ** (detune / 1200)
    for k in range(1, harm + 1):
        if ff * k > 9000:
            break
        out += np.sin(2 * np.pi * ff * k * tt + rng.uniform(0, 6.28)) / k
    return out * 0.6


def adsr(n, a, d, s, r, total):
    e = np.zeros(n)
    na, nd, nr, nt = int(a * SR), int(d * SR), int(r * SR), int(total * SR)
    e[:na] = np.linspace(0, 1, na, endpoint=False)
    e[na:na + nd] = np.linspace(1, s, nd, endpoint=False)
    e[na + nd:nt] = s
    if nt + nr <= n:
        e[nt:nt + nr] = np.linspace(s, 0, nr)
    return e


def pad(bus, chords, dur, gain=0.07, cutoff=1600):
    p = np.zeros((2, bus.n))
    for i, (start, notes) in enumerate(chords):
        end = chords[i + 1][0] if i + 1 < len(chords) else dur
        n = int((end - start + 0.5) * SR)
        e = adsr(n, 0.35 if i == 0 else 0.12, 0.3, 0.8, 0.45, end - start)
        i0 = int(start * SR)
        k = min(n, bus.n - i0)
        for m in notes:
            for ch, det in ((0, -7), (1, 7)):
                p[ch, i0:i0 + k] += (saw(note(m), n, det) * e)[:k]
    p = signal.sosfilt(signal.butter(2, cutoff, 'lp', fs=SR, output='sos'), p, axis=1) * gain
    return p


def chord_at(chords, x):
    cur = chords[0][1]
    for s0, ns in chords:
        if x >= s0:
            cur = ns
    return cur


def bass(bus, chords, t0, t1, step=0.25, gain=0.22):
    for k in range(int(t0 / step), int(t1 / step)):
        x = k * step
        root = chord_at(chords, x)[0] - 24
        n = int(0.24 * SR)
        tt = np.arange(n) / SR
        s = (np.sin(2 * np.pi * note(root) * tt) + 0.25 * np.sin(4 * np.pi * note(root) * tt)) * np.exp(-tt * 6)
        bus.add(np.tanh(s * np.minimum(1, tt / 0.01) * 1.6) * gain, x)


def kick():
    n = int(0.45 * SR)
    tt = np.arange(n) / SR
    s = np.sin(2 * np.pi * np.cumsum(50 + 110 * np.exp(-tt * 28)) / SR) * np.exp(-tt * 7.5)
    s[:int(0.002 * SR)] += rng.uniform(-0.5, 0.5, int(0.002 * SR))
    return s


def hat(dec=0.035):
    n = int(0.12 * SR)
    tt = np.arange(n) / SR
    return signal.sosfilt(signal.butter(4, 7000, 'hp', fs=SR, output='sos'), rng.standard_normal(n)) * np.exp(-tt / dec)


def clap():
    n = int(0.3 * SR)
    tt = np.arange(n) / SR
    s = signal.sosfilt(signal.butter(2, [900, 3200], 'bp', fs=SR, output='sos'), rng.standard_normal(n))
    e = np.exp(-tt / 0.09)
    for d in (0.0, 0.011, 0.022):
        e += np.where(tt >= d, np.exp(-(tt - d) / 0.006), 0) * 0.8
    return s * e * 0.5


def pluck(f, dur=0.35, dec=11):
    n = int(dur * SR)
    tt = np.arange(n) / SR
    return (np.sin(2 * np.pi * f * tt) + 0.3 * np.sin(4 * np.pi * f * tt) + 0.1 * np.sin(6 * np.pi * f * tt)) * np.exp(-tt * dec) * np.minimum(1, tt / 0.003)


def noise_sweep(dur, f0, f1, shape, q=2.0):
    n = int(dur * SR)
    x = rng.standard_normal(n)
    out = np.zeros(n)
    zi = None
    for i in range(0, n, 256):
        fc = f0 * (f1 / f0) ** (i / n)
        sos = signal.butter(2, [fc / (1 + 1 / q), min(fc * (1 + 1 / q), SR / 2 - 100)], 'bp', fs=SR, output='sos')
        if zi is None:
            zi = signal.sosfilt_zi(sos) * 0
        out[i:i + 256], zi = signal.sosfilt(sos, x[i:i + 256], zi=zi)
    return out * shape(np.linspace(0, 1, n))


def whoosh(dur=0.6, f0=700, f1=3500):
    return noise_sweep(dur, f0, f1, lambda u: np.sin(np.pi * u) ** 2, 1.6)


def bloop(f0=260, f1=760, dur=0.16):
    n = int(dur * SR)
    tt = np.arange(n) / SR
    f = f0 + (f1 - f0) * (1 - np.exp(-tt * 40))
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-tt * 22) * np.minimum(1, tt / 0.004)


def bell(f, dur=1.6):
    n = int(dur * SR)
    tt = np.arange(n) / SR
    s = np.zeros(n)
    for p, a, d in ((1, 1, 2.2), (2.0, 0.45, 3.5), (3.0, 0.25, 5), (4.16, 0.18, 7), (5.43, 0.1, 9)):
        s += a * np.sin(2 * np.pi * f * p * tt) * np.exp(-tt * d)
    return s * np.minimum(1, tt / 0.002)


def pop(f0=700, f1=1300):
    n = int(0.09 * SR)
    tt = np.arange(n) / SR
    f = f0 + (f1 - f0) * np.minimum(1, tt / 0.03)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-tt * 45) * np.minimum(1, tt / 0.002)


def tick():
    n = int(0.03 * SR)
    tt = np.arange(n) / SR
    s = signal.sosfilt(signal.butter(2, [2500, 6000], 'bp', fs=SR, output='sos'), rng.standard_normal(n))
    return s * np.exp(-tt * 260)


def tap():
    n = int(0.08 * SR)
    tt = np.arange(n) / SR
    return np.sin(2 * np.pi * np.cumsum(1400 - 600 * np.minimum(1, tt / 0.04)) / SR) * np.exp(-tt * 70) * 0.8


def boom(dur=1.3):
    n = int(dur * SR)
    tt = np.arange(n) / SR
    s = np.sin(2 * np.pi * np.cumsum(38 + 40 * np.exp(-tt * 9)) / SR) * np.exp(-tt * 3.2)
    nz = signal.sosfilt(signal.butter(2, 400, 'lp', fs=SR, output='sos'), rng.standard_normal(n)) * np.exp(-tt * 18) * 0.6
    return s + nz


def success(f=note(84)):
    a, b = bell(f, 1.0), bell(f * 1.5, 1.2)
    out = np.zeros(len(b) + int(0.09 * SR))
    out[:len(a)] += a
    out[int(0.09 * SR):int(0.09 * SR) + len(b)] += b
    return out


def scooter(dur=0.9):
    n = int(dur * SR)
    tt = np.arange(n) / SR
    u = tt / dur
    f = 90 + 70 * u + 20 * np.sin(2 * np.pi * 9 * tt)
    s = np.sign(np.sin(2 * np.pi * np.cumsum(f) / SR)) * 0.5
    s = signal.sosfilt(signal.butter(2, 900, 'lp', fs=SR, output='sos'), s)
    return s * np.sin(np.pi * u) ** 1.5


def reverb(bus, amount=0.18, length=1.3):
    n = int(length * SR)
    ir = rng.standard_normal((2, n)) * np.exp(-np.arange(n) / SR * 4.5)
    ir = signal.sosfilt(signal.butter(1, 5000, 'lp', fs=SR, output='sos'), ir, axis=1)
    ir /= np.sqrt((ir ** 2).sum(axis=1, keepdims=True))
    bus.L = bus.L + amount * signal.fftconvolve(bus.L, ir[0])[:bus.n]
    bus.R = bus.R + amount * signal.fftconvolve(bus.R, ir[1])[:bus.n]


def duck_mix(music, vo, mask, music_gain=0.45, depth=0.6):
    k = int(0.08 * SR)
    gr = np.convolve(mask, np.ones(k) / k, 'same')
    duck = signal.sosfiltfilt(signal.butter(1, 6, 'lp', fs=SR, output='sos'), 1 - depth * np.clip(gr, 0, 1))
    mix = music * music_gain * duck + np.stack([vo, vo])
    return np.tanh(mix * 1.1) / np.tanh(1.1)
