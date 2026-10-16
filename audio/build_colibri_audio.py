# Builds the Colibri ad soundtrack: French voice-over (Kokoro, ff_siwis), a light
# music bed and sound effects synced to colibri-pub.html, then a ducked mix.
import json
import os
import numpy as np
import soundfile as sf
from scipy import signal
from kokoro_onnx import Kokoro

SR = 48000
DUR = 12.0
N = int(SR * DUR)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '')
# Models (Apache-2.0), not committed: put them next to this script
#   https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx
#   https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin
# pip install kokoro-onnx soundfile scipy ; then python build_colibri_audio.py
# Loudness: ffmpeg loudnorm to -14 LUFS / -1 dBTP on mix_raw.wav gives colibri-pub-mix.wav
rng = np.random.default_rng(7)

# ---------------------------------------------------------------- voice-over
kok = Kokoro(OUT + 'kokoro-v1.0.onnx', OUT + 'voices-v1.0.bin')


def tts(text, speed):
    s, sr = kok.create(text, voice='ff_siwis', speed=speed, lang='fr-fr')
    s = signal.resample_poly(s.astype(np.float64), SR, sr)
    env = np.abs(s)
    thr = 10 ** (-42 / 20) * env.max()
    idx = np.where(env > thr)[0]
    a, b = max(0, idx[0] - int(0.02 * SR)), min(len(s), idx[-1] + int(0.06 * SR))
    s = s[a:b]
    fade = int(0.01 * SR)
    s[:fade] *= np.linspace(0, 1, fade)
    s[-fade:] *= np.linspace(1, 0, fade)
    return s


LINES = [
    # (start s, text, speed)
    (0.35, 'Un colis à envoyer ?', 1.0),
    (2.10, 'Pensez Colibri !', 0.95),
    (4.15, 'Un clic, un livreur !', 0.95),
    (5.60, 'Des livreurs près de vous,', 1.2),
    ('after', 'un suivi en temps réel,', 1.2),
    ('after', 'et un paiement sécurisé.', 1.2),
    (9.35, "Téléchargez Colibri sur l'App Store et Gougueule Plé !", 1.12),
]
vo = np.zeros(N)
vo_mask = np.zeros(N)
timing = []
cursor = 0.0
for start, text, speed in LINES:
    s = tts(text, speed)
    t0 = cursor + 0.1 if start == 'after' else start
    if t0 < cursor + 0.08:
        raise SystemExit(f'line overlaps the previous one: {text} at {t0:.2f}s, previous ends {cursor:.2f}s')
    i0 = int(t0 * SR)
    if i0 + len(s) > N:
        raise SystemExit(f'line overruns the video: {text} ends at {(i0 + len(s)) / SR:.2f}s')
    vo[i0:i0 + len(s)] += s
    vo_mask[i0:i0 + len(s)] = 1
    cursor = (i0 + len(s)) / SR
    timing.append((round(t0, 3), round(cursor, 3), text))
for t in timing:
    print(f'{t[0]:6.2f} - {t[1]:6.2f}  {t[2]}')


def biquad_peak(f0, gain_db, q):
    A = 10 ** (gain_db / 40)
    w = 2 * np.pi * f0 / SR
    al = np.sin(w) / (2 * q)
    b = [1 + al * A, -2 * np.cos(w), 1 - al * A]
    a = [1 + al / A, -2 * np.cos(w), 1 - al / A]
    return np.array(b) / a[0], np.array(a) / a[0]


def compress(x, thr_db=-20, ratio=3.0, att=0.004, rel=0.12):
    env = np.zeros_like(x)
    ga, gr = np.exp(-1 / (att * SR)), np.exp(-1 / (rel * SR))
    e = 0.0
    ax = np.abs(x)
    for i in range(len(x)):
        c = ga if ax[i] > e else gr
        e = c * e + (1 - c) * ax[i]
        env[i] = e
    lvl = 20 * np.log10(env + 1e-9)
    over = np.maximum(lvl - thr_db, 0)
    gain = 10 ** (-over * (1 - 1 / ratio) / 20)
    return x * gain


# voice chain: high-pass, presence lift, gentle compression
vo = signal.sosfilt(signal.butter(2, 90, 'hp', fs=SR, output='sos'), vo)
b, a = biquad_peak(3500, 2.5, 0.8)
vo = signal.lfilter(b, a, vo)
b, a = biquad_peak(250, -1.5, 1.0)
vo = signal.lfilter(b, a, vo)
vo = vo / (np.abs(vo).max() + 1e-9) * 0.7
vo = compress(vo, -18, 3.0)
vo = vo / (np.abs(vo).max() + 1e-9) * 0.8

# ---------------------------------------------------------------- music + sfx
t = np.arange(N) / SR
L = np.zeros(N)
R = np.zeros(N)


def add(sig, at, gain=1.0, pan=0.0):
    i0 = int(at * SR)
    n = min(len(sig), N - i0)
    if n <= 0:
        return
    l, r = np.cos((pan + 1) * np.pi / 4), np.sin((pan + 1) * np.pi / 4)
    L[i0:i0 + n] += sig[:n] * gain * l * 1.4142
    R[i0:i0 + n] += sig[:n] * gain * r * 1.4142


def env_adsr(n, a, d, s, r, total):
    e = np.zeros(n)
    na, nd, nr = int(a * SR), int(d * SR), int(r * SR)
    nt = int(total * SR)
    e[:na] = np.linspace(0, 1, na, endpoint=False)
    e[na:na + nd] = np.linspace(1, s, nd, endpoint=False)
    e[na + nd:nt] = s
    if nt + nr <= n:
        e[nt:nt + nr] = np.linspace(e[nt - 1] if nt > 0 else s, 0, nr)
    return e


def saw(f, n, detune=0.0, harm=14):
    tt = np.arange(n) / SR
    out = np.zeros(n)
    ff = f * 2 ** (detune / 1200)
    for k in range(1, harm + 1):
        if ff * k > 9000:
            break
        out += np.sin(2 * np.pi * ff * k * tt + rng.uniform(0, 6.28)) / k
    return out * 0.6


def note(m):
    return 440.0 * 2 ** ((m - 69) / 12)


# chords per 2 s bar, in C major
CH = [(0, [60, 64, 67, 71, 74]), (2, [57, 60, 64, 67]), (4, [53, 57, 60, 64]),
      (6, [55, 59, 62, 67]), (8, [60, 64, 67, 71]), (10, [53, 57, 60, 65]),
      (11, [55, 59, 62, 67]), (11.5, [60, 64, 67, 72])]
pad = np.zeros((2, N))
for idx, (start, notes) in enumerate(CH):
    end = CH[idx + 1][0] if idx + 1 < len(CH) else DUR
    dur = end - start
    n = int((dur + 0.5) * SR)
    e = env_adsr(n, 0.35 if idx == 0 else 0.12, 0.3, 0.8, 0.45, dur)
    for m in notes:
        for ch, det in ((0, -7), (1, 7)):
            sig = saw(note(m), n, det, 10) * e
            i0 = int(start * SR)
            k = min(n, N - i0)
            pad[ch, i0:i0 + k] += sig[:k]
lp = signal.butter(2, 1600, 'lp', fs=SR, output='sos')
pad = signal.sosfilt(lp, pad, axis=1) * 0.07
swell = np.clip(t / 2.3, 0, 1) ** 2 * 0.6 + 0.4 * (t >= 2.3)
pad *= swell
L += pad[0]
R += pad[1]

# bass on 8ths from 2.3 s, root of the chord, pumped
BEAT = 0.5


def chord_at(x):
    cur = CH[0][1]
    for s0, ns in CH:
        if x >= s0:
            cur = ns
    return cur


for k in range(int(2.5 / (BEAT / 2)), int(DUR / (BEAT / 2))):
    x = k * BEAT / 2
    root = chord_at(x)[0] - 24
    n = int(0.24 * SR)
    tt = np.arange(n) / SR
    sig = (np.sin(2 * np.pi * note(root) * tt) + 0.25 * np.sin(4 * np.pi * note(root) * tt)) * np.exp(-tt * 6)
    sig *= np.minimum(1, tt / 0.01)
    add(np.tanh(sig * 1.6) * 0.22, x)

# drums
def kick():
    n = int(0.45 * SR)
    tt = np.arange(n) / SR
    f = 50 + 110 * np.exp(-tt * 28)
    ph = 2 * np.pi * np.cumsum(f) / SR
    s = np.sin(ph) * np.exp(-tt * 7.5)
    s[:int(0.002 * SR)] += rng.uniform(-0.5, 0.5, int(0.002 * SR))
    return s


def hat(dec=0.035):
    n = int(0.12 * SR)
    tt = np.arange(n) / SR
    s = signal.sosfilt(signal.butter(4, 7000, 'hp', fs=SR, output='sos'), rng.standard_normal(n))
    return s * np.exp(-tt / dec)


def clap():
    n = int(0.3 * SR)
    tt = np.arange(n) / SR
    s = signal.sosfilt(signal.butter(2, [900, 3200], 'bp', fs=SR, output='sos'), rng.standard_normal(n))
    e = np.exp(-tt / 0.09)
    for d in (0.0, 0.011, 0.022):
        e += np.where(tt >= d, np.exp(-(tt - d) / 0.006), 0) * 0.8
    return s * e * 0.5


KICK, CLAP = kick(), clap()
for k in range(int(DUR / BEAT)):
    x = k * BEAT
    if x < 2.3 or x > 11.55:
        continue
    half = x < 4.0
    if not half or k % 2 == 0:
        add(KICK, x, 0.55)
    add(hat(), x + BEAT / 2, 0.06, 0.3)
    if x >= 4.0:
        add(hat(0.02), x + BEAT / 4, 0.03, -0.3)
        add(hat(0.02), x + 3 * BEAT / 4, 0.03, -0.3)
    if x >= 6.0 and k % 2 == 1:
        add(CLAP, x, 0.32, 0.1)

# plucky arpeggio from 4 s
for k in range(int(4.0 / (BEAT / 2)), int(11.5 / (BEAT / 2))):
    x = k * BEAT / 2
    ns = chord_at(x)
    m = ns[k % len(ns)] + 12
    n = int(0.35 * SR)
    tt = np.arange(n) / SR
    f = note(m)
    s = (np.sin(2 * np.pi * f * tt) + 0.3 * np.sin(4 * np.pi * f * tt) + 0.1 * np.sin(6 * np.pi * f * tt)) * np.exp(-tt * 11)
    s *= np.minimum(1, tt / 0.003)
    pan = 0.35 if k % 2 else -0.35
    add(s * 0.05, x, 1.0, pan)
    add(s * 0.02, x + 0.375, 1.0, -pan)     # dotted-8th echo


# ---------------------------------------------------------------- sound effects
def noise_sweep(dur, f0, f1, shape, q=2.0):
    n = int(dur * SR)
    x = rng.standard_normal(n)
    out = np.zeros(n)
    blk = 256
    zi = None
    for i in range(0, n, blk):
        u = i / n
        fc = f0 * (f1 / f0) ** u
        lo, hi = fc / (1 + 1 / q), min(fc * (1 + 1 / q), SR / 2 - 100)
        sos = signal.butter(2, [lo, hi], 'bp', fs=SR, output='sos')
        if zi is None:
            zi = signal.sosfilt_zi(sos) * 0
        out[i:i + blk], zi = signal.sosfilt(sos, x[i:i + blk], zi=zi)
    return out * shape(np.linspace(0, 1, n))


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


def boom(dur=1.3):
    n = int(dur * SR)
    tt = np.arange(n) / SR
    f = 38 + 40 * np.exp(-tt * 9)
    s = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-tt * 3.2)
    nz = signal.sosfilt(signal.butter(2, 400, 'lp', fs=SR, output='sos'), rng.standard_normal(n)) * np.exp(-tt * 18) * 0.6
    return s + nz


# droplets fly in, merge, liquid morph, impact as the logo locks
add(noise_sweep(0.6, 3000, 900, lambda u: np.sin(np.pi * u) ** 2), 0.25, 0.08, -0.6)
add(noise_sweep(0.6, 3000, 900, lambda u: np.sin(np.pi * u) ** 2), 0.38, 0.07, 0.6)
add(noise_sweep(0.6, 2500, 800, lambda u: np.sin(np.pi * u) ** 2), 0.5, 0.06, 0.1)
for at, f0 in ((0.92, 240), (1.02, 300), (1.12, 360)):
    add(bloop(f0, f0 * 2.6), at, 0.32)
add(noise_sweep(1.0, 500, 5500, lambda u: u ** 2 * (1 - u) ** 0.3), 1.3, 0.12)
add(boom(), 2.28, 0.6)
add(bell(note(84)), 2.3, 0.10, -0.2)
add(bell(note(91)), 2.33, 0.07, 0.3)
for i in range(7):
    add(bell(note(96 + [0, 4, 7, 12, 7, 4, 12][i]), 0.6), 2.4 + i * 0.09, 0.025, rng.uniform(-0.8, 0.8))
# logo glint shimmer
add(noise_sweep(0.9, 4000, 12000, lambda u: np.sin(np.pi * u) ** 2, 3.0), 2.65, 0.035, -0.4)
# wordmark reveal swish and tagline
add(noise_sweep(0.5, 1200, 6000, lambda u: np.sin(np.pi * u)), 3.8, 0.05, -0.5)
# transition to header + mascot fly-in from the right
add(noise_sweep(0.55, 800, 3500, lambda u: np.sin(np.pi * u) ** 2), 5.55, 0.07, 0.0)
wh = noise_sweep(0.95, 600, 2800, lambda u: np.sin(np.pi * np.clip(u * 1.1, 0, 1)) ** 1.5, 1.5)
add(wh, 5.85, 0.14, 0.7)
add(boom(0.5) * 0.6, 6.72, 0.45)
# feature chips
for at, m in ((6.75, 84), (7.1, 88), (8.25, 91)):
    add(pop(note(m) * 0.6, note(m)), at, 0.22)
    add(bell(note(m + 12), 0.5), at + 0.01, 0.03)
# end card
add(noise_sweep(0.9, 600, 7000, lambda u: u ** 2), 8.45, 0.07)
add(boom(0.9), 9.3, 0.35)
for at, m in ((9.9, 79), (10.05, 84)):
    add(pop(note(m) * 0.6, note(m)), at, 0.2)
for at, pan in ((10.6, -0.4), (10.75, 0.4)):
    add(noise_sweep(0.5, 5000, 12000, lambda u: np.sin(np.pi * u) ** 2, 3.0), at, 0.03, pan)
for at, m in ((11.25, 91), (11.4, 96)):
    add(bell(note(m), 0.9), at, 0.06)
add(bell(note(72), 1.2), 11.5, 0.08)
add(boom(0.8), 11.5, 0.3)

# a touch of space on the bed
ir_n = int(1.3 * SR)
ir = rng.standard_normal((2, ir_n)) * np.exp(-np.arange(ir_n) / SR * 4.5)
ir = signal.sosfilt(signal.butter(1, 5000, 'lp', fs=SR, output='sos'), ir, axis=1)
ir /= np.sqrt((ir ** 2).sum(axis=1, keepdims=True))
wetL = signal.fftconvolve(L, ir[0])[:N]
wetR = signal.fftconvolve(R, ir[1])[:N]
L, R = L + 0.18 * wetL, R + 0.18 * wetR

# final tail fade
fo = int(0.25 * SR)
for ch in (L, R):
    ch[-fo:] *= np.linspace(1, 0, fo) ** 2

music = np.stack([L, R])
music /= np.abs(music).max() + 1e-9

# ---------------------------------------------------------------- ducking + mix
gr = np.convolve(vo_mask, np.ones(int(0.08 * SR)) / int(0.08 * SR), 'same')
duck = 1 - 0.6 * np.clip(gr, 0, 1)                   # about -8 dB under the voice
duck = signal.sosfiltfilt(signal.butter(1, 6, 'lp', fs=SR, output='sos'), duck)
mix = music * 0.45 * duck + np.stack([vo, vo]) * 1.0
mix = np.tanh(mix * 1.1) / np.tanh(1.1)
sf.write(OUT + 'vo.wav', np.stack([vo, vo]).T, SR, subtype='PCM_24')
sf.write(OUT + 'music.wav', (music * 0.5).T, SR, subtype='PCM_24')
sf.write(OUT + 'mix_raw.wav', (mix * 0.8).T, SR, subtype='PCM_24')
json.dump(timing, open(OUT + 'vo_timing.json', 'w'), ensure_ascii=False, indent=1)
print('peak mix', np.abs(mix).max())
