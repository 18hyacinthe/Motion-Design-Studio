# Soundtrack for colibri-tuto.html (30 s): same French voice as the first ad,
# a light 120 BPM bed and sound effects on the on-screen events.
# Models (Apache-2.0), not committed, next to this file:
#   https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx
#   https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin
# pip install kokoro-onnx soundfile scipy ; ffmpeg loudnorm to -14 LUFS / -1 dBTP for the final mix.
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import soundfile as sf
from synth import *  # noqa: F401,F403

DUR = 30.0
OUT = HERE + '/'

LINES = [
    (3.0, 'Un colis à envoyer ?', 1.0),
    (5.8, 'Choisissez votre service : colis, pharmacie, achats ou factures.', 1.08),
    (9.6, 'Indiquez le retrait et la livraison, en quelques gestes.', 1.08),
    (13.0, 'Proposez votre prix, et lancez la recherche.', 1.05),
    (16.2, 'Les livreurs proches vous font une offre. Choisissez la meilleure !', 1.1),
    (20.1, 'Suivez votre livreur en temps réel.', 1.05),
    (23.4, 'À la réception, validez : le paiement est sécurisé.', 1.05),
    (26.6, 'Colibri. Un clic, un livreur !', 0.98),
    ('after', "Téléchargez l'application.", 1.0),
]
bus = Bus(DUR)
vo, mask, timing = place_lines(LINES, bus.n)
for a, b, txt in timing:
    print(f'{a:6.2f} - {b:6.2f}  {txt}')
vo = voice_chain(vo)

# ---- music: C Am F G in 2 s bars, resolving on C at 28 s
prog = [[60, 64, 67, 71], [57, 60, 64, 67], [53, 57, 60, 64], [55, 59, 62, 67]]
CH = [(2.0 * k, prog[k % 4]) for k in range(13)] + [(26.0, prog[2]), (27.0, prog[3]), (28.0, [60, 64, 67, 72])]
p = pad(bus, CH, DUR)
p *= np.clip(np.arange(bus.n) / SR / 2.8, 0, 1) ** 1.5 * 0.5 + 0.5
bus.L += p[0]
bus.R += p[1]
bass(bus, CH, 2.75, 28.5)
K, CL = kick(), clap()
for k in range(int(DUR / 0.5)):
    x = k * 0.5
    if x < 2.75 or x > 28.3:
        continue
    bus.add(K, x, 0.5)
    bus.add(hat(), x + 0.25, 0.055, 0.3)
    if x >= 5.5:
        bus.add(hat(0.02), x + 0.125, 0.028, -0.3)
        bus.add(hat(0.02), x + 0.375, 0.028, -0.3)
    if x >= 9.4 and k % 2 == 1:
        bus.add(CL, x, 0.3, 0.1)
for k in range(int(5.6 / 0.25), int(28.0 / 0.25)):
    x = k * 0.25
    ns = chord_at(CH, x)
    s = pluck(note(ns[k % len(ns)] + 12))
    pan = 0.35 if k % 2 else -0.35
    g = 0.045 if x < 26.4 else 0.06
    bus.add(s * g, x, 1.0, pan)
    bus.add(s * g * 0.4, x + 0.375, 1.0, -pan)

# ---- sound effects on the picture
add = bus.add
add(boom(), 0.12, 0.5)
add(bell(note(84)), 0.15, 0.09, -0.2)
add(bell(note(91)), 0.18, 0.06, 0.3)
add(noise_sweep(0.6, 1500, 7000, lambda u: np.sin(np.pi * u)), 0.45, 0.04)
add(noise_sweep(0.9, 4000, 12000, lambda u: np.sin(np.pi * u) ** 2, 3.0), 1.0, 0.03, -0.3)
add(whoosh(0.5, 900, 4000), 1.45, 0.05)
add(scooter(0.95), 1.85, 0.07, 0.0)
add(whoosh(0.9, 500, 2600), 1.85, 0.1, 0.4)
for c in (2.8, 9.4, 12.8, 23.2):                     # panel wipes
    add(whoosh(0.64, 600, 4200), c - 0.32, 0.13)
for c in (5.6, 16.0, 19.8, 26.4):                    # soft scene changes
    add(whoosh(0.5, 900, 3000), c - 0.25, 0.06)
add(pop(note(79) * 0.6, note(79)), 3.25, 0.15)
for k in range(10):
    add(tick(), 4.0 + (k + 1) / 12, 0.12, rng.uniform(-0.2, 0.2))
add(pop(note(84) * 0.6, note(84)), 4.95, 0.15)
for i in range(6):
    add(tick(), 6.0 + i * 0.07, 0.06)
for at, m in ((7.2, 79), (7.75, 81), (8.25, 84), (8.7, 88)):
    add(pop(note(m) * 0.6, note(m)), at, 0.18)
add(tap(), 8.85, 0.16)
for k in range(10):
    add(tick(), 10.2 + (k + 1) / 11, 0.1)
for k in range(5):
    add(tick(), 10.95 + (k + 1) / 10, 0.1)
for at in (11.55, 11.95, 12.45, 14.45, 15.35, 18.65, 25.6):
    add(tap(), at, 0.16)
add(noise_sweep(0.7, 600, 5000, lambda u: (1 - u) ** 0.5 * np.minimum(1, u * 8)), 12.95, 0.08)
add(pop(note(84) * 0.6, note(84)), 14.8, 0.18)
add(bell(note(96), 0.6), 14.8, 0.03)
for at, m in ((16.8, 76), (16.9, 79), (17.18, 81), (17.46, 84)):
    add(pop(note(m) * 0.6, note(m)), at, 0.14)
add(success(), 18.9, 0.09)
for i, at in enumerate((20.3, 21.05, 21.8, 22.55)):
    add(pop(note(76 + i * 3) * 0.6, note(76 + i * 3)), at, 0.13)
    add(bell(note(88 + i * 3), 0.5), at, 0.025)
for i in range(4):
    add(tick(), 23.75 + i * 0.12, 0.12)
for i in range(4):
    add(pop(note(79 + i * 2) * 0.6, note(79 + i * 2)), 24.0 + i * 0.12, 0.12)
add(whoosh(0.4, 800, 3000), 24.75, 0.05)
add(success(note(86)), 26.0, 0.1)
add(whoosh(0.6, 400, 2400), 26.35, 0.08, -0.3)
add(boom(), 26.55, 0.4)
add(bell(note(84)), 26.57, 0.08)
add(noise_sweep(0.5, 1200, 6000, lambda u: np.sin(np.pi * u)), 26.75, 0.04, 0.3)
add(pop(note(84) * 0.6, note(84)), 27.4, 0.18)
add(noise_sweep(0.9, 4000, 12000, lambda u: np.sin(np.pi * u) ** 2, 3.0), 27.4, 0.03)
for at, m in ((27.75, 79), (27.9, 84)):
    add(pop(note(m) * 0.6, note(m)), at, 0.16)
add(boom(0.9), 28.0, 0.3)
add(bell(note(72), 1.6), 28.0, 0.08)
for at, m in ((29.3, 91), (29.4, 96)):
    add(bell(note(m), 0.8), at, 0.05)

reverb(bus)
fo = int(0.35 * SR)
bus.L[-fo:] *= np.linspace(1, 0, fo) ** 2
bus.R[-fo:] *= np.linspace(1, 0, fo) ** 2
music = np.stack([bus.L, bus.R])
music /= np.abs(music).max() + 1e-9
mix = duck_mix(music, vo, mask, music_gain=0.36, depth=0.66)
sf.write(OUT + 'tuto_vo.wav', vo, SR, subtype='PCM_24')
sf.write(OUT + 'tuto_mix_raw.wav', (mix * 0.8).T, SR, subtype='PCM_24')
json.dump(timing, open(OUT + 'tuto_vo_timing.json', 'w'), ensure_ascii=False, indent=1)
print('peak', float(np.abs(mix).max()))
