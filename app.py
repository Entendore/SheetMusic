import sys
import math
import struct
import array
import random
import os
import subprocess
import tempfile
import wave
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QComboBox, QLabel, QPushButton, QTextEdit, QSlider, QGridLayout, QProgressBar, QFileDialog, QSpinBox
)
from PySide6.QtGui import QPainter, QPen, QColor, QFont, QBrush, QPainterPath, QImage
from PySide6.QtCore import Qt, QRectF, QByteArray, QBuffer, QIODeviceBase, QThread, Signal, QSize
from PySide6.QtMultimedia import QAudioSink, QAudioFormat

# ==========================================
# MUSIC THEORY DATA & HELPERS
# ==========================================
NOTE_LETTER_STEPS = {'C': 0, 'D': 1, 'E': 2, 'F': 3, 'G': 4, 'A': 5, 'B': 6}

def note_to_midi(note_str):
    letter = note_str[0]
    accidental, idx = 0, 1
    if len(note_str) > 1 and note_str[1] == '#': accidental, idx = 1, 2
    elif len(note_str) > 1 and note_str[1] == 'b': accidental, idx = -1, 2
    octave = int(note_str[idx:])
    return (octave + 1) * 12 + NOTE_LETTER_STEPS[letter] + accidental

def midi_to_freq(midi_note):
    return 440.0 * (2.0 ** ((midi_note - 69) / 12.0))

KEY_SIGNATURES = {
    "C Major": [], "G Major": ["F#"], "D Major": ["F#", "C#"], 
    "A Major": ["F#", "C#", "G#"], "E Major": ["F#", "C#", "G#", "D#"], 
    "F Major": ["Bb"], "Bb Major": ["Bb", "Eb"], "Eb Major": ["Bb", "Eb", "Ab"],
    "A Natural Minor": [], "E Natural Minor": ["F#"], "B Natural Minor": ["F#", "C#"],
    "D Natural Minor": ["Bb"], "G Natural Minor": ["Bb", "Eb"], "C Natural Minor": ["Bb", "Eb", "Ab"],
}

THEORY_DATA = {
    "Scales": {
        "items": {
            "C Major": [[("C4", 'q')], [("D4", 'q')], [("E4", 'q')], [("F4", 'q')], [("G4", 'q')], [("A4", 'q')], [("B4", 'q')], [("C5", 'q')]],
            "A Natural Minor": [[("A3", 'q')], [("B3", 'q')], [("C4", 'q')], [("D4", 'q')], [("E4", 'q')], [("F4", 'q')], [("G4", 'q')], [("A4", 'q')]],
        },
        "info": "Scales are ordered sequences of notes."
    },
    "Chords": {
        "items": {
            "C Major Triad": [[("C4", 'w'), ("E4", 'w'), ("G4", 'w')]],
            "C Minor Triad": [[("C4", 'w'), ("Eb4", 'w'), ("G4", 'w')]],
            "C Major 7th": [[("C4", 'w'), ("E4", 'w'), ("G4", 'w'), ("B4", 'w')]],
        },
        "info": "Chords are 3+ notes played simultaneously."
    },
    "Progressions": {
        "items": {
            "I - IV - V - I (C Major)": [
                [("C4", 'h'), ("E4", 'h'), ("G4", 'h')], 
                [("F4", 'h'), ("A4", 'h'), ("C5", 'h')],
                [("G4", 'h'), ("B4", 'h'), ("D5", 'h')],
                [("C4", 'h'), ("E4", 'h'), ("G4", 'h')]
            ],
        },
        "info": "Chord progressions are sequences of chords."
    }
}

# ==========================================
# DSP CLASSES (FX & Filters)
# ==========================================
class GranularPitchShift:
    def __init__(self, sr, ratio=2.0):
        self.sr = sr
        self.ratio = ratio
        self.buf_len = int(sr * 0.05)
        self.buf = array.array('f', [0.0] * self.buf_len)
        self.write_idx = 0
        self.read_idx1 = 0.0
        self.read_idx2 = self.buf_len / 2.0
        
    def process(self, inp):
        self.buf[self.write_idx] = inp
        self.write_idx = (self.write_idx + 1) % self.buf_len
        r1 = int(self.read_idx1); f1 = self.read_idx1 - r1
        out1 = self.buf[r1] * (1.0 - f1) + self.buf[(r1 + 1) % self.buf_len] * f1
        r2 = int(self.read_idx2); f2 = self.read_idx2 - r2
        out2 = self.buf[r2] * (1.0 - f2) + self.buf[(r2 + 1) % self.buf_len] * f2
        cf = abs((self.read_idx1 / self.buf_len) - 0.5) * 2.0
        out = out1 * (1.0 - cf) + out2 * cf
        self.read_idx1 += self.ratio; self.read_idx2 += self.ratio
        if self.read_idx1 >= self.buf_len: self.read_idx1 -= self.buf_len
        if self.read_idx2 >= self.buf_len: self.read_idx2 -= self.buf_len
        return out

class FDNReverb:
    def __init__(self, sr, pitch_shifter=None):
        self.sr = sr; self.pitch_shifter = pitch_shifter
        times = [0.0297, 0.0371, 0.0411, 0.0437]
        self.delays = [int(sr * t) for t in times]
        self.buffers = [array.array('f', [0.0] * d) for d in self.delays]
        self.idxs = [0, 0, 0, 0]; self.lp_state = [0.0, 0.0, 0.0, 0.0]; self.lp_coeff = 0.2
        self.H = [[1, 1, 1, 1], [1, -1, 1, -1], [1, 1, -1, -1], [1, -1, -1, 1]]
    def process(self, inp, mix, shimmer_mix=0.0):
        if mix <= 0.0: return inp
        outs = [self.buffers[i][self.idxs[i]] for i in range(4)]
        for i in range(4):
            self.lp_state[i] += self.lp_coeff * (outs[i] - self.lp_state[i])
            outs[i] = self.lp_state[i]
        wet_inp = inp
        if self.pitch_shifter and shimmer_mix > 0.0:
            wet_inp = inp * (1.0 - shimmer_mix) + self.pitch_shifter.process(inp) * shimmer_mix
        mixed = [0.0, 0.0, 0.0, 0.0]
        for i in range(4):
            for j in range(4): mixed[i] += outs[j] * self.H[i][j]
            mixed[i] *= 0.5
        for i in range(4):
            self.buffers[i][self.idxs[i]] = wet_inp + mixed[i] * 0.80
            self.idxs[i] = (self.idxs[i] + 1) % self.delays[i]
        wet_out = sum(outs) * 0.25
        return (inp * (1.0 - mix)) + (wet_out * mix)

class Biquad:
    def __init__(self, sr, type, freq, q=0.707):
        self.sr, self.type = sr, type
        self.set_freq(freq, q)
        self.x1 = self.x2 = self.y1 = self.y2 = 0.0
    def set_freq(self, freq, q):
        w0 = 2 * math.pi * freq / self.sr; cos_w0, sin_w0 = math.cos(w0), math.sin(w0)
        alpha = sin_w0 / (2 * q)
        if self.type == "lp":
            self.b0 = (1 - cos_w0) / 2; self.b1 = 1 - cos_w0; self.b2 = (1 - cos_w0) / 2
            self.a0 = 1 + alpha; self.a1 = -2 * cos_w0; self.a2 = 1 - alpha
        elif self.type == "hp":
            self.b0 = (1 + cos_w0) / 2; self.b1 = -(1 + cos_w0); self.b2 = (1 + cos_w0) / 2
            self.a0 = 1 + alpha; self.a1 = -2 * cos_w0; self.a2 = 1 - alpha
    def process(self, inp):
        out = self.b0 * inp + self.b1 * self.x1 + self.b2 * self.x2 - self.a1 * self.y1 - self.a2 * self.y2
        out /= self.a0
        self.x2 = self.x1; self.x1 = inp; self.y2 = self.y1; self.y1 = out
        return out

class OnePoleLPF:
    def __init__(self, sr, cutoff):
        self.x = 0.0
        self.a = 1.0 - math.exp(-2.0 * math.pi * cutoff / sr)
    def process(self, inp):
        self.x += self.a * (inp - self.x)
        return self.x

class TPT_SVF:
    def __init__(self, sr, cutoff, res):
        self.sr = sr; self.set_cutoff(cutoff, res); self.ic1eq = 0.0; self.ic2eq = 0.0
    def set_cutoff(self, freq, res):
        self.g = math.tan(math.pi * freq / self.sr); self.k = 2.0 - 2.0 * res
    def process(self, inp):
        v1 = (inp - self.ic2eq - self.ic1eq * self.k) / (1.0 + self.g * (self.g + self.k))
        v2 = self.ic1eq + self.g * v1
        self.ic1eq += 2.0 * self.g * v1; self.ic2eq += 2.0 * self.g * v2
        return self.ic2eq

class DCBlocker:
    def __init__(self): self.x1 = self.y1 = 0.0; self.a = 0.995
    def process(self, inp):
        out = inp - self.x1 + self.a * self.y1; self.x1 = inp; self.y1 = out
        return out

class StereoChorus:
    def __init__(self, sr, depth=0.004, rate=0.6, mix=0.5):
        self.sr, self.depth = sr, int(depth * sr)
        self.max_delay = self.depth * 2 + 2
        self.bufferL = array.array('f', [0.0] * self.max_delay)
        self.bufferR = array.array('f', [0.0] * self.max_delay)
        self.idx, self.phase = 0, 0.0
        self.phase_inc = (2.0 * math.pi * rate) / sr; self.mix = mix
    def process(self, inpL, inpR):
        lfo = math.sin(self.phase); self.phase += self.phase_inc
        delayL = int(self.depth * (1.0 + lfo)); delayR = int(self.depth * (1.0 - lfo))
        outL = inpL + self.bufferL[(self.idx - delayL) % self.max_delay] * self.mix
        outR = inpR + self.bufferR[(self.idx - delayR) % self.max_delay] * self.mix
        self.bufferL[self.idx], self.bufferR[self.idx] = inpL, inpR
        self.idx = (self.idx + 1) % self.max_delay
        return outL, outR

class AnalogPingPongDelay:
    def __init__(self, sr, time_ms, feedback, mix):
        self.dl = int(sr * (time_ms / 1000.0))
        self.bufL = array.array('f', [0.0] * self.dl); self.bufR = array.array('f', [0.0] * self.dl)
        self.idx, self.fb, self.mix = 0, feedback, mix
        self.dly_lpL = OnePoleLPF(sr, 2000.0); self.dly_lpR = OnePoleLPF(sr, 2000.0)
    def process(self, inpL, inpR):
        outL = self.bufL[self.idx]; outR = self.bufR[self.idx]
        fbL = self.dly_lpL.process(inpR + outR * self.fb); fbR = self.dly_lpR.process(inpL + outL * self.fb)
        self.bufL[self.idx] = fbL; self.bufR[self.idx] = fbR
        self.idx = (self.idx + 1) % self.dl
        return inpL * (1-self.mix) + outL * self.mix, inpR * (1-self.mix) + outR * self.mix

class LeslieSpeaker:
    def __init__(self, sr):
        self.sr = sr; self.horn_phase = 0.0; self.drum_phase = 0.0
        self.dl_len = int(sr * 0.015)
        self.bufL = array.array('f', [0.0] * self.dl_len); self.bufR = array.array('f', [0.0] * self.dl_len)
        self.idx = 0
    def process(self, inpL, inpR):
        self.horn_phase += (2 * math.pi * 5.5) / self.sr
        delay_h = int((1.0 + math.sin(self.horn_phase)) * 0.5 * (self.dl_len - 1))
        out_hL = self.bufL[(self.idx - delay_h) % self.dl_len]
        out_hR = self.bufR[(self.idx + delay_h) % self.dl_len]
        amp_hL = 0.5 + 0.5 * math.sin(self.horn_phase); amp_hR = 0.5 + 0.5 * math.sin(self.horn_phase + math.pi)
        self.drum_phase += (2 * math.pi * 1.2) / self.sr
        amp_dL = 0.5 + 0.5 * math.sin(self.drum_phase); amp_dR = 0.5 + 0.5 * math.sin(self.drum_phase + math.pi)
        self.bufL[self.idx] = inpL; self.bufR[self.idx] = inpR
        self.idx = (self.idx + 1) % self.dl_len
        return (out_hL * amp_hL + inpL * (1.0 - amp_hL) * 0.5, 
                out_hR * amp_hR + inpR * (1.0 - amp_hR) * 0.5)

# ==========================================
# AUDIO ENGINE
# ==========================================
class AudioEngine:
    def __init__(self):
        self.format = QAudioFormat()
        self.format.setSampleRate(48000) # Upgraded to 48kHz for Video
        self.format.setChannelCount(2)
        self.format.setSampleFormat(QAudioFormat.Int16)
        
        self.audio_sink = QAudioSink(self.format)
        self.audio_sink.setVolume(0.9)
        self.instrument = "Grand Piano"
        self.reverb_mix = 0.35; self.chorus_mix = 0.4; self.delay_mix = 0.3
        self.drive = 0.0; self.brightness = 1.0; self.width_mix = 0.5
        self.air_mix = 0.3; self.shimmer_mix = 0.3

    def generate_voice(self, freq, duration_ms):
        sr = self.format.sampleRate()
        num_samples = int(sr * (duration_ms / 1000.0))
        buf = array.array('f', [0.0] * num_samples)
        
        sin = math.sin; exp = math.exp; tanh = math.tanh
        pi2 = 2.0 * math.pi; random_uniform = random.uniform
        
        velocity_mult = max(0.6, min(1.5, (freq / 200.0))) 
        attack, decay, sustain, release = 0.01, 0.2, 0.6, 0.3
        
        if self.instrument == "Grand Piano": attack, decay, sustain, release = 0.005, 0.4, 0.2, 0.6
        elif self.instrument == "Nylon Guitar": attack, decay, sustain, release = 0.005, 0.8, 0.0, 0.6
        elif self.instrument == "Vector Pad": attack, decay, sustain, release = 0.8, 0.5, 0.9, 2.0
        elif self.instrument == "FM E-Piano": attack, decay, sustain, release = 0.005, 1.2, 0.3, 0.5
        elif self.instrument == "Church Organ": attack, decay, sustain, release = 0.05, 0.1, 0.9, 0.2

        cutoff = (500 + 8000 * self.brightness) * velocity_mult
        svf_filter = TPT_SVF(sr, min(18000, cutoff), res=0.2)
        
        if self.instrument == "Nylon Guitar":
            N = max(2, int(sr / freq)); s1 = array.array('f', [0.0] * N); idx1 = 0; ks_filter = 0.0
            pick_noise = [random_uniform(-1, 1) * exp(-30.0 * (i/sr)) for i in range(int(sr*0.05))]
            for i in range(N): s1[i] = pick_noise[i % len(pick_noise)]
            
        if self.instrument == "Vector Pad":
            wt_phase = 0.0; lfo_x_phase, lfo_y_phase = 0.0, math.pi / 2.0
            
        if self.instrument == "FM E-Piano":
            fm_phase_c1, fm_phase_c2 = 0.0, 0.0; fm_phase_m1, fm_phase_m2 = 0.0, 0.0; fm_fb_m1 = 0.0

        for i in range(num_samples):
            t = i / sr; dur_s = duration_ms / 1000.0
            if t < attack: env = t / attack
            elif t < attack + decay: env = 1.0 - (1.0 - sustain) * ((t - attack) / decay)
            elif t < dur_s - release: env = sustain
            else: env = max(0.0, sustain * (1.0 - ((t - (dur_s - release)) / release)))
            
            val = 0.0; noise_val = random_uniform(-1, 1)
            drift = 1.0 + (random_uniform(-1, 1) * 0.0005); f_drift = freq * drift
            
            if self.instrument == "Grand Piano":
                dec = exp(-2.5 * t)
                h1 = sin(pi2 * f_drift * t); h2 = sin(pi2 * (f_drift*1.002) * t + 0.1); h3 = sin(pi2 * (f_drift*0.998) * t - 0.1)
                val = (h1 + h2 + h3) * 0.33 * dec * env * 0.6
                val += noise_val * exp(-150.0 * t) * 0.3 * velocity_mult
            elif self.instrument == "Nylon Guitar":
                o1 = s1[idx1]; ks_filter = 0.3 * o1 + 0.7 * ks_filter; damped = ks_filter * 0.995
                s1[idx1] = damped; idx1 = (idx1 + 1) % N; val = o1 * env * 0.9
            elif self.instrument == "Vector Pad":
                lfo_x = 0.5 + 0.5 * sin(lfo_x_phase); lfo_y = 0.5 + 0.5 * sin(lfo_y_phase)
                wt_phase += f_drift / sr
                if wt_phase >= 1.0: wt_phase -= 1.0
                w_sine = sin(pi2 * wt_phase); w_tri = 2.0 * abs(2.0 * (wt_phase - math.floor(wt_phase + 0.5))) - 1.0
                w_saw = 2.0 * wt_phase - 1.0; w_sqr = 1.0 if wt_phase < 0.5 else -1.0
                if lfo_x < 0.5: p = lfo_x * 2.0; mix_x = w_sine * (1.0-p) + w_tri * p
                else: p = (lfo_x - 0.5) * 2.0; mix_x = w_tri * (1.0-p) + w_saw * p
                mix_final = mix_x * (1.0 - lfo_y) + w_sqr * lfo_y
                svf_filter.set_cutoff(400 + 2000 * lfo_x * self.brightness, 0.2)
                val = svf_filter.process(mix_final) * env * 0.4
            elif self.instrument == "FM E-Piano":
                mod1_env = exp(-3.0 * t); mod1 = sin(pi2 * f_drift * 2.0 * t + fm_fb_m1) * 2.0 * mod1_env; fm_fb_m1 = mod1 * 0.5
                mod2_env = exp(-1.0 * t); mod2 = sin(pi2 * f_drift * 14.0 * t) * 1.5 * mod2_env
                c1 = sin(pi2 * f_drift * 1.0 * t + mod1); c2 = sin(pi2 * f_drift * 1.0 * t + mod2)
                val = (c1 * 0.7 + c2 * 0.3) * env * 0.5
            elif self.instrument == "Church Organ":
                f1 = sin(pi2 * f_drift * t); f2 = sin(pi2 * f_drift * 2 * t); f3 = sin(pi2 * f_drift * 3 * t); f4 = sin(pi2 * f_drift * 4 * t)
                chiff = noise_val * exp(-50.0 * t) * 0.2
                val = (f1*0.8 + f2*0.4 + f3*0.2 + f4*0.2) * env * 0.3 + chiff

            if self.drive > 0.0:
                drive_amt = 1.0 + (self.drive * 10.0)
                val = math.atan(val * drive_amt) / math.atan(drive_amt)

            if self.instrument == "Vector Pad":
                lfo_x_phase += (pi2 * 0.1) / sr; lfo_y_phase += (pi2 * 0.07) / sr
                
            buf[i] = val
        return buf

    def generate_audio_bytes(self, steps):
        """Runs in worker thread. Returns raw PCM bytes."""
        if not steps: return b''
        master_buffer = array.array('f')
        
        for step in steps:
            if not step: continue
            duration_ms = 400 if step[0][1] == 'q' else 800 if step[0][1] == 'h' else 1500
            voices = [self.generate_voice(midi_to_freq(note_to_midi(n[0])), duration_ms) for n in step]
            max_len = max(len(v) for v in voices)
            mixed_buffer = array.array('f', [0.0] * max_len)
            num_voices = len(voices)
            voice_gain = 1.0 / math.sqrt(num_voices)
            for v in voices:
                for i in range(len(v)): mixed_buffer[i] += v[i] * voice_gain
            mixed_buffer.extend([0.0] * int(self.format.sampleRate() * 0.03))
            master_buffer.extend(mixed_buffer)
            
        dc_blocker = DCBlocker()
        dc_blocked = array.array('f', [dc_blocker.process(s) for s in master_buffer])
        pitch_shifter = GranularPitchShift(self.format.sampleRate(), 2.0)
        fdn = FDNReverb(self.format.sampleRate(), pitch_shifter)
        wet_buffer = array.array('f', [fdn.process(s, self.reverb_mix, self.shimmer_mix) for s in dc_blocked])
        
        chorus = StereoChorus(self.format.sampleRate(), mix=self.chorus_mix)
        delay = AnalogPingPongDelay(self.format.sampleRate(), 250, 0.4, self.delay_mix)
        leslie = LeslieSpeaker(self.format.sampleRate())
        
        stereo_L = array.array('f', [0.0] * len(wet_buffer))
        stereo_R = array.array('f', [0.0] * len(wet_buffer))
        shift = int(self.format.sampleRate() * 0.01)
        
        for i in range(len(wet_buffer)):
            val = wet_buffer[i]; int_val = max(-1.0, min(1.0, val))
            left_val = int_val if i < shift else max(-1.0, min(1.0, wet_buffer[i-shift] * 0.8))
            right_val = int_val
            if self.chorus_mix > 0 or self.delay_mix > 0:
                fL, fR = left_val, right_val
                if self.delay_mix > 0: fL, fR = delay.process(fL, fR)
                if self.chorus_mix > 0: fL, fR = chorus.process(fL, fR)
                left_val, right_val = fL, fR
            if self.instrument == "Church Organ": left_val, right_val = leslie.process(left_val, right_val)
            stereo_L[i] = left_val; stereo_R[i] = right_val
            
        lp1_l = Biquad(self.format.sampleRate(), "lp", 250, 0.5); lp2_l = Biquad(self.format.sampleRate(), "lp", 250, 0.5)
        hp1_l = Biquad(self.format.sampleRate(), "hp", 250, 0.5); hp2_l = Biquad(self.format.sampleRate(), "hp", 250, 0.5)
        bp1_l = Biquad(self.format.sampleRate(), "lp", 2500, 0.5); bp2_l = Biquad(self.format.sampleRate(), "hp", 2500, 0.5)
        lp1_r = Biquad(self.format.sampleRate(), "lp", 250, 0.5); lp2_r = Biquad(self.format.sampleRate(), "lp", 250, 0.5)
        hp1_r = Biquad(self.format.sampleRate(), "hp", 250, 0.5); hp2_r = Biquad(self.format.sampleRate(), "hp", 250, 0.5)
        bp1_r = Biquad(self.format.sampleRate(), "lp", 2500, 0.5); bp2_r = Biquad(self.format.sampleRate(), "hp", 2500, 0.5)
        tape_lp = OnePoleLPF(self.format.sampleRate(), 6000)
        
        audio_bytes = bytearray()
        for i in range(len(stereo_L)):
            low_L = lp2_l.process(lp1_l.process(stereo_L[i])); high_L = hp2_l.process(hp1_l.process(stereo_L[i])); mid_L = bp2_l.process(bp1_l.process(stereo_L[i]))
            low_R = lp2_r.process(lp1_r.process(stereo_R[i])); high_R = hp2_r.process(hp1_r.process(stereo_R[i])); mid_R = bp2_r.process(bp1_r.process(stereo_R[i]))
            low_L_sat = tape_lp.process(math.tanh(low_L * 1.2) * 0.9); low_R_sat = tape_lp.process(math.tanh(low_R * 1.2) * 0.9)
            mid_L_warm = math.tanh(mid_L * 0.8); mid_R_warm = math.tanh(mid_R * 0.8)
            high_L_exc = high_L + (abs(high_L) * 0.5 * self.air_mix); high_R_exc = high_R + (abs(high_R) * 0.5 * self.air_mix)
            m = (high_L_exc + high_R_exc) * 0.5; s = (high_L_exc - high_R_exc) * 0.5; s *= (1.0 + self.width_mix)
            high_L_wide = m + s; high_R_wide = m - s
            out_L = math.tanh((low_L_sat + mid_L_warm + high_L_wide) * 0.9)
            out_R = math.tanh((low_R_sat + mid_R_warm + high_R_wide) * 0.9)
            i_L = int(max(-1.0, min(1.0, out_L)) * 32767); i_R = int(max(-1.0, min(1.0, out_R)) * 32767)
            audio_bytes.extend(struct.pack('<hh', i_L, i_R))
                
        return bytes(audio_bytes)

    def play_audio(self, audio_bytes):
        self.audio_sink.stop()
        self._qba = QByteArray(audio_bytes)
        self._buf = QBuffer(self._qba)
        self._buf.open(QIODeviceBase.ReadOnly)
        self.audio_sink.start(self._buf)

# ==========================================
# HIGH RESOLUTION VIDEO RENDERER
# ==========================================
class VideoRenderer:
    @staticmethod
    def render_frame(painter, w, h, t, audio_data, steps, title_text, active_key):
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(0, 0, w, h, QColor("#0f0f1e"))
        
        # 1. Title
        painter.setPen(QColor("#cdd6f4"))
        painter.setFont(QFont("Arial", int(h * 0.04), QFont.Bold))
        painter.drawText(QRectF(0, h * 0.03, w, h * 0.06), Qt.AlignCenter, title_text)
        
        # 2. Circle of Fifths (Top Right/Left depending on layout)
        cof_r = min(w, h) * 0.15
        cof_cx = w - cof_r - (w * 0.03)
        cof_cy = h * 0.25
        VideoRenderer.draw_cof(painter, cof_cx, cof_cy, cof_r, active_key)
        
        # 3. Piano Keyboard (Bottom)
        kb_h = h * 0.15
        kb_y = h - kb_h - (h * 0.03)
        VideoRenderer.draw_piano(painter, 0, kb_y, w, kb_h, steps, t)
        
        # 4. Sheet Music & Note Highlighting (Center)
        VideoRenderer.draw_sheet_music(painter, w * 0.05, h * 0.35, w * 0.6, h * 0.4, steps, t)
        
        # 5. Audio Waveform (Center Right)
        wf_x = w * 0.70; wf_y = h * 0.35; wf_w = w * 0.25; wf_h = h * 0.4
        VideoRenderer.draw_waveform(painter, wf_x, wf_y, wf_w, wf_h, audio_data, t)

    @staticmethod
    def draw_cof(painter, cx, cy, r, active_key):
        m = ["C", "G", "D", "A", "E", "B", "F#", "Db", "Ab", "Eb", "Bb", "F"]
        mi = ["Am", "Em", "Bm", "F#m", "C#m", "G#m", "D#m", "Bbm", "Fm", "Cm", "Gm", "Dm"]
        painter.setFont(QFont("Arial", int(r * 0.12), QFont.Bold))
        for i in range(12):
            a = 90 - (i * 30) + 15
            pm = QPainterPath(); pm.moveTo(cx, cy)
            pm.arcTo(QRectF(cx - r, cy - r, r * 2, r * 2), a, -30); pm.closeSubpath()
            painter.fillPath(pm, QBrush(QColor("#89b4fa") if f"{m[i]} Major" == active_key else QColor("#313244")))
            painter.setPen(QPen(QColor("#11111b"), 1)); painter.drawPath(pm)
            ir = r * 0.6
            pmn = QPainterPath(); pmn.moveTo(cx, cy)
            pmn.arcTo(QRectF(cx - ir, cy - ir, ir * 2, ir * 2), a, -30); pmn.closeSubpath()
            painter.fillPath(pmn, QBrush(QColor("#f38ba8") if f"{mi[i][:-1]} Natural Minor" == active_key else QColor("#45475a")))
            painter.drawPath(pmn)
            ta = math.radians(90 - (i * 30))
            painter.setPen(QColor("#cdd6f4"))
            painter.drawText(QRectF(cx + (r * 0.8) * math.cos(ta) - 20, cy - (r * 0.8) * math.sin(ta) - 10, 40, 20), Qt.AlignCenter, m[i])

    @staticmethod
    def draw_piano(painter, x, y, w, h, steps, t):
        wk = ['C', 'D', 'E', 'F', 'G', 'A', 'B']
        bk = {'C': 'C#', 'D': 'D#', 'F': 'F#', 'G': 'G#', 'A': 'A#'}
        nwk = len(wk) * 2 + 1
        ww, bw, bh = w / nwk, (w / nwk) * 0.6, h * 0.6
        
        # Find active notes at time t
        curr_time = 0.0
        active_notes = []
        for step in steps:
            if not step: continue
            dur = 0.4 if step[0][1] == 'q' else 0.8 if step[0][1] == 'h' else 1.5
            if curr_time <= t < curr_time + dur:
                active_notes = [n[0] for n in step]
                break
            curr_time += dur + 0.03
            
        aw, ab = [], []
        for n in active_notes:
            o = int(n[-1]); idx = (o - 4) * 7 + wk.index(n[0])
            if len(n) > 2 and n[1] in ['#', 'b']: ab.append((n[0], o, n))
            else: aw.append((idx, n))
            
        for i in range(nwk):
            px = x + i * ww; nn = f"{wk[i % 7]}{4 + (i // 7)}"
            c = QColor("#f38ba8") if nn == (active_notes[0] if active_notes else "") else QColor("#89b4fa") if any(nn == n for _, n in aw) else QColor("#ffffff")
            painter.setBrush(QBrush(c)); painter.setPen(QPen(QColor("#11111b"), 1))
            painter.drawRoundedRect(QRectF(px + 2, y, ww - 4, h), 4, 4)
            
        for i in range(nwk - 1):
            px = x + (i + 1) * ww; l = wk[i % 7]
            if l in bk:
                nn = f"{bk[l]}{4 + (i // 7)}"
                c = QColor("#f38ba8") if nn == (active_notes[0] if active_notes else "") else QColor("#89b4fa") if nn in [n for _, _, n in ab] else QColor("#1e1e2e")
                painter.setBrush(QBrush(c)); painter.setPen(QPen(QColor("#000000"), 1))
                painter.drawRoundedRect(QRectF(px - (bw / 2), y, bw, bh), 3, 3)

    @staticmethod
    def draw_sheet_music(painter, x, y, w, h, steps, t):
        painter.setPen(QPen(QColor("#cdd6f4"), 2))
        staff_top = y + h * 0.2; ls = 14
        for i in range(5): painter.drawLine(x, staff_top + (i * ls), x + w, staff_top + (i * ls))
        painter.setFont(QFont("Segoe UI Symbol", 55))
        painter.drawText(x + 10, staff_top + (4 * ls) + 8, "𝄞")
        
        curr_time = 0.0; x_off = x + 70
        for step in steps:
            if not step: continue
            dur = 0.4 if step[0][1] == 'q' else 0.8 if step[0][1] == 'h' else 1.5
            is_active = curr_time <= t < curr_time + dur
            curr_time += dur + 0.03
            
            for ns, d in step:
                letter, acc, oi = ns[0], None, 1
                if len(ns) > 1 and ns[1] in ['#', 'b']: acc, oi = ns[1], 2
                oct = int(ns[oi:])
                y_note = staff_top + (4 * ls) - (((oct - 4) * 7 + NOTE_LETTER_STEPS[letter] - 2) * (ls / 2))
                
                painter.setPen(QPen(QColor("#f38ba8") if is_active else QColor("#cdd6f4"), 2))
                painter.setBrush(QBrush(QColor("#f38ba8") if is_active else QColor("#cdd6f4")) if d == 'q' else QBrush(QColor("#0f0f1e")))
                painter.save(); painter.translate(x_off, y_note); painter.rotate(-20)
                painter.drawEllipse(-7, -5, 14, 10); painter.restore()
            x_off += 60

    @staticmethod
    def draw_waveform(painter, x, y, w, h, audio_data, t):
        if not audio_data: return
        sr = 48000
        total_samples = len(audio_data) // 4 # 16-bit stereo = 4 bytes per sample
        curr_sample = int(t * sr)
        
        painter.setPen(QPen(QColor("#a6e3a1"), 2))
        mid_y = y + h / 2
        samples_to_draw = int(w * 2) # Draw roughly 2 samples per pixel
        for i in range(0, int(w), 2):
            idx = curr_sample + i * 2
            if idx < total_samples:
                # Unpack 16-bit stereo
                L = struct.unpack('<h', audio_data[idx*4 : idx*4+2])[0] / 32767.0
                R = struct.unpack('<h', audio_data[idx*4+2 : idx*4+4])[0] / 32767.0
                val = (L + R) * 0.5
                painter.drawLine(x + i, mid_y, x + i, mid_y - val * (h * 0.4))

# ==========================================
# YOUTUBE / SHORTS EXPORT WORKER
# ==========================================
class VideoExportWorker(QThread):
    progress = Signal(int)
    finished = Signal(str)
    error = Signal(str)

    def __init__(self, engine, steps, title, resolution, fps):
        super().__init__()
        self.engine = engine; self.steps = steps; self.title = title
        self.resolution = resolution; self.fps = fps

    def run(self):
        try:
            # 1. Generate Audio
            self.progress.emit(5)
            audio_bytes = self.engine.generate_audio_bytes(self.steps)
            if not audio_bytes:
                self.error.emit("No audio generated.")
                return

            # 2. Save Temp WAV
            self.progress.emit(20)
            temp_dir = tempfile.gettempdir()
            wav_path = os.path.join(temp_dir, "music_theory_audio.wav")
            with wave.open(wav_path, 'w') as wf:
                wf.setnchannels(2)
                wf.setsampwidth(2)
                wf.setframerate(48000)
                wf.writeframes(audio_bytes)

            # 3. Setup FFmpeg
            self.progress.emit(30)
            w, h = self.resolution
            out_path, _ = QFileDialog.getSaveFileName(None, "Save Video", f"{self.title}.mp4", "MP4 Video (*.mp4)")
            if not out_path:
                self.error.emit("Export cancelled.")
                return

            cmd = [
                'ffmpeg', '-y',
                '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                '-s', f'{w}x{h}', '-r', str(self.fps),
                '-i', '-',
                '-i', wav_path,
                '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                '-c:a', 'aac', '-shortest',
                out_path
            ]
            process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

            # 4. Render Frames
            audio_dur = len(audio_bytes) / (48000 * 4) # 48k stereo 16-bit
            total_frames = int(audio_dur * self.fps)
            
            for frame_idx in range(total_frames):
                t = frame_idx / self.fps
                
                img = QImage(w, h, QImage.Format_RGB888)
                img.fill(QColor("#0f0f1e"))
                painter = QPainter(img)
                VideoRenderer.render_frame(
                    painter, w, h, t, audio_bytes, self.steps, 
                    self.title, "C Major" # Simplified for export
                )
                painter.end()
                
                # Write to FFmpeg
                process.stdin.write(img.constBits().tobytes())
                
                if frame_idx % 5 == 0:
                    prog = 30 + int((frame_idx / total_frames) * 70)
                    self.progress.emit(prog)

            process.stdin.close()
            process.wait()
            os.remove(wav_path)
            self.progress.emit(100)
            self.finished.emit(out_path)

        except Exception as e:
            self.error.emit(str(e))

# ==========================================
# UI WIDGETS (For Live Preview)
# ==========================================
class VideoCanvas(QWidget):
    def __init__(self):
        super().__init__()
        self.steps = []; self.title = ""; self.audio_data = b""
        self.t = 0.0; self.active_key = "C Major"
        self.setMinimumSize(720, 1280) # Shorts default
        self.setStyleSheet("background-color: #0f0f1e; border-radius: 8px;")
        self.timer = self.startTimer(33) # ~30fps

    def set_data(self, steps, title, audio_data, active_key):
        self.steps = steps; self.title = title
        self.audio_data = audio_data; self.active_key = active_key
        self.t = 0.0

    def timerEvent(self, event):
        if self.audio_data:
            audio_dur = len(self.audio_data) / (48000 * 4)
            self.t += 0.033
            if self.t > audio_dur: self.t = 0.0
            self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        # Scale preview to fit widget while maintaining aspect ratio
        s = min(self.width() / 1080.0, self.height() / 1920.0)
        painter.scale(s, s)
        VideoRenderer.render_frame(painter, 1080, 1920, self.t, self.audio_data, self.steps, self.title, self.active_key)

# ==========================================
# MAIN APPLICATION WINDOW
# ==========================================
class MusicTheoryApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Music Theory Studio Pro - YouTube Creator Edition")
        self.resize(1200, 900)
        self.setStyleSheet("QMainWindow { background-color: #1e1e2e; }")

        cw = QWidget(); self.setCentralWidget(cw)
        ml = QVBoxLayout(cw); ml.setSpacing(10); ml.setContentsMargins(20, 20, 20, 20)

        title_lbl = QLabel("🎬 Music Theory Studio")
        title_lbl.setStyleSheet("font-size: 24px; font-weight: bold; color: #cdd6f4;")
        ml.addWidget(title_lbl)
        
        cl = QHBoxLayout()
        self.cat = QComboBox(); self.cat.addItems(THEORY_DATA.keys()); self.cat.currentTextChanged.connect(self.upd_sub)
        self.item = QComboBox(); self.item.currentTextChanged.connect(self.display)
        self.play_btn = QPushButton("▶ Play Audio"); self.play_btn.clicked.connect(self.play)
        self.inst = QComboBox()
        self.inst.addItems(["Grand Piano", "Nylon Guitar", "Vector Pad", "FM E-Piano", "Church Organ"])
        self.inst.currentTextChanged.connect(lambda i: setattr(self.ae, 'instrument', i))
        self.vol = QSlider(Qt.Horizontal); self.vol.setRange(0, 100); self.vol.setValue(90); self.vol.valueChanged.connect(self.set_vol)
        
        cl.addWidget(QLabel("Category:")); cl.addWidget(self.cat, 1)
        cl.addWidget(QLabel("Selection:")); cl.addWidget(self.item, 2)
        cl.addWidget(self.play_btn); cl.addSpacing(20)
        cl.addWidget(QLabel("Instrument:")); cl.addWidget(self.inst)
        cl.addWidget(QLabel("Vol:")); cl.addWidget(self.vol)
        ml.addLayout(cl)
        
        # FX Matrix
        fx_grid = QGridLayout()
        self.rev = self._make_slider("Reverb:", 35, self.set_rev)
        self.shm = self._make_slider("Shimmer:", 30, self.set_shm)
        self.chr = self._make_slider("Chorus:", 40, self.set_chr)
        self.dly = self._make_slider("Delay:", 30, self.set_dly)
        self.drv = self._make_slider("Drive:", 0, self.set_drv)
        self.brt = self._make_slider("Bright:", 80, self.set_brt)
        self.wdt = self._make_slider("Width:", 50, self.set_wdt)
        self.air = self._make_slider("Air:", 30, self.set_air)
        
        for i, (lbl, sld) in enumerate([self.rev, self.shm, self.chr, self.dly, self.drv, self.brt, self.wdt, self.air]):
            fx_grid.addWidget(lbl, 0, i); fx_grid.addWidget(sld, 1, i)
        ml.addLayout(fx_grid)

        # Content Creation Area
        mid = QHBoxLayout()
        
        # Left side: YouTube/Shorts Canvas
        self.canvas = VideoCanvas()
        mid.addWidget(self.canvas, 2)
        
        # Right side: Controls & Info
        right_panel = QVBoxLayout()
        
        export_box = QGridLayout()
        self.title_input = QTextEdit(); self.title_input.setPlaceholderText("Enter Video Title..."); self.title_input.setMaximumHeight(40)
        self.fmt_combo = QComboBox(); self.fmt_combo.addItems(["YouTube Shorts (1080x1920)", "YouTube Video (1920x1080)"])
        self.fps_spin = QSpinBox(); self.fps_spin.setRange(24, 60); self.fps_spin.setValue(30)
        self.export_btn = QPushButton("📤 Export MP4"); self.export_btn.setStyleSheet("background-color: #f38ba8; color: #11111b; font-weight: bold; padding: 10px;")
        self.export_btn.clicked.connect(self.export_video)
        self.progress_bar = QProgressBar(); self.progress_bar.setValue(0); self.progress_bar.setTextVisible(True)
        
        export_box.addWidget(QLabel("Title:"), 0, 0); export_box.addWidget(self.title_input, 0, 1)
        export_box.addWidget(QLabel("Format:"), 1, 0); export_box.addWidget(self.fmt_combo, 1, 1)
        export_box.addWidget(QLabel("FPS:"), 2, 0); export_box.addWidget(self.fps_spin, 2, 1)
        export_box.addWidget(self.export_btn, 3, 0, 1, 2)
        export_box.addWidget(self.progress_bar, 4, 0, 1, 2)
        right_panel.addLayout(export_box)

        self.tt = QTextEdit(); self.tt.setReadOnly(True); self.tt.setFixedHeight(80)
        self.tt.setStyleSheet("background-color: #313244; color: #a6e3a1; border-radius: 8px; padding: 10px; font-size: 13px;")
        right_panel.addWidget(self.tt)
        
        mid.addLayout(right_panel, 1)
        ml.addLayout(mid, 1)

        self.ae = AudioEngine()
        self.worker = None; self.video_worker = None
        self.current_audio = b""
        self.upd_sub()

    def _make_slider(self, label, default, callback):
        lbl = QLabel(label); sld = QSlider(Qt.Horizontal)
        sld.setRange(0, 100); sld.setValue(default); sld.valueChanged.connect(callback)
        return lbl, sld

    def upd_sub(self):
        self.item.clear()
        self.item.addItems(list(THEORY_DATA[self.cat.currentText()]["items"].keys()))

    def display(self):
        c, i = self.cat.currentText(), self.item.currentText()
        if not i: return
        s = THEORY_DATA[c]["items"][i]
        ks = KEY_SIGNATURES.get(i, [])
        if c == "Progressions": ks = KEY_SIGNATURES.get("C Major", [])
        self.tt.setText(f"--- {i} ---\nNotes: {', '.join([n[0] for st in s for n in st])}\n\n{THEORY_DATA[c]['info']}")
        self.title_input.setPlainText(i)
        # Trigger preview generation in background
        self.play_btn.setEnabled(False); self.play_btn.setText("⏳ Synthesizing DSP...")
        self.worker = AudioWorker(self.ae, s)
        self.worker.finished.connect(self.on_preview_ready)
        self.worker.start()

    def on_preview_ready(self, audio_bytes):
        self.current_audio = audio_bytes
        s = THEORY_DATA[self.cat.currentText()]["items"][self.item.currentText()]
        self.canvas.set_data(s, self.title_input.toPlainText(), audio_bytes, self.item.currentText())
        self.play_btn.setEnabled(True); self.play_btn.setText("▶ Play Audio")

    def play(self):
        if not self.current_audio: return
        self.ae.play_audio(self.current_audio)

    def export_video(self):
        if not self.current_audio:
            self.error("Generate audio first by selecting a piece.")
            return
            
        fmt = self.fmt_combo.currentText()
        res = (1080, 1920) if "Shorts" in fmt else (1920, 1080)
        fps = self.fps_spin.value()
        title = self.title_input.toPlainText() or "Music Theory"
        s = THEORY_DATA[self.cat.currentText()]["items"][self.item.currentText()]
        
        self.export_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Rendering...")
        
        self.video_worker = VideoExportWorker(self.ae, s, title, res, fps)
        self.video_worker.progress.connect(lambda p: self.progress_bar.setValue(p))
        self.video_worker.finished.connect(self.on_video_done)
        self.video_worker.error.connect(self.on_video_error)
        self.video_worker.start()

    def on_video_done(self, path):
        self.progress_bar.setFormat(f"Saved: {os.path.basename(path)}")
        self.export_btn.setEnabled(True)

    def on_video_error(self, msg):
        self.progress_bar.setFormat(f"Error: {msg}")
        self.export_btn.setEnabled(True)

    # Setters
    def set_vol(self, v): self.ae.audio_sink.setVolume(v / 100.0)
    def set_rev(self, v): self.ae.reverb_mix = v / 100.0
    def set_shm(self, v): self.ae.shimmer_mix = v / 100.0
    def set_chr(self, v): self.ae.chorus_mix = v / 100.0
    def set_dly(self, v): self.ae.delay_mix = v / 100.0
    def set_drv(self, v): self.ae.drive = v / 100.0
    def set_brt(self, v): self.ae.brightness = v / 100.0
    def set_wdt(self, v): self.ae.width_mix = v / 100.0
    def set_air(self, v): self.ae.air_mix = v / 100.0

class AudioWorker(QThread):
    finished = Signal(bytes)
    def __init__(self, engine, steps):
        super().__init__(); self.engine, self.steps = engine, steps
    def run(self):
        audio_bytes = self.engine.generate_audio_bytes(self.steps)
        self.finished.emit(audio_bytes)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet("""
        QComboBox { background-color: #313244; color: #cdd6f4; padding: 8px; border-radius: 4px; border: 1px solid #45475a; }
        QComboBox::drop-down { border: none; }
        QComboBox QAbstractItemView { background-color: #313244; color: #cdd6f4; selection-background-color: #89b4fa; }
        QPushButton { background-color: #a6e3a1; color: #11111b; font-weight: bold; padding: 8px 16px; border-radius: 4px; }
        QPushButton:hover { background-color: #94e896; }
        QPushButton:disabled { background-color: #585b70; color: #11111b; }
        QLabel { color: #cdd6f4; font-weight: bold; }
        QSlider::groove:horizontal { border-radius: 3px; height: 6px; background: #313244; }
        QSlider::handle:horizontal { background: #89b4fa; width: 16px; margin: -5px 0; border-radius: 8px; }
        QTextEdit { background-color: #313244; color: #cdd6f4; border: 1px solid #45475a; border-radius: 4px; }
    """)
    win = MusicTheoryApp()
    win.show()
    sys.exit(app.exec())