// BEGIN StreamResampler16k — extracted verbatim from frontend/index.html.
// The legacy html copy is deleted in phase B; this module is the canonical one.
// Streaming 16 kHz resampler: windowed-sinc anti-alias FIR at the input
// rate (cutoff 7200 Hz, Blackman window), then linear-interpolation read
// at inRate/16000. Stateful across chunks: FIR history ring + fractional
// read position, so chunk boundaries add no clicks. Plain ES2015 — also
// runs unchanged in Node (tests/test_resampler_js.py imports this module).
export class StreamResampler16k {
  constructor(inRate) {
    this.inRate = inRate | 0;
    this.outRate = 16000;
    this.passthrough = this.inRate === this.outRate;
    this.step = this.passthrough ? 1 : this.inRate / this.outRate;
    this.ntaps = 0;
    this.h = null;
    this.ring = null;
    this.ringPos = 0;
    this.absIn = 0;
    this.fy = [];
    this.readPos = 0;
    if (this.passthrough) return;

    const fc = Math.min(7200, 0.45 * this.inRate);
    let n = Math.round((129 * this.inRate) / 48000);
    if (n % 2 === 0) n += 1;
    if (n < 33) n = 33;
    this.ntaps = n;
    const h = new Float64Array(n);
    const fcNorm = fc / this.inRate;
    const mid = (n - 1) / 2;
    let sum = 0;
    for (let i = 0; i < n; i++) {
      const x = i - mid;
      const t = 2 * fcNorm * x;
      const sinc = t === 0 ? 1 : Math.sin(Math.PI * t) / (Math.PI * t);
      const u = i / (n - 1);
      const win = 0.42 - 0.5 * Math.cos(2 * Math.PI * u) + 0.08 * Math.cos(4 * Math.PI * u);
      const v = 2 * fcNorm * sinc * win;
      h[i] = v;
      sum += v;
    }
    for (let i = 0; i < n; i++) h[i] /= sum;
    this.h = h;
    this.ring = new Float64Array(n);
  }

  process(chunk) {
    if (this.passthrough) return chunk;
    const out = [];
    const n = this.ntaps, h = this.h, ring = this.ring;
    const fy = this.fy;
    for (let s = 0; s < chunk.length; s++) {
      ring[this.ringPos] = chunk[s];
      let acc = 0, idx = this.ringPos;
      for (let k = 0; k < n; k++) {
        acc += h[k] * ring[idx];
        idx = idx === 0 ? n - 1 : idx - 1;
      }
      this.ringPos = this.ringPos === n - 1 ? 0 : this.ringPos + 1;
      this.absIn++;
      fy.push(acc);
      if (fy.length > 64) fy.splice(0, fy.length - 64);
      const lastIdx = this.absIn - 1;
      while (this.readPos + 1 <= lastIdx) {
        const i0 = Math.floor(this.readPos);
        const frac = this.readPos - i0;
        const base = this.absIn - fy.length;
        const j0 = i0 - base, j1 = j0 + 1;
        const y0 = j0 >= 0 ? fy[j0] : 0;
        const y1 = j1 >= 0 && j1 < fy.length ? fy[j1] : 0;
        out.push(y0 + (y1 - y0) * frac);
        this.readPos += this.step;
      }
    }
    return Float32Array.from(out);
  }
}
