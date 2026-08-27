"""
Test NVIDIA's Armenian ASR model on a telephony recording.

Model: nvidia/stt_hy_fastconformer_hybrid_large_pc
        FastConformer-Hybrid Large (~115M params), joint RNNT (Transducer) + CTC heads,
        1024-token SentencePiece vocab, outputs Armenian text WITH punctuation and
        capitalization ("_pc").

The model expects 16 kHz mono. Telephony recordings are usually 8 kHz narrowband,
so we upsample -- see the caveat printed at the end of the run.

Two decode strategies are compared, because feeding a 4-minute call in as a single
utterance is well outside what this model was trained on (short read-speech clips):

  full     one forward pass over the whole recording
  chunked  split on silence into <=CHUNK_MAX_SEC pieces, transcribe as a batch, stitch

Usage:
    python test_stt_model.py                          # both modes, both decoders
    python test_stt_model.py --audio other.mp3
    python test_stt_model.py --decoders rnnt          # skip the CTC pass
    python test_stt_model.py --mode chunked
    python test_stt_model.py --local-attn             # bounded-memory long-audio mode
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Armenian output dies on the default Windows console codepage.
for _stream in (sys.stdout, sys.stderr):
    _reconfigure = getattr(_stream, "reconfigure", None)
    if _reconfigure is not None:
        _reconfigure(encoding="utf-8", errors="replace")

# Keep the ~460 MB checkpoint and HF cache off the (nearly full) system drive.
# Must be set before NeMo / huggingface_hub are imported.
CACHE_ROOT = Path(os.environ.get("STT_CACHE_ROOT", r"E:\nemo_stt_hy"))
os.environ.setdefault("HF_HOME", str(CACHE_ROOT / "hf"))
os.environ.setdefault("NEMO_CACHE_DIR", str(CACHE_ROOT / "nemo_cache"))

MODEL_NAME = "nvidia/stt_hy_fastconformer_hybrid_large_pc"
# Pre-downloaded checkpoint; if absent we fall back to from_pretrained() over the network.
LOCAL_NEMO = CACHE_ROOT / "stt_hy_fastconformer_hybrid_large_pc.nemo"
TARGET_SR = 16_000
DEFAULT_AUDIO = HERE / "call_1785309714.115644_0.mp3"

# Armenian sentence enders. NeMo's own segment splitter only knows . ? ! so it never
# splits Armenian text -- U+0589 ARMENIAN FULL STOP is the real terminator here.
TERMINATORS = "։՞՜.?!"  # ։ ՞ ՜ . ? !
CHUNK_MAX_SEC = 20.0   # keep each chunk near the model's training utterance length
CHUNK_PAD_SEC = 0.2    # don't clip word onsets at a silence boundary
SILENCE_TOP_DB = 30.0


def banner(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}", flush=True)


# --------------------------------------------------------------------------- #
# 1. Audio preparation
# --------------------------------------------------------------------------- #
def prepare_audio(src: Path, dst: Path) -> dict:
    """Decode `src` to 16 kHz mono PCM-16 WAV at `dst`. Returns a stats dict."""
    import librosa
    import numpy as np
    import soundfile as sf

    banner("AUDIO PREPARATION")

    native_sr = librosa.get_samplerate(str(src))
    raw, _ = librosa.load(str(src), sr=None, mono=False)
    n_channels = 1 if raw.ndim == 1 else raw.shape[0]

    # sr=TARGET_SR triggers a soxr_hq resample; mono=True averages channels.
    wav, sr = librosa.load(str(src), sr=TARGET_SR, mono=True)
    sf.write(str(dst), wav, sr, subtype="PCM_16")

    duration = len(wav) / sr
    stats = {
        "source": str(src),
        "wav": str(dst),
        "native_sample_rate": native_sr,
        "native_channels": n_channels,
        "resampled_to": sr,
        "duration_sec": round(duration, 2),
        "peak": round(float(np.abs(wav).max()), 4),
        "rms_dbfs": round(float(20 * np.log10(np.sqrt((wav ** 2).mean()) + 1e-12)), 2),
        "silence_ratio": round(float((np.abs(wav) < 1e-3).mean()), 3),
    }
    for k, v in stats.items():
        print(f"  {k:<20}: {v}")

    if native_sr < TARGET_SR:
        print(
            f"\n  NOTE: source is {native_sr} Hz narrowband; upsampled to {TARGET_SR} Hz."
            "\n        There is no real energy above "
            f"{native_sr // 2} Hz, which the model was trained to use."
        )
    return stats


# --------------------------------------------------------------------------- #
# 2. Model loading
# --------------------------------------------------------------------------- #
def load_model(local_attn: bool):
    import torch

    banner("MODEL LOADING")
    torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))
    print(f"  torch {torch.__version__} | cuda={torch.cuda.is_available()} "
          f"| threads={torch.get_num_threads()}")

    import nemo.collections.asr as nemo_asr
    from nemo.utils import logging as nemo_logging

    nemo_logging.setLevel("ERROR")

    cls = nemo_asr.models.EncDecHybridRNNTCTCBPEModel
    t0 = time.perf_counter()
    if LOCAL_NEMO.exists():
        print(f"  source              : {LOCAL_NEMO}")
        model = cls.restore_from(restore_path=str(LOCAL_NEMO), map_location="cpu")
    else:
        print(f"  source              : hub -> {MODEL_NAME}")
        model = cls.from_pretrained(model_name=MODEL_NAME, map_location="cpu")
    model.eval()
    print(f"  loaded in {time.perf_counter() - t0:.1f}s")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  parameters          : {n_params / 1e6:.1f}M")
    print(f"  encoder             : {type(model.encoder).__name__}")
    print(f"  vocab size          : {len(model.tokenizer.vocab)}")
    print(f"  train sample rate   : {model.cfg.preprocessor.sample_rate} Hz")

    if local_attn:
        # Bounds attention memory on long recordings (context = 128 frames each side).
        model.change_attention_model("rel_pos_local_attn", [128, 128])
        model.change_subsampling_conv_chunking_factor(1)
        print("  attention           : rel_pos_local_attn [128,128] (long-audio mode)")

    return model


# --------------------------------------------------------------------------- #
# 3. Transcription
# --------------------------------------------------------------------------- #
def _unwrap(out):
    """NeMo returns either [Hypothesis] or (best, all_beams) depending on version."""
    if isinstance(out, tuple):
        out = out[0]
    if out and isinstance(out[0], list):  # nested best-hyp list
        out = out[0]
    return out


def _select_decoder(model, decoder_type: str) -> None:
    try:
        model.change_decoding_strategy(decoder_type=decoder_type)
    except TypeError:
        # Older signature: change_decoding_strategy(decoding_cfg, decoder_type=...)
        model.change_decoding_strategy(None, decoder_type=decoder_type)


def _stamped(hyp, key: str) -> list[dict]:
    """Pull `key`-level timestamps out of a Hypothesis, tolerating key variations."""
    ts = getattr(hyp, "timestamp", None)
    if not isinstance(ts, dict):
        return []
    out = []
    for s in ts.get(key) or []:
        if not isinstance(s, dict):
            continue
        text = s.get(key) or s.get("word") or s.get("char") or ""
        start, end = s.get("start"), s.get("end")
        if start is None:  # older builds expose frame offsets only
            start = s.get("start_offset", 0) * 0.08  # 8x subsampling @ 10 ms
            end = s.get("end_offset", 0) * 0.08
        out.append({"start": float(start), "end": float(end), "text": str(text).strip()})
    return out


def sentences_from_words(words: list[dict]) -> list[dict]:
    """Group word timestamps into sentences, splitting on Armenian terminators."""
    out, buf = [], []
    for w in words:
        if not w["text"]:
            continue
        buf.append(w)
        if w["text"][-1] in TERMINATORS:
            out.append({"start": buf[0]["start"], "end": buf[-1]["end"],
                        "text": " ".join(b["text"] for b in buf)})
            buf = []
    if buf:
        out.append({"start": buf[0]["start"], "end": buf[-1]["end"],
                    "text": " ".join(b["text"] for b in buf)})
    return out


def split_on_silence(wav: Path, out_dir: Path) -> list[dict]:
    """Cut `wav` at silences into <=CHUNK_MAX_SEC pieces. Returns chunk metadata."""
    import librosa
    import soundfile as sf

    y, sr = librosa.load(str(wav), sr=None, mono=True)
    voiced = librosa.effects.split(y, top_db=SILENCE_TOP_DB,
                                   frame_length=2048, hop_length=512)

    # Greedily merge consecutive voiced runs until the span would exceed the cap.
    spans: list[list[int]] = []
    for s, e in voiced:
        if spans and (e - spans[-1][0]) / sr <= CHUNK_MAX_SEC:
            spans[-1][1] = int(e)
        else:
            spans.append([int(s), int(e)])

    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("chunk_*.wav"):
        stale.unlink()

    pad = int(CHUNK_PAD_SEC * sr)
    chunks = []
    for i, (s, e) in enumerate(spans):
        s, e = max(0, s - pad), min(len(y), e + pad)
        path = out_dir / f"chunk_{i:03d}.wav"
        sf.write(str(path), y[s:e], sr, subtype="PCM_16")
        chunks.append({"path": str(path), "offset": s / sr, "dur": (e - s) / sr})

    total = sum(c["dur"] for c in chunks)
    print(f"  chunks              : {len(chunks)} "
          f"(speech {total:.1f}s of {len(y) / sr:.1f}s, "
          f"mean {total / max(1, len(chunks)):.1f}s, "
          f"max {max(c['dur'] for c in chunks):.1f}s)")
    return chunks


def _report(label: str, decoder: str, text: str, segs: list[dict],
            elapsed: float, duration: float) -> dict:
    print(f"\n  wall clock          : {elapsed:.1f}s")
    print(f"  real-time factor    : {elapsed / duration:.3f}x "
          f"({duration / elapsed:.1f}x faster than real time)")
    print(f"  words / characters  : {len(text.split())} / {len(text)}")
    print(f"  sentences           : {len(segs)}")

    print("\n--- TRANSCRIPT " + "-" * 57)
    for s in segs:
        if s["text"]:
            print(f"  [{s['start']:7.2f} - {s['end']:7.2f}]  {s['text']}")
    if not segs:
        print(text)
    print("-" * 72)

    return {
        "mode": label, "decoder": decoder, "text": text, "segments": segs,
        "seconds": round(elapsed, 2), "rtf": round(elapsed / duration, 4),
        "n_words": len(text.split()), "n_chars": len(text), "n_segments": len(segs),
    }


def run_full(model, wav: Path, decoder_type: str, duration: float) -> dict:
    """One forward pass over the entire recording."""
    banner(f"FULL-FILE PASS -- decoder = {decoder_type.upper()}")
    _select_decoder(model, decoder_type)

    import torch

    t0 = time.perf_counter()
    with torch.inference_mode():
        try:
            out = model.transcribe([str(wav)], batch_size=1, timestamps=True,
                                   return_hypotheses=True)
        except Exception as exc:  # timestamps unsupported for this head/version
            print(f"  (timestamps unavailable: {type(exc).__name__}: {exc})")
            out = model.transcribe([str(wav)], batch_size=1)
    elapsed = time.perf_counter() - t0

    hyp = _unwrap(out)[0]
    text = hyp.text if hasattr(hyp, "text") else str(hyp)
    words = _stamped(hyp, "word")
    segs = sentences_from_words(words) if words else _stamped(hyp, "segment")
    return _report("full", decoder_type, text, segs, elapsed, duration)


def run_chunked(model, chunks: list[dict], decoder_type: str, duration: float,
                batch_size: int = 4) -> dict:
    """Transcribe silence-split chunks as a batch and stitch the results back."""
    banner(f"CHUNKED PASS -- decoder = {decoder_type.upper()}")
    _select_decoder(model, decoder_type)

    import torch

    paths = [c["path"] for c in chunks]
    t0 = time.perf_counter()
    with torch.inference_mode():
        out = model.transcribe(paths, batch_size=batch_size)
    elapsed = time.perf_counter() - t0

    hyps = _unwrap(out)
    segs = []
    for c, h in zip(chunks, hyps):
        txt = (h.text if hasattr(h, "text") else str(h)).strip()
        if txt:
            segs.append({"start": round(c["offset"], 2),
                         "end": round(c["offset"] + c["dur"], 2), "text": txt})
    text = " ".join(s["text"] for s in segs)
    return _report("chunked", decoder_type, text, segs, elapsed, duration)


def _clean_segments(segs: list[dict]) -> list[dict]:
    """Drop content-free segments (silence-only chunks decode to stray punctuation)
    and force timestamps monotonic -- chunk padding makes neighbours overlap."""
    out: list[dict] = []
    for s in segs:
        if sum(ch.isalpha() for ch in s["text"]) < 2:
            continue
        s = dict(s)
        if out and s["start"] < out[-1]["end"]:
            s["start"] = out[-1]["end"]
        if s["end"] <= s["start"]:
            s["end"] = s["start"] + 0.5
        out.append(s)
    return out


def _srt_time(t: float) -> str:
    ms = round(t * 1000)
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_transcripts(result: dict, stem: Path) -> tuple[Path, Path, int]:
    segs = _clean_segments(result["segments"])
    # NB: not with_suffix() -- these filenames contain dots ("...115644_0").
    txt = stem.with_name(stem.name + ".txt")
    txt.write_text("\n".join(s["text"] for s in segs) + "\n", encoding="utf-8")
    srt = stem.with_name(stem.name + ".srt")
    srt.write_text(
        "\n".join(f"{i}\n{_srt_time(s['start'])} --> {_srt_time(s['end'])}\n"
                  f"{s['text']}\n" for i, s in enumerate(segs, 1)),
        encoding="utf-8",
    )
    return txt, srt, len(segs)


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", type=Path, default=DEFAULT_AUDIO)
    ap.add_argument("--decoders", default="rnnt,ctc",
                    help="comma-separated: rnnt,ctc")
    ap.add_argument("--mode", default="both", choices=["full", "chunked", "both"])
    ap.add_argument("--local-attn", action="store_true",
                    help="limited-context attention (lower memory on long audio)")
    ap.add_argument("--out", type=Path, default=HERE / "stt_hy_results.json")
    args = ap.parse_args()

    if not args.audio.exists():
        print(f"audio not found: {args.audio}", file=sys.stderr)
        return 1

    work = Path(os.environ.get("TEMP", ".")) / "stt_hy_work"
    work.mkdir(parents=True, exist_ok=True)
    wav = work / (args.audio.stem + "_16k.wav")
    stats = prepare_audio(args.audio, wav)

    decoders = [d.strip().lower() for d in args.decoders.split(",") if d.strip()]
    modes = ["full", "chunked"] if args.mode == "both" else [args.mode]

    chunks = []
    if "chunked" in modes:
        banner("SILENCE SPLIT")
        chunks = split_on_silence(wav, work / "chunks")

    model = load_model(args.local_attn)

    duration = stats["duration_sec"]
    results = []
    for mode in modes:
        for dec in decoders:
            try:
                if mode == "full":
                    results.append(run_full(model, wav, dec, duration))
                else:
                    results.append(run_chunked(model, chunks, dec, duration))
            except Exception as exc:
                import traceback
                print(f"\n  {mode}/{dec.upper()} FAILED: {type(exc).__name__}: {exc}")
                traceback.print_exc()

    banner("SUMMARY")
    print(f"  {'mode':<9}{'decoder':<9}{'sec':>7}{'RTF':>8}{'words':>8}"
          f"{'chars':>8}{'sents':>7}")
    for r in results:
        print(f"  {r['mode']:<9}{r['decoder']:<9}{r['seconds']:>7.1f}{r['rtf']:>8.3f}"
              f"{r['n_words']:>8}{r['n_chars']:>8}{r['n_segments']:>7}")

    import difflib

    def agree(a: dict, b: dict) -> float:
        return difflib.SequenceMatcher(
            None, a["text"].split(), b["text"].split()).ratio() * 100

    print("\n  Word-level agreement (a proxy for confidence -- the two heads share an\n"
          "  encoder, so they only diverge where the acoustics are ambiguous):")
    by_key = {(r["mode"], r["decoder"]): r for r in results}
    for mode in modes:
        if (mode, "rnnt") in by_key and (mode, "ctc") in by_key:
            print(f"    {mode:<9} RNNT vs CTC : "
                  f"{agree(by_key[(mode, 'rnnt')], by_key[(mode, 'ctc')]):.1f}%")
    for dec in decoders:
        if ("full", dec) in by_key and ("chunked", dec) in by_key:
            print(f"    {dec:<9} full vs chunked : "
                  f"{agree(by_key[('full', dec)], by_key[('chunked', dec)]):.1f}%")

    args.out.write_text(
        json.dumps({"audio": stats, "model": MODEL_NAME, "results": results},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n  results written to {args.out}")

    # Export the configuration that reads best on this material.
    pick = by_key.get(("chunked", "rnnt")) or (results[0] if results else None)
    if pick:
        stem = args.out.with_name(f"{args.audio.stem}.{pick['mode']}_{pick['decoder']}")
        txt, srt, n = write_transcripts(pick, stem)
        print(f"  transcript ({pick['mode']}/{pick['decoder']}, {n} cues) ->"
              f"\n    {txt}\n    {srt}")

    print(f"\n  CAVEAT: this recording is {stats['native_sample_rate']} Hz telephony "
          f"audio upsampled to {TARGET_SR} Hz.\n"
          "  The model's published WER (9.9% on Common Voice, 12.3% on FLEURS) comes from\n"
          "  clean 16 kHz read speech. Expect materially worse accuracy here.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
