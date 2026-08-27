"""
Local web UI for browsing call recordings and transcribing them with
nvidia/stt_hy_fastconformer_hybrid_large_pc.

    E:\\nemo_stt_hy\\venv\\Scripts\\python.exe stt_server.py
    E:\\nemo_stt_hy\\venv\\Scripts\\python.exe stt_server.py --dir D:\\calls --port 8080

Then open http://127.0.0.1:8000

Runs entirely on this machine -- the model, the audio and the transcripts never
leave it. The heavy lifting is imported from test_stt_model.py so both entry
points share one implementation.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import threading
import time
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse, Response
from pydantic import BaseModel

import test_stt_model as engine

HERE = Path(__file__).resolve().parent
AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".opus", ".aac"}
MODES = ("chunked", "full")
DECODERS = ("rnnt", "ctc")

# Populated by main().
AUDIO_DIR = HERE
CACHE_DIR = HERE / ".transcripts"
WORK_DIR = Path(engine.CACHE_ROOT) / "ui_work"

app = FastAPI(title="Armenian Call Transcriber")

# The model is a single shared object whose decoding strategy we mutate per
# request, so all inference is serialised behind this lock.
_model = None
_model_lock = threading.Lock()
_state = {"loading": False, "error": None, "busy": None}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def safe_audio(name: str) -> Path:
    """Resolve a client-supplied filename to a real file inside AUDIO_DIR."""
    path = (AUDIO_DIR / Path(name).name).resolve()
    if path.parent != AUDIO_DIR.resolve() or not path.is_file():
        raise HTTPException(404, f"no such recording: {name}")
    if path.suffix.lower() not in AUDIO_EXTS:
        raise HTTPException(400, f"not an audio file: {name}")
    return path


def cache_path(name: str, mode: str, decoder: str) -> Path:
    return CACHE_DIR / f"{name}.{mode}_{decoder}.json"


def validate(mode: str, decoder: str) -> None:
    if mode not in MODES:
        raise HTTPException(400, f"mode must be one of {MODES}")
    if decoder not in DECODERS:
        raise HTTPException(400, f"decoder must be one of {DECODERS}")


def get_model():
    global _model
    with _model_lock:
        if _model is None:
            _state["loading"] = True
            try:
                _model = engine.load_model(local_attn=False)
            except Exception as exc:
                _state["error"] = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                _state["loading"] = False
    return _model


def prepare_16k(src: Path) -> tuple[Path, float]:
    """Cached 8k/44k -> 16 kHz mono WAV conversion. Quiet counterpart of
    engine.prepare_audio(), which is CLI-oriented and prints a report."""
    import librosa
    import soundfile as sf

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    dst = WORK_DIR / (src.stem + "_16k.wav")
    if not dst.exists() or dst.stat().st_mtime < src.stat().st_mtime:
        y, sr = librosa.load(str(src), sr=engine.TARGET_SR, mono=True)
        sf.write(str(dst), y, sr, subtype="PCM_16")
    info = sf.info(str(dst))
    return dst, info.duration


def speech_spans(wav: Path) -> list[list[float]]:
    """Voiced regions, for the timeline strip. Cheap: no model involved."""
    import librosa

    y, sr = librosa.load(str(wav), sr=None, mono=True)
    runs = librosa.effects.split(y, top_db=engine.SILENCE_TOP_DB,
                                 frame_length=2048, hop_length=512)
    return [[round(s / sr, 2), round(e / sr, 2)] for s, e in runs]


def transcribe_sync(path: Path, mode: str, decoder: str) -> dict:
    """Blocking. Called via asyncio.to_thread so the event loop stays free."""
    import torch

    model = get_model()
    wav, duration = prepare_16k(path)

    with _model_lock:
        engine._select_decoder(model, decoder)
        t0 = time.perf_counter()
        with torch.inference_mode():
            if mode == "chunked":
                chunks = engine.split_on_silence(wav, WORK_DIR / f"chunks_{path.stem}")
                hyps = engine._unwrap(
                    model.transcribe([c["path"] for c in chunks], batch_size=4)
                )
                segs = []
                for c, h in zip(chunks, hyps):
                    txt = (h.text if hasattr(h, "text") else str(h)).strip()
                    if txt:
                        segs.append({"start": round(c["offset"], 2),
                                     "end": round(c["offset"] + c["dur"], 2),
                                     "text": txt})
            else:
                try:
                    out = model.transcribe([str(wav)], batch_size=1,
                                           timestamps=True, return_hypotheses=True)
                except Exception:
                    out = model.transcribe([str(wav)], batch_size=1)
                hyp = engine._unwrap(out)[0]
                words = engine._stamped(hyp, "word")
                segs = (engine.sentences_from_words(words) if words
                        else engine._stamped(hyp, "segment"))
                if not segs:
                    text = hyp.text if hasattr(hyp, "text") else str(hyp)
                    segs = [{"start": 0.0, "end": duration, "text": text}]
        elapsed = time.perf_counter() - t0

    segs = engine._clean_segments(segs)
    result = {
        "name": path.name,
        "mode": mode,
        "decoder": decoder,
        "duration": round(duration, 2),
        "segments": segs,
        "text": " ".join(s["text"] for s in segs),
        "seconds": round(elapsed, 2),
        "rtf": round(elapsed / duration, 4) if duration else None,
        "speech": speech_spans(wav),
    }
    result["n_words"] = len(result["text"].split())

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path(path.name, mode, decoder).write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


# --------------------------------------------------------------------------- #
# API
# --------------------------------------------------------------------------- #
@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    html = HERE / "stt_ui.html"
    if not html.exists():
        raise HTTPException(500, "stt_ui.html is missing next to stt_server.py")
    return HTMLResponse(html.read_text(encoding="utf-8"))


@app.get("/api/status")
def status() -> dict:
    return {
        "dir": str(AUDIO_DIR),
        "model": engine.MODEL_NAME,
        "loaded": _model is not None,
        "loading": _state["loading"],
        "busy": _state["busy"],
        "error": _state["error"],
    }


@app.get("/api/recordings")
def recordings() -> list[dict]:
    import soundfile as sf

    done: dict[str, list[str]] = {}
    if CACHE_DIR.is_dir():
        for f in CACHE_DIR.glob("*.json"):
            m = re.match(r"^(?P<name>.+)\.(?P<key>(chunked|full)_(rnnt|ctc))$", f.stem)
            if m:
                done.setdefault(m["name"], []).append(m["key"])

    out = []
    for path in sorted(AUDIO_DIR.iterdir()):
        if not path.is_file() or path.suffix.lower() not in AUDIO_EXTS:
            continue
        try:
            info = sf.info(str(path))
            duration, sample_rate, channels = info.duration, info.samplerate, info.channels
        except Exception:
            duration = sample_rate = channels = None
        stat = path.stat()
        out.append({
            "name": path.name,
            "size": stat.st_size,
            "mtime": stat.st_mtime,
            "duration": round(duration, 2) if duration else None,
            "sample_rate": sample_rate,
            "channels": channels,
            "transcripts": sorted(done.get(path.name, [])),
        })
    return out


@app.get("/api/audio/{name}")
def audio(name: str) -> Response:
    # Starlette's FileResponse honours Range requests, so scrubbing works
    # without re-downloading the file.
    return FileResponse(safe_audio(name))


@app.get("/api/transcript/{name}")
def transcript(name: str, mode: str = Query("chunked"),
               decoder: str = Query("rnnt")) -> JSONResponse:
    validate(mode, decoder)
    safe_audio(name)
    cached = cache_path(Path(name).name, mode, decoder)
    if not cached.exists():
        raise HTTPException(404, "not transcribed yet")
    return JSONResponse(json.loads(cached.read_text(encoding="utf-8")))


class TranscribeRequest(BaseModel):
    name: str
    mode: str = "chunked"
    decoder: str = "rnnt"
    force: bool = False


@app.post("/api/transcribe")
async def transcribe(req: TranscribeRequest) -> JSONResponse:
    validate(req.mode, req.decoder)
    path = safe_audio(req.name)
    cached = cache_path(path.name, req.mode, req.decoder)
    if cached.exists() and not req.force:
        return JSONResponse(json.loads(cached.read_text(encoding="utf-8")))

    _state["busy"] = f"{path.name} ({req.mode}/{req.decoder})"
    try:
        result = await asyncio.to_thread(transcribe_sync, path, req.mode, req.decoder)
    except Exception as exc:
        raise HTTPException(500, f"{type(exc).__name__}: {exc}") from exc
    finally:
        _state["busy"] = None
    return JSONResponse(result)


@app.get("/api/export/{name}")
def export(name: str, mode: str = Query("chunked"), decoder: str = Query("rnnt"),
           fmt: str = Query("srt")) -> PlainTextResponse:
    validate(mode, decoder)
    safe_audio(name)
    cached = cache_path(Path(name).name, mode, decoder)
    if not cached.exists():
        raise HTTPException(404, "not transcribed yet")
    data = json.loads(cached.read_text(encoding="utf-8"))
    segs = data["segments"]

    if fmt == "txt":
        body = "\n".join(s["text"] for s in segs) + "\n"
    elif fmt == "srt":
        body = "\n".join(
            f"{i}\n{engine._srt_time(s['start'])} --> {engine._srt_time(s['end'])}\n"
            f"{s['text']}\n" for i, s in enumerate(segs, 1))
    else:
        raise HTTPException(400, "fmt must be srt or txt")

    stem = Path(name).stem
    return PlainTextResponse(body, media_type="text/plain; charset=utf-8", headers={
        "Content-Disposition": f'attachment; filename="{stem}.{mode}_{decoder}.{fmt}"'})


# --------------------------------------------------------------------------- #
def main() -> int:
    global AUDIO_DIR, CACHE_DIR

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", type=Path, default=HERE,
                    help="folder of recordings to browse (default: this folder)")
    ap.add_argument("--cache-dir", type=Path, default=None,
                    help="where transcripts are stored (default: <dir>/.transcripts)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--preload", action="store_true",
                    help="load the model at startup instead of on first transcription")
    args = ap.parse_args()

    AUDIO_DIR = args.dir.resolve()
    if not AUDIO_DIR.is_dir():
        print(f"not a directory: {AUDIO_DIR}")
        return 1
    CACHE_DIR = (args.cache_dir or AUDIO_DIR / ".transcripts").resolve()

    n = sum(1 for p in AUDIO_DIR.iterdir()
            if p.is_file() and p.suffix.lower() in AUDIO_EXTS)
    print(f"  recordings : {AUDIO_DIR}  ({n} audio file{'s' if n != 1 else ''})")
    print(f"  transcripts: {CACHE_DIR}")
    print(f"  open       : http://{args.host}:{args.port}\n")

    if args.preload:
        get_model()

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
