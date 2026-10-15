from __future__ import annotations

import asyncio
import base64
import ipaddress
import os
import re
import socket
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import edge_tts
import requests
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse
from PIL import Image, ImageDraw, ImageFont, ImageOps
from pydantic import BaseModel, Field, HttpUrl

APP_VERSION = "0.3.0"
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "/tmp/flowinsoul-output"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MAX_ASSET_BYTES = int(os.getenv("MAX_ASSET_BYTES", str(30 * 1024 * 1024)))
MAX_ASSET_COUNT = 40

app = FastAPI(title="FlowinSoul Render Engine", version=APP_VERSION)


def require_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    expected = os.getenv("FLOWINSOUL_API_KEY")
    if not expected:
        raise HTTPException(status_code=503, detail="FLOWINSOUL_API_KEY is not configured")
    if x_api_key is None or x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid API key")


class TTSRequest(BaseModel):
    text: str = Field(min_length=1, max_length=100000)
    voice: str = "zh-TW-YunJheNeural"


class SRTRequest(BaseModel):
    text: str = Field(min_length=1, max_length=100000)
    duration_seconds: float = Field(gt=0, le=7200)
    max_chars_per_cue: int = Field(default=18, ge=8, le=42)


class ShortNarration(BaseModel):
    title: str = Field(min_length=1, max_length=160)
    text: str = Field(min_length=1, max_length=12000)


class ProductionNarrationRequest(BaseModel):
    project_id: str = Field(min_length=1, max_length=80)
    long_text: str = Field(min_length=1, max_length=100000)
    shorts: list[ShortNarration] = Field(min_length=1, max_length=5)
    long_voice: str = "zh-TW-YunJheNeural"
    short_voice: str = "zh-TW-HsiaoChenNeural"


class CoverVariant(BaseModel):
    image_url: HttpUrl
    title: str = Field(min_length=1, max_length=80)
    subtitle: str = Field(default="", max_length=100)


class CoverRequest(BaseModel):
    variants: list[CoverVariant] = Field(min_length=1, max_length=3)
    aspect_ratio: str = "16:9"


class VideoRequest(BaseModel):
    audio_url: HttpUrl
    image_urls: list[HttpUrl] = Field(min_length=1, max_length=MAX_ASSET_COUNT)
    srt_text: str = Field(default="", max_length=500000)
    output_name: str = Field(default="video.mp4", max_length=100)
    background_music_url: Optional[HttpUrl] = None
    background_music_volume: float = Field(default=0.08, ge=0, le=0.5)
    aspect_ratio: str = "16:9"


class UploadAssetRequest(BaseModel):
    filename: str = Field(min_length=1, max_length=100)
    mime_type: str = Field(min_length=1, max_length=100)
    data_base64: str = Field(min_length=1, max_length=45_000_000)


def _download(url: str, destination: Path) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https":
        raise HTTPException(status_code=400, detail="Asset URLs must use HTTPS")
    try:
        # Refuse private/local destinations to prevent the service being used for SSRF.
        for info in socket.getaddrinfo(parsed.hostname, 443, type=socket.SOCK_STREAM):
            address = ipaddress.ip_address(info[4][0])
            if not address.is_global:
                raise HTTPException(status_code=400, detail="Private or local asset URL is not allowed")
        with requests.get(url, stream=True, timeout=(10, 60)) as response:
            response.raise_for_status()
            size = 0
            with destination.open("wb") as out:
                for chunk in response.iter_content(1024 * 256):
                    if not chunk:
                        continue
                    size += len(chunk)
                    if size > MAX_ASSET_BYTES:
                        raise HTTPException(status_code=413, detail="Asset exceeds size limit")
                    out.write(chunk)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not download asset: {exc}") from exc


def _font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def _wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    lines: list[str] = []
    line = ""
    for char in text:
        if char == "\n":
            lines.append(line)
            line = ""
        elif not line or draw.textlength(line + char, font=font) <= max_width:
            line += char
        else:
            lines.append(line)
            line = char
    if line:
        lines.append(line)
    return lines[:3]


def _make_cover(source: Path, target: Path, title: str, subtitle: str) -> None:
    image = Image.open(source).convert("RGB")
    image = ImageOps.fit(image, (1280, 720), method=Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(image, "RGBA")
    # A dark lower gradient improves legibility while preserving the generated artwork.
    for y in range(390, 720):
        alpha = int(185 * (y - 390) / 330)
        draw.rectangle((0, y, 1280, y + 1), fill=(0, 0, 0, alpha))
    title_font = _font(76)
    subtitle_font = _font(36)
    lines = _wrap_text(draw, title, title_font, 1160)
    line_h = 92
    y = max(430, 650 - line_h * len(lines) - (58 if subtitle else 0))
    for line in lines:
        draw.text((62 + 4, y + 4), line, font=title_font, fill=(0, 0, 0, 230), stroke_width=4, stroke_fill=(0, 0, 0, 230))
        draw.text((62, y), line, font=title_font, fill=(255, 226, 112, 255), stroke_width=2, stroke_fill=(40, 25, 5, 255))
        y += line_h
    if subtitle:
        draw.text((66, min(y + 4, 678)), subtitle, font=subtitle_font, fill=(255, 255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0, 220))
    image.save(target, "JPEG", quality=92, optimize=True)


def _srt_timestamp(seconds: float) -> str:
    milliseconds = max(0, round(seconds * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"


def _make_srt(text: str, duration: float, max_chars: int) -> str:
    sentences = [s.strip() for s in re.split(r"(?<=[。！？!?；;])\s*", text) if s.strip()]
    chunks: list[str] = []
    for sentence in sentences:
        chars = list(sentence)
        while len(chars) > max_chars:
            cut = max_chars
            for i in range(max_chars, max(0, max_chars // 2), -1):
                if chars[i - 1] in "，、：：, ":
                    cut = i
                    break
            chunks.append("".join(chars[:cut]).strip())
            chars = chars[cut:]
        if chars:
            chunks.append("".join(chars).strip())
    chunks = [chunk for chunk in chunks if chunk]
    if not chunks:
        return ""
    weights = [max(1, len(re.sub(r"\s", "", chunk))) for chunk in chunks]
    total = sum(weights)
    cursor = 0.0
    entries = []
    for index, (chunk, weight) in enumerate(zip(chunks, weights), start=1):
        start = cursor
        end = duration if index == len(chunks) else cursor + duration * weight / total
        entries.append(f"{index}\n{_srt_timestamp(start)} --> {_srt_timestamp(end)}\n{chunk}")
        cursor = end
    return "\n\n".join(entries) + "\n"


@app.get("/")
@app.get("/health")
def health():
    return {"status": "ok", "version": APP_VERSION}


@app.post("/generate-tts", dependencies=[Depends(require_api_key)])
async def generate_tts(req: TTSRequest, request: Request):
    filename = f"{uuid.uuid4().hex}.mp3"
    path = OUTPUT_DIR / filename
    try:
        await edge_tts.Communicate(req.text, req.voice).save(str(path))
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"TTS generation failed: {exc}") from exc
    try:
        probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(path)], check=True, capture_output=True, text=True, timeout=30)
        duration = float(probe.stdout.strip())
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Could not read generated audio duration: {exc}") from exc
    return {"status": "success", "filename": filename, "duration_seconds": duration, "download_url": str(request.base_url).rstrip("/") + f"/files/{filename}"}


@app.post("/generate-srt", dependencies=[Depends(require_api_key)])
def generate_srt(req: SRTRequest):
    return {"status": "success", "duration_seconds": req.duration_seconds, "srt_text": _make_srt(req.text, req.duration_seconds, req.max_chars_per_cue)}


@app.post("/prepare-narration", dependencies=[Depends(require_api_key)])
async def prepare_narration(req: ProductionNarrationRequest, request: Request):
    """Create narration MP3 and editable zh-TW SRT for one long video and up to five Shorts."""
    stem = re.sub(r"[^A-Za-z0-9_-]+", "-", req.project_id).strip("-")[:60] or uuid.uuid4().hex
    jobs = [(f"{stem}_LONG", req.long_text, req.long_voice)] + [
        (f"{stem}_SHORT_{i:02}", short.text, req.short_voice)
        for i, short in enumerate(req.shorts, start=1)
    ]
    outputs = []
    base_url = str(request.base_url).rstrip("/")
    for filename_stem, text, voice in jobs:
        audio_path = OUTPUT_DIR / f"{filename_stem}_{uuid.uuid4().hex[:8]}.mp3"
        srt_path = OUTPUT_DIR / f"{filename_stem}_{uuid.uuid4().hex[:8]}.srt"
        try:
            await edge_tts.Communicate(text, voice).save(str(audio_path))
            probe = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(audio_path)],
                check=True, capture_output=True, text=True, timeout=60,
            )
            duration = float(probe.stdout.strip())
            srt_path.write_text(_make_srt(text, duration, 18), encoding="utf-8")
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Narration preparation failed for {filename_stem}: {exc}") from exc
        outputs.append({
            "name": filename_stem,
            "voice": voice,
            "duration_seconds": duration,
            "audio_url": f"{base_url}/files/{audio_path.name}",
            "srt_url": f"{base_url}/files/{srt_path.name}",
        })
    return {"status": "success", "project_id": req.project_id, "assets": outputs}


@app.post("/upload-asset", dependencies=[Depends(require_api_key)])
def upload_asset(req: UploadAssetRequest, request: Request):
    """Store a generated image/audio asset so the render endpoints can fetch it."""
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "-", Path(req.filename).name).strip(".-") or "asset"
    try:
        raw = base64.b64decode(req.data_base64, validate=True)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="data_base64 must be valid base64") from exc
    if not raw or len(raw) > MAX_ASSET_BYTES:
        raise HTTPException(status_code=413, detail="Asset is empty or exceeds size limit")
    allowed = {"image/jpeg": ".jpg", "image/png": ".png", "audio/mpeg": ".mp3", "audio/wav": ".wav", "audio/mp4": ".m4a"}
    suffix = allowed.get(req.mime_type.lower())
    if suffix is None:
        raise HTTPException(status_code=415, detail="Unsupported asset type")
    filename = f"{uuid.uuid4().hex}-{safe_name[:60]}"
    if Path(filename).suffix.lower() != suffix:
        filename += suffix
    path = OUTPUT_DIR / filename
    path.write_bytes(raw)
    return {"status": "success", "filename": filename, "mime_type": req.mime_type,
            "size_bytes": len(raw), "download_url": str(request.base_url).rstrip("/") + f"/files/{filename}"}


@app.post("/render-cover", dependencies=[Depends(require_api_key)])
def render_cover(req: CoverRequest, request: Request):
    if req.aspect_ratio != "16:9":
        raise HTTPException(status_code=400, detail="Only 16:9 covers are supported")
    results = []
    with tempfile.TemporaryDirectory() as temp_dir:
        temp = Path(temp_dir)
        for index, variant in enumerate(req.variants, start=1):
            source = temp / f"source-{index}"
            filename = f"cover-{uuid.uuid4().hex}.jpg"
            destination = OUTPUT_DIR / filename
            _download(str(variant.image_url), source)
            try:
                _make_cover(source, destination, variant.title, variant.subtitle)
            except Exception as exc:
                raise HTTPException(status_code=400, detail=f"Cover processing failed: {exc}") from exc
            results.append({"variant": index, "filename": filename, "title": variant.title,
                            "download_url": str(request.base_url).rstrip("/") + f"/files/{filename}"})
    return {"status": "success", "covers": results}


@app.post("/render-video", dependencies=[Depends(require_api_key)])
def render_video(req: VideoRequest, request: Request):
    if req.aspect_ratio not in {"16:9", "9:16"}:
        raise HTTPException(status_code=400, detail="aspect_ratio must be 16:9 or 9:16")
    safe_output = re.sub(r"[^A-Za-z0-9._-]+", "-", Path(req.output_name).name).strip(".-") or "video.mp4"
    if not safe_output.lower().endswith(".mp4"):
        safe_output += ".mp4"
    filename = f"{uuid.uuid4().hex[:10]}-{safe_output[:80]}"
    output = OUTPUT_DIR / filename
    with tempfile.TemporaryDirectory() as temp_dir:
        temp = Path(temp_dir)
        audio = temp / "narration.mp3"
        _download(str(req.audio_url), audio)
        music = None
        if req.background_music_url:
            music = temp / "background-music"
            _download(str(req.background_music_url), music)
        images: list[Path] = []
        for i, url in enumerate(req.image_urls):
            image_path = temp / f"scene-{i:03}.jpg"
            _download(str(url), image_path)
            # Normalize to consistent 1080p landscape or vertical frames.
            normalized = temp / f"frame-{i:03}.jpg"
            try:
                frame_size = (1080, 1920) if req.aspect_ratio == "9:16" else (1920, 1080)
                ImageOps.fit(Image.open(image_path).convert("RGB"), frame_size, method=Image.Resampling.LANCZOS).save(normalized, quality=90)
            except Exception as exc:
                raise HTTPException(status_code=400, detail=f"Invalid scene image: {exc}") from exc
            images.append(normalized)

        duration_probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(audio)],
            check=True, capture_output=True, text=True, timeout=60,
        )
        duration = float(duration_probe.stdout.strip())
        if duration <= 0:
            raise HTTPException(status_code=400, detail="Audio duration is invalid")
        per_scene = duration / len(images)
        concat_file = temp / "scenes.txt"
        with concat_file.open("w", encoding="utf-8") as handle:
            for frame in images:
                handle.write(f"file '{frame.as_posix()}'\n")
                handle.write(f"duration {per_scene:.3f}\n")
            handle.write(f"file '{images[-1].as_posix()}'\n")
        subtitle_file = temp / "captions.srt"
        subtitle_file.write_text(req.srt_text, encoding="utf-8")
        command = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_file), "-i", str(audio)]
        if music:
            command += ["-stream_loop", "-1", "-i", str(music)]
        if req.srt_text.strip():
            command += ["-vf", f"subtitles={subtitle_file}:force_style='FontName=Noto Sans CJK TC,FontSize=22,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,Outline=2,Shadow=1,MarginV=48'"]
        if music:
            volume = req.background_music_volume
            command += ["-filter_complex", f"[1:a]volume=1.0[voice];[2:a]volume={volume}[music];[voice][music]amix=inputs=2:duration=first:dropout_transition=3[aout]", "-map", "0:v:0", "-map", "[aout]"]
        else:
            command += ["-map", "0:v:0", "-map", "1:a:0"]
        frame_size = "1080:1920" if req.aspect_ratio == "9:16" else "1920:1080"
        command += ["-t", f"{duration:.3f}", "-r", "30", "-s", frame_size, "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(output)]
        try:
            subprocess.run(command, check=True, capture_output=True, text=True, timeout=max(900, int(duration * 3)))
        except subprocess.TimeoutExpired as exc:
            raise HTTPException(status_code=504, detail="FFmpeg rendering timed out") from exc
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or "FFmpeg failed")[-3000:]
            raise HTTPException(status_code=500, detail=detail) from exc
    return {"status": "success", "filename": filename, "duration_seconds": duration,
            "download_url": str(request.base_url).rstrip("/") + f"/files/{filename}"}


@app.get("/files/{filename}")
def get_file(filename: str):
    if not re.fullmatch(r"[A-Za-z0-9._-]+", filename):
        raise HTTPException(status_code=400, detail="Invalid filename")
    path = OUTPUT_DIR / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    media_types = {".mp4": "video/mp4", ".mp3": "audio/mpeg", ".m4a": "audio/mp4",
                   ".wav": "audio/wav", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}
    media_type = media_types.get(path.suffix.lower(), "application/octet-stream")
    return FileResponse(path, media_type=media_type, filename=filename)
