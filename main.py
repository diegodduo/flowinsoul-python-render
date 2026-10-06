from __future__ import annotations

import asyncio
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

APP_VERSION = "0.2.0"
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
    return {"status": "success", "filename": filename, "download_url": str(request.base_url).rstrip("/") + f"/files/{filename}"}


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
    filename = f"video-{uuid.uuid4().hex}.mp4"
    output = OUTPUT_DIR / filename
    with tempfile.TemporaryDirectory() as temp_dir:
        temp = Path(temp_dir)
        audio = temp / "narration.mp3"
        _download(str(req.audio_url), audio)
        images: list[Path] = []
        for i, url in enumerate(req.image_urls):
            image_path = temp / f"scene-{i:03}.jpg"
            _download(str(url), image_path)
            # Normalize to a consistent 1080p landscape frame.
            normalized = temp / f"frame-{i:03}.jpg"
            try:
                ImageOps.fit(Image.open(image_path).convert("RGB"), (1920, 1080), method=Image.Resampling.LANCZOS).save(normalized, quality=90)
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
        if req.srt_text.strip():
            command += ["-vf", f"subtitles={subtitle_file}:force_style='FontName=Noto Sans CJK TC,FontSize=22,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,Outline=2,Shadow=1,MarginV=48'"]
        command += ["-map", "0:v:0", "-map", "1:a:0", "-t", f"{duration:.3f}", "-r", "30", "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(output)]
        try:
            subprocess.run(command, check=True, capture_output=True, text=True, timeout=max(900, int(duration * 3)))
        except subprocess.TimeoutExpired as exc:
            raise HTTPException(status_code=504, detail="FFmpeg rendering timed out") from exc
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or "FFmpeg failed")[-3000:]
            raise HTTPException(status_code=500, detail=detail) from exc
    return {"status": "success", "filename": filename, "duration_seconds": duration,
            "download_url": str(request.base_url).rstrip("/") + f"/files/{filename}"}


@app.get("/files/{filename}", dependencies=[Depends(require_api_key)])
def get_file(filename: str):
    if not re.fullmatch(r"[A-Za-z0-9._-]+", filename):
        raise HTTPException(status_code=400, detail="Invalid filename")
    path = OUTPUT_DIR / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    media_type = "video/mp4" if path.suffix == ".mp4" else "audio/mpeg" if path.suffix == ".mp3" else "image/jpeg"
    return FileResponse(path, media_type=media_type, filename=filename)
