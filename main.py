from fastapi import FastAPI
from pydantic import BaseModel
import edge_tts
import subprocess

app = FastAPI()

class TTSRequest(BaseModel):
    text: str
    voice: str = "zh-TW-YunJieNeural"
    output_filename: str = "voice.mp3"

@app.get("/")
def read_root():
    return {"status": "FlowinSoul Render Engine Running"}

@app.post("/generate-tts")
async def generate_tts(req: TTSRequest):
    output_path = f"/tmp/{req.output_filename}"
    communicate = edge_tts.Communicate(req.text, req.voice)
    await communicate.save(output_path)
    return {"status": "success", "file_path": output_path}
