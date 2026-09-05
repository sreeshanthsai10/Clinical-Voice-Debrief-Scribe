import json
import os
import re
import shutil
import tempfile
from typing import List, Optional
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from openai import OpenAI
from faster_whisper import WhisperModel
import httpx

load_dotenv()

app = FastAPI(title="Clinical Voice Debrief - Scribe")

os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def serve_index():
    return FileResponse("static/index.html")

# 1. Local Whisper Model
print("Loading local speech model (faster-whisper)...")
whisper_model = WhisperModel("base.en", device="cpu", compute_type="int8")
print("Local speech model ready.")

# 2. NVIDIA NIM Configuration
nvidia_api_key = os.getenv("NVIDIA_API_KEY", "").strip()
if not nvidia_api_key:
    print("WARNING: NVIDIA_API_KEY is not set in .env")

# HTTP/1.1 transport prevents Windows socket resets with integrate.api.nvidia.com
http_client = httpx.Client(
    timeout=60.0,
    http2=False,
    headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) ClinicalScribe/1.0"}
)

nvidia_client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=nvidia_api_key,
    http_client=http_client
)

# Active Free Endpoint on your NVIDIA dashboard
NVIDIA_MODEL = "meta/llama-3.2-11b-vision-instruct"


class Medication(BaseModel):
    name: str = Field(description="Generic or brand name of medication")
    dosage: Optional[str] = Field(None, description="e.g., 500mg, 2 puffs, 10ml")
    frequency: Optional[str] = Field(None, description="e.g., Twice daily, PRN, every 6 hours")
    duration: Optional[str] = Field(None, description="e.g., 5 days, 10 days, 2 weeks")

class SOAPNote(BaseModel):
    patient_identifier: Optional[str] = Field(None, description="Patient identifier, age, or demographics")
    subjective: str = Field(description="History of present illness, symptoms, pain scale, and duration")
    objective: str = Field(description="Physical examination observations, vitals, auscultation, tenderness")
    assessment: str = Field(description="Clinical impression and primary or differential diagnoses")
    plan_medications: List[Medication] = Field(default_factory=list, description="Prescribed medications")
    plan_instructions: List[str] = Field(
        default_factory=list,
        description="Non-pharmacological clinical instructions, lifestyle changes, referrals, or lab tests"
    )


def extract_clean_json(text: str) -> str:
    """Strips markdown code fences and isolates valid JSON boundaries."""
    text = text.strip()
    if "```" in text:
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        if match:
            text = match.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        text = text[start:end + 1]
    return text


@app.post("/process-voice-debrief", response_model=SOAPNote)
async def process_voice_debrief(file: UploadFile = File(...)):
    allowed_extensions = (".wav", ".mp3", ".m4a", ".webm", ".ogg")
    suffix = os.path.splitext(file.filename)[-1].lower() if file.filename else ".webm"
    if suffix not in allowed_extensions:
        suffix = ".webm"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        # Step A: Local Speech-to-Text via Faster-Whisper
        segments, _ = whisper_model.transcribe(
            tmp_path,
            beam_size=5,
            initial_prompt="Clinical consultation, medical debrief, medications, anatomy, diagnosis"
        )
        transcript_text = " ".join([seg.text for seg in segments]).strip()
        print(f"\n[Transcribed Audio]: {transcript_text}\n")

        if not transcript_text:
            raise HTTPException(status_code=400, detail="No speech detected in recording.")

        # Step B: Clinical SOAP Extraction via NVIDIA NIM
        system_prompt = (
            "You are a clinical medical scribe. Process the doctor's transcript and produce a JSON object.\n"
            "Respond ONLY with a valid JSON object matching this exact structure without markdown or backticks:\n"
            "{\n"
            '  "patient_identifier": "string or null",\n'
            '  "subjective": "string",\n'
            '  "objective": "string",\n'
            '  "assessment": "string",\n'
            '  "plan_medications": [\n'
            '    {"name": "string", "dosage": "string or null", "frequency": "string or null", "duration": "string or null"}\n'
            '  ],\n'
            '  "plan_instructions": ["string"]\n'
            "}"
        )
        user_prompt = f"Doctor's Voice Debrief Transcript:\n\"\"\"\n{transcript_text}\n\"\"\""

        print(f"[NVIDIA NIM] Sending transcript to {NVIDIA_MODEL}...")
        completion = nvidia_client.chat.completions.create(
            model=NVIDIA_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=1000
        )
        print("[NVIDIA NIM] Extraction completed successfully.")

        raw_response = completion.choices[0].message.content
        cleaned_json = extract_clean_json(raw_response)

        return SOAPNote.model_validate_json(cleaned_json)

    except Exception as e:
        print(f"[Error processing note]: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)