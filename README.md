# Clinical Voice Debrief Scribe 🩺🎙️

An AI-powered clinical documentation pipeline that transforms spoken physician debriefs into structured, EHR-ready **SOAP** (Subjective, Objective, Assessment, Plan) notes.

The system pairs edge-local speech transcription (`faster-whisper`) with clinical reasoning via **NVIDIA NIM** (`meta/llama-3.2-11b-vision-instruct`), enforced by **Pydantic v2** validation to guarantee deterministic schema compliance.

---

## 🏗️ Architecture

```text
[ Browser Audio Capture (WebM) ]
               │
               ▼
    FastAPI Ingestion Endpoint (/process-voice-debrief)
               │
               ├─► 1. Local Edge ASR (faster-whisper base.en, INT8)
               │        └─► Raw Spoken Transcript (Zero API cost / No latency quotas)
               │
               ├─► 2. NVIDIA NIM Reasoning (meta/llama-3.2-11b-vision-instruct)
               │        └─► Clinical Entity Extraction & Normalization
               │
               ├─► 3. Pydantic v2 Schema Validation
               │        └─► Strict JSON Parsing & Boundary Sanitization
               │
               ▼
[ Web UI: Formatted SOAP Card | Clipboard Copy | Print-to-PDF ]
```

---

## ✨ Features

- **Physician Debrief Workflow:** Summarize encounters in 30–60 seconds naturally via voice instead of manual EHR typing.
- **Edge-Local Audio Transcription:** Runs `faster-whisper` on-device (CPU-friendly via INT8 quantization), eliminating third-party audio upload limits, rate caps, and cloud speech expenses.
- **Discrete Medication Parsing:** Normalizes spoken prescriptions into discrete fields: `name`, `dosage`, `frequency`, and `duration`.
- **Schema Boundary Enforcement:** Custom regex-based sanitization strips markdown fences and echoes to protect Pydantic validation from malformed LLM responses.
- **Windows-Resilient Network Transport:** Employs explicit HTTP/1.1 transport and standard headers via `httpx` to eliminate socket drops and TLS handshake resets with NVIDIA's inference gateway.
- **Print & EHR Integration:** One-click clipboard formatting for instant EHR pasting, paired with dedicated `@media print` styles for clean physical chart generation.

---

## 🛠️ Tech Stack

- **Backend:** Python 3.10+, FastAPI, Uvicorn
- **Speech-to-Text (ASR):** `faster-whisper` (CTranslate2, `base.en`, INT8)
- **Clinical Reasoning:** NVIDIA NIM (`meta/llama-3.2-11b-vision-instruct`)
- **Data Validation:** Pydantic v2
- **Networking:** HTTPX (configured with HTTP/1.1 and custom timeout controls)
- **Frontend:** Vanilla JavaScript, HTML5 MediaRecorder API, CSS3

---

## 📂 Project Structure

```text
clinical-scribe/
├── .env                  # API configuration (git-ignored)
├── .gitignore            # Environment, model cache, and audio ignore rules
├── requirements.txt      # Python runtime dependencies
├── main.py               # FastAPI pipeline, Whisper model, & NIM extraction logic
├── README.md             # Project documentation
└── static/
    └── index.html        # Audio recorder, SOAP viewer, clipboard & print controls
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone [https://github.com/](https://github.com/)<your-username>/clinical-scribe.git
cd clinical-scribe
```

### 2. Set Up a Virtual Environment

**Windows (Command Prompt):**
```cmd
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the root directory:

```env
NVIDIA_API_KEY=nvapi-your_nvidia_api_key_here
```

> Get a free API key at [build.nvidia.com](https://build.nvidia.com).

### 5. Start the Application

```bash
python main.py
```

Navigate to `http://127.0.0.1:8000` in your browser.

---

## 📋 Data Schema

The extraction layer produces validated JSON adhering to the following schema:

```json
{
  "patient_identifier": "David Clark, 52 years old",
  "subjective": "Burning epigastric pain and acid reflux worsening after meals for two weeks.",
  "objective": "Mild epigastric tenderness on exam, no guarding or rebound.",
  "assessment": "Gastroesophageal reflux disease (GERD)",
  "plan_medications": [
    {
      "name": "Omeprazole",
      "dosage": "20 mg",
      "frequency": "Once daily before breakfast",
      "duration": "4 weeks"
    }
  ],
  "plan_instructions": [
    "Avoid late-night meals",
    "Avoid spicy foods"
  ]
}
```

---

## 🧪 Sample Audio Dictation for Testing

> *"Patient John Miller, 40 years old, presents with a productive cough and mild fever for three days. Lungs reveal scattered bilateral wheezing on auscultation. Diagnosis is acute bronchitis. Prescribing Albuterol inhaler two puffs every six hours as needed, and recommended rest with increased fluids."*

---

## 📄 License

Distributed under the MIT License.
