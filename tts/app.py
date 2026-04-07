from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import torch
import io
import soundfile as sf
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer

app = FastAPI()
device = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_ID = "ai4bharat/indic-parler-tts"

model = ParlerTTSForConditionalGeneration.from_pretrained(
    MODEL_ID
).to(device)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
description_tokenizer = AutoTokenizer.from_pretrained(
    model.config.text_encoder._name_or_path
)

class TTSRequest(BaseModel):
    text: str
    description: str

@app.post("/tts")
def tts(req: TTSRequest):
    desc = description_tokenizer(
        req.description,
        return_tensors="pt"
    ).to(device)

    prompt = tokenizer(
        req.text,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        audio = model.generate(
            input_ids=desc.input_ids,
            attention_mask=desc.attention_mask,
            prompt_input_ids=prompt.input_ids,
            prompt_attention_mask=prompt.attention_mask,
        )

    audio = audio.cpu().numpy().squeeze()

    # Write WAV to memory
    buffer = io.BytesIO()
    sf.write(buffer, audio, model.config.sampling_rate, format="WAV")
    buffer.seek(0)

    return StreamingResponse(
        buffer,
        media_type="audio/wav"
    )
