import os
import torch
import soundfile as sf
import base64
from faster_whisper import WhisperModel
import config 
import gc

# Conditionally import TTS-related libraries only if enabled to avoid heavy GPU usage
if config.ENABLE_TTS:
    try:
        from parler_tts import ParlerTTSForConditionalGeneration
        from transformers import AutoTokenizer
    except Exception as e:
        print(f"⚠️ Unable to import TTS libraries: {e}")
        # We'll handle the absence gracefully in the class

class AudioEngine:
    def __init__(self):
        print("🎧 Initializing Audio Engine (AI4Bharat Indic)...")
        
        # 1. Load Faster Whisper (Strictly CPU to save VRAM for Parler/LLM)
        print(f"Loading Faster Whisper ({config.WHISPER_MODEL_ID}) on CPU...")
        self.stt_model = WhisperModel(config.WHISPER_MODEL_ID, device="cpu", compute_type="int8")

        # 2. Load Parler TTS (only if enabled)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if config.ENABLE_TTS:
            print(f"Loading AI4Bharat Parler TTS on {self.device}...")
            try:
                self.tts_model = ParlerTTSForConditionalGeneration.from_pretrained(config.PARLER_MODEL_ID).to(self.device)
                self.tts_tokenizer = AutoTokenizer.from_pretrained(config.PARLER_MODEL_ID)
                # Pre-compute the default description token (English Indian)
                self.default_description_tokens = self.tts_tokenizer(
                    config.TTS_DESCRIPTION, return_tensors="pt"
                ).input_ids.to(self.device)
                self.tts_enabled = True
            except Exception as e:
                print(f"❌ Error loading Parler TTS: {e}")
                print("⚠️ Disabling TTS fallback due to error.")
                self.tts_enabled = False
                self.tts_model = None
                self.tts_tokenizer = None
                self.default_description_tokens = None
        else:
            print("⚠️ TTS is disabled in config. AudioEngine will only provide STT functionality.")
            self.tts_model = None
            self.tts_tokenizer = None
            self.default_description_tokens = None
            self.tts_enabled = False

    def speech_to_text(self, audio_path):
        """Converts audio file path to text using Faster Whisper."""
        try:
            segments, info = self.stt_model.transcribe(audio_path, beam_size=5)
            text = " ".join([segment.text for segment in segments])
            return text.strip()
        except Exception as e:
            print(f"❌ STT Error: {e}")
            return ""

    def text_to_speech(self, text, output_path):
        """Converts text to audio using AI4Bharat Parler TTS. If TTS is disabled, this is a no-op returning None."""
        if not getattr(self, "tts_enabled", False):
            print("⚠️ TTS requested but TTS is disabled. Skipping audio generation.")
            return None
        try:
            # 1. Clean memory before generation (Crucial for 6GB VRAM)
            if self.device == "cuda":
                torch.cuda.empty_cache()

            # 2. Tokenize text
            prompt_input_ids = self.tts_tokenizer(text, return_tensors="pt").input_ids.to(self.device)
            
            # 3. Generate Audio
            # We use the description from config to enforce Indian accent
            generation = self.tts_model.generate(
                input_ids=self.default_description_tokens,
                prompt_input_ids=prompt_input_ids
            )
            
            # 4. Save Audio
            audio_arr = generation.cpu().numpy().squeeze()
            sf.write(output_path, audio_arr, self.tts_model.config.sampling_rate)
            
            return output_path
        
        except Exception as e:
            print(f"❌ TTS Generation Error: {e}")
            # print stack trace for debugging
            import traceback
            traceback.print_exc()
            return None

    def audio_file_to_base64(self, file_path):
        """Helper to send audio back via API."""
        if not os.path.exists(file_path):
            return None
        with open(file_path, "rb") as audio_file:
            return base64.b64encode(audio_file.read()).decode('utf-8')