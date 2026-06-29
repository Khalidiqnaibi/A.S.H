# tts.py
import io
import re
import logging

try:
    import soundfile as sf
    from kokoro_onnx import Kokoro
    KOKORO_AVAILABLE = True
except ImportError:
    KOKORO_AVAILABLE = False

logger = logging.getLogger("ash.tts")
logger.setLevel(logging.INFO)
if not logger.handlers:
    import sys
    logger.addHandler(logging.StreamHandler(stream=sys.stderr))

class TTSEngine:
    """Always-on Kokoro TTS engine."""
    def __init__(self, model_path="models/kokoro-v0_19.onnx", voices_path="models/voices.json"):
        """Initializes the always-on Kokoro TTS engine."""
        self.engine = None
        if not KOKORO_AVAILABLE:
            logger.warning("Kokoro-ONNX or soundfile not installed. Local TTS disabled.")
            return
        
        try:
            logger.info(f"Loading Kokoro TTS Engine from {model_path}...")
            self.engine = Kokoro(model_path, voices_path)
            logger.info("Kokoro TTS Engine ready.")
        except Exception as e:
            logger.error(f"Failed to load Kokoro: {e}")

    def stream_audio(self, text: str, voice: str = "af_heart", speed: float = 1.0):
        """
        Takes a full response block, splits it into sentences, 
        and yields binary WAV audio chunks one by one.
        """
        if not self.engine:
            return

        # 1. Split text into logical sentences
        sentences = re.split(r'(?<=[.!?]) +', text)
        
        for sentence in sentences:
            # 2. Clean out Markdown (asterisks, hashes) so A.S.H doesn't read them aloud
            clean_sentence = re.sub(r'[*_`#>-]', '', sentence).strip()
            
            if not clean_sentence:
                continue
            
            try:
                # 3. Generate raw audio array
                samples, sample_rate = self.engine.create(
                    clean_sentence, voice=voice, speed=speed, lang="en-us"
                )
                
                # 4. Pack into a WAV byte stream
                wav_io = io.BytesIO()
                sf.write(wav_io, samples, sample_rate, format='WAV', subtype='PCM_16')
                
                # 5. Yield the binary data
                yield wav_io.getvalue()
                
            except Exception as e:
                logger.error(f"TTS Generation failed for chunk '{clean_sentence}': {e}")