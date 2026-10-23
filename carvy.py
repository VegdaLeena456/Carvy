import sounddevice as sd
import numpy as np
import openwakeword
from openwakeword.model import Model
from faster_whisper import WhisperModel
import edge_tts
import asyncio
import tempfile
import os
import ollama
from scipy.io.wavfile import write

# ====================== SETTINGS ======================
WAKE_WORD = "hey_jarvis"          # We will change this to "hey_carvy" later
SAMPLE_RATE = 16000
RECORD_SECONDS = 6                # How long to record after wake word
WHISPER_MODEL_SIZE = "base"       # Options: tiny, base, small, medium
OLLAMA_MODEL = "llama3.2"
VOICE = "en-US-AriaNeural"        # Nice female voice (you can change later)
# ======================================================

print("Loading models... please wait")

# Load Wake Word model
oww_model = Model(wakeword_models=[WAKE_WORD], inference_framework="onnx")

# Load Whisper (Speech-to-Text)
whisper = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")

print("Models loaded successfully!")
print("Carvy is ready. Say 'Hey Jarvis'...")


def record_audio(duration=RECORD_SECONDS):
    print("Listening...")
    audio = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    sd.wait()
    return audio.flatten()


def detect_wake_word():
    """Continuously listen until wake word is detected"""
    print("Waiting for wake word...")
    
    # Buffer for openWakeWord (it needs chunks)
    audio_buffer = np.array([], dtype=np.float32)
    
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32') as stream:
        while True:
            chunk, _ = stream.read(1280)  # ~80ms chunks
            chunk = chunk.flatten()
            audio_buffer = np.concatenate((audio_buffer, chunk))
            
            # Keep only last 1.5 seconds
            if len(audio_buffer) > SAMPLE_RATE * 1.5:
                audio_buffer = audio_buffer[-int(SAMPLE_RATE * 1.5):]
            
            # Predict
            prediction = oww_model.predict(audio_buffer)
            
            for mdl in prediction:
                score = prediction[mdl]
                if score > 0.5:  # Threshold
                    print(f"\nWake word detected! (score: {score:.2f})")
                    return True


def transcribe(audio):
    # Save temporary wav
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        temp_path = f.name
        write(temp_path, SAMPLE_RATE, (audio * 32767).astype(np.int16))
    
    segments, _ = whisper.transcribe(temp_path, language="en")
    text = " ".join([seg.text for seg in segments]).strip()
    
    os.unlink(temp_path)
    return text


async def speak(text):
    print(f"Carvy: {text}")
    communicate = edge_tts.Communicate(text, VOICE)
    
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        temp_path = f.name
    
    await communicate.save(temp_path)
    
    # Play the audio
    data, fs = sd.read(temp_path)
    sd.play(data, fs)
    sd.wait()
    
    os.unlink(temp_path)


def ask_ollama(prompt):
    response = ollama.chat(model=OLLAMA_MODEL, messages=[
        {"role": "system", "content": "You are Carvy, a helpful, friendly and concise AI assistant. Keep answers short and natural for speaking."},
        {"role": "user", "content": prompt}
    ])
    return response['message']['content']


# ====================== MAIN LOOP ======================
async def main():
    while True:
        detect_wake_word()
        
        # Record the command
        audio = record_audio()
        
        # Convert speech to text
        text = transcribe(audio)
        print(f"You said: {text}")
        
        if not text:
            await speak("I didn't catch that. Please try again.")
            continue
        
        if "exit" in text.lower() or "goodbye" in text.lower() or "stop" in text.lower():
            await speak("Goodbye!")
            break
        
        # Get reply from Ollama
        reply = ask_ollama(text)
        
        # Speak the reply
        await speak(reply)


if __name__ == "__main__":
    asyncio.run(main())