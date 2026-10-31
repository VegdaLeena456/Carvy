import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
import edge_tts
import asyncio
import tempfile
import os
import ollama
from scipy.io.wavfile import write

# MCP imports
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# ====================== SETTINGS ======================
SAMPLE_RATE = 16000
RECORD_SECONDS = 6
WHISPER_MODEL_SIZE = "base"
OLLAMA_MODEL = "llama3.2"
VOICE = "en-US-AriaNeural"
# ======================================================

print("Loading models... please wait")
whisper = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")
print("Models loaded successfully!")
print("\nCarvy is ready!")
print("→ Type your message and press Enter")
print("→ Or just press Enter to speak with your voice")
print("→ Type 'exit' to quit\n")


def record_audio(duration=RECORD_SECONDS):
    print("Listening... (speak now)")
    audio = sd.rec(
        int(duration * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype='float32'
    )
    sd.wait()
    return audio.flatten()


def transcribe(audio):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        temp_path = f.name

    write(temp_path, SAMPLE_RATE, (audio * 32767).astype(np.int16))

    try:
        segments, _ = whisper.transcribe(temp_path, language="en")
        text = " ".join([seg.text for seg in segments]).strip()
    finally:
        try:
            os.unlink(temp_path)
        except:
            pass

    return text


async def speak(text):
    print(f"Carvy: {text}")

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        temp_path = f.name

    try:
        communicate = edge_tts.Communicate(text, VOICE)
        await communicate.save(temp_path)

        safe_path = temp_path.replace("\\", "/")

        ps_command = f'''
        Add-Type -AssemblyName presentationCore
        $player = New-Object System.Windows.Media.MediaPlayer
        $player.Open([uri]"{safe_path}")
        $player.Play()
        Start-Sleep -Milliseconds 800
        while ($player.NaturalDuration.HasTimeSpan -eq $false) {{
            Start-Sleep -Milliseconds 200
        }}
        while ($player.Position -lt $player.NaturalDuration.TimeSpan) {{
            Start-Sleep -Milliseconds 200
        }}
        $player.Close()
        '''

        process = await asyncio.create_subprocess_exec(
            "powershell",
            "-Command",
            ps_command,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL
        )
        await process.wait()

    finally:
        try:
            os.unlink(temp_path)
        except:
            pass


async def ask_ollama_with_tools(prompt: str):
    """Talk to Ollama + use MCP browser tools (only when needed)"""
    
    # Keywords that mean the user wants to use a tool
    tool_keywords = ["open", "search", "google", "website", "browse", "go to", "visit"]

    user_wants_tool = any(keyword in prompt.lower() for keyword in tool_keywords)

    try:
        server_params = StdioServerParameters(
            command="python",
            args=["browser_server.py"],
        )

        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                tools_result = await session.list_tools()
                print("Tools found:", [t.name for t in tools_result.tools])

                ollama_tools = []
                for tool in tools_result.tools:
                    ollama_tools.append({
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description or "",
                            "parameters": tool.input_schema
                        }
                    })

                messages = [
                    {
                        "role": "system",
                        "content": (
                            "You are Carvy, a helpful and friendly AI assistant.\n"
                            "Keep answers short and natural.\n\n"
                            "Only use tools when the user clearly wants to open a website or search something."
                        )
                    },
                    {"role": "user", "content": prompt}
                ]

                # Only give tools if the user seems to want them
                if user_wants_tool:
                    response = ollama.chat(
                        model=OLLAMA_MODEL,
                        messages=messages,
                        tools=ollama_tools
                    )
                else:
                    # Normal conversation - no tools
                    response = ollama.chat(
                        model=OLLAMA_MODEL,
                        messages=messages
                    )

                # Handle tool calls
                if hasattr(response.message, "tool_calls") and response.message.tool_calls:
                    for tool_call in response.message.tool_calls:
                        tool_name = tool_call.function.name
                        arguments = tool_call.function.arguments

                        print(f"→ Using tool: {tool_name} | {arguments}")

                        result = await session.call_tool(tool_name, arguments)

                        messages.append(response.message)
                        messages.append({
                            "role": "tool",
                            "content": str(result.content)
                        })

                    final_response = ollama.chat(
                        model=OLLAMA_MODEL,
                        messages=messages
                    )
                    return final_response.message.content
                else:
                    return response.message.content

    except Exception as e:
        import traceback
        print("\n=== FULL ERROR ===")
        traceback.print_exc()
        print("==================\n")
        return "Sorry, I had a problem using the tools."


async def main():
    while True:
        try:
            user_input = input("You: ").strip()

            if user_input.lower() in ["exit", "quit", "goodbye", "stop"]:
                await speak("Goodbye!")
                break

            if user_input == "":
                # Voice input
                audio = record_audio()
                text = transcribe(audio)
                print(f"You said: {text}")
            else:
                text = user_input

            if not text:
                await speak("I didn't catch that. Please try again.")
                continue

            # Use the new function that supports tools
            reply = await ask_ollama_with_tools(text)
            await speak(reply)

        except KeyboardInterrupt:
            print("\nStopped by user.")
            break
        except Exception as e:
            print(f"\nError occurred: {e}")
            print("Please try again.\n")


if __name__ == "__main__":
    asyncio.run(main())