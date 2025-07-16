import asyncio
import base64
import os
import numpy as np
import pyaudio
from google import genai
from google.genai import types
model_name = 'gemini-2.0-flash-live-001'
system_prompt = 'you are helpful travel assistant'
gemini_api_key = '[your gemini api key]'
class AudioUtils:
    @staticmethod
    def encode_bytes(data: bytes) -> str:
        return base64.b64encode(data).decode('utf-8')
    
    @staticmethod
    def create_blob(data: np.ndarray) -> dict:
        """Convert float32 audio data to blob format for Gemini"""
        # Convert float32 (-1 to 1) to int16 (-32768 to 32767)
        int16_data = (data * 32768).astype(np.int16)
        
        # Convert to bytes
        audio_bytes = int16_data.tobytes()
        
        return {
            'data': AudioUtils.encode_bytes(audio_bytes),
            'mimeType': 'audio/pcm;rate=16000'
        }
    
    @staticmethod
    def decode_audio_data(data: bytes, sample_rate: int = 24000, num_channels: int = 1) -> np.ndarray:
        """Decode audio data from bytes to float32 array"""
        int16_data = np.frombuffer(data, dtype=np.int16)
        float32_data = int16_data.astype(np.float32) / 32768.0
        if num_channels == 1:
            return float32_data
        else:
            # Deinterleave channels
            return float32_data.reshape(-1, num_channels)

class GeminiLiveAudio:
    def __init__(self):
        self.is_recording = False
        self.is_starting = False
        self.is_session_ready = False
        self.status = ""
        self.error = ""
        
        self.input_sample_rate = 16000
        self.output_sample_rate = 24000
        self.chunk_size = 1600
        self.audio_format = pyaudio.paInt16
        self.channels = 1
        
        self.audio = pyaudio.PyAudio()
        self.input_stream = None
        self.output_stream = None
        
        self.client = None
        self.session = None
        self.endturn = False
        self.stop_put = False
        
        self.audio_queue = asyncio.Queue()
        self.playback_queue = asyncio.Queue()
        self.sources = set()
        
        # Initialize
        self.init_client()
    
    def init_client(self):
        """Initialize Gemini client"""
        try:
            self.client = genai.Client(api_key=gemini_api_key)            
            self.update_status("Client initialized")
            asyncio.create_task(self.init_session())
        except Exception as e:
            self.update_error(f"Failed to initialize client: {str(e)}")

            
    async def init_session(self):
        """Initialize Gemini live session"""
    
        self.update_status("Connecting to Gemini...")
        config = {
        "response_modalities": ["AUDIO"],
        "system_instruction": system_prompt,
        "realtime_input_config": {
        "automatic_activity_detection": {
                    "disabled": False, # default
                    "silence_duration_ms": 2000, #2s
                }
            }
        }
        
        async with self.client.aio.live.connect(model=model_name, config=config) as session:
            self.session = session
            self.is_session_ready = True
            self.update_status("Ready to chat!")

            while True:
                self.endturn = False
                self.stop_put = False
                self.audio_queue = asyncio.Queue()
                self.playback_queue = asyncio.Queue()
                send_task = asyncio.create_task(self._process_audio())
                receive_task = asyncio.create_task(self._listen_for_gemini_responses())
                playback_task = asyncio.create_task(self._handle_playback())
                await asyncio.gather(send_task, receive_task, playback_task)
                        
        return

    def update_status(self, msg: str):
        self.status = msg
        if msg:
            self.error = ""
        print(f"Status: {msg}")
    
    def update_error(self, msg: str):
        self.error = msg
        print(f"Error: {msg}")
    
    def start_recording(self):
        """Start audio recording"""
        if self.is_recording and self.is_starting:
            return
        
        self.is_starting = True
        self.update_status("Starting microphone...")
        
        try:
            # Initialize input stream
            self.input_stream = self.audio.open(
                format=self.audio_format,
                input_device_index=1,#in my computer 1 is default micro
                channels=self.channels,#1
                rate=self.input_sample_rate,#16000
                input=True,
                frames_per_buffer=self.chunk_size,#160
                stream_callback=self._audio_callback
            )
            # Initialize output stream for playback
            self.output_stream = self.audio.open(
                format=self.audio_format,
                input_device_index=2,
                channels=self.channels,
                rate=self.output_sample_rate,
                output=True,
                frames_per_buffer=self.chunk_size
            )
            
            self.input_stream.start_stream()
            self.output_stream.start_stream()
            
            self.is_recording = True
            self.update_status("🔴 Recording...")
            # Start audio processing task
            # asyncio.create_task(self._process_audio())
            # asyncio.create_task(self._handle_playback())
            
        except Exception as e:
            self.update_error(f"Error starting recording: {str(e)}")
            self.stop_recording()
        finally:
            self.is_starting = False
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Audio input callback"""
        if self.is_recording:
            audio_data = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
            try:
                if not self.stop_put:
                    self.audio_queue.put_nowait(audio_data)
            except asyncio.QueueFull:
                pass  # Skip if queue is full
        
        return (None, pyaudio.paContinue)
    
    async def _process_audio(self):#send
        """Process audio data and send to Gemini"""
        while self.is_recording:
            try:
                if self.endturn == True:
                    break
                audio_data = await asyncio.wait_for(self.audio_queue.get(), timeout=2)
                
                if self.session and self.is_session_ready:
                    # Create blob and send to Gemini
                    blob = AudioUtils.create_blob(audio_data)
                    # Send audio to Gemini
                    await self.session.send_realtime_input(audio=types.Blob(data = blob['data'],
                                                                            mime_type = blob['mimeType']))
                else:
                    pass 
            except asyncio.TimeoutError:
                continue        
    
    async def _listen_for_gemini_responses(self):#get
            # Iterate asynchronously over responses from the Gemini session
        async for response in self.session.receive():
            audio_data_bytes = response.data
            await self.playback_queue.put(audio_data_bytes)

    
    async def _handle_playback(self):#read
        while self.is_recording:
            try:
                # Get audio data for playback
                audio_data = await asyncio.wait_for(self.playback_queue.get(), timeout=3)
                if audio_data == None:
                    self.endturn = True
                    break
                if self.output_stream and self.output_stream.is_active():
                    # Convert and play audio
                    audio_bytes = AudioUtils.decode_audio_data(audio_data, self.output_sample_rate)
                    int16_data = (audio_bytes * 32768).astype(np.int16)
                    self.output_stream.write(int16_data.tobytes())
                    self.stop_put = True
            except asyncio.TimeoutError:
                print(3)
                continue
            except Exception as e:
                print(f"Error during playback: {e}")
    
    def stop_recording(self):
        """Stop audio recording"""
        if not self.is_recording:
            return
        
        self.update_status("Stopping recording...")
        
        self.is_recording = False
        
        # Stop and close streams
        if self.input_stream:
            self.input_stream.stop_stream()
            self.input_stream.close()
            self.input_stream = None
        
        if self.output_stream:
            self.output_stream.stop_stream()
            self.output_stream.close()
            self.output_stream = None
        
        self.update_status("Recording stopped. Click Start to begin again.")
    
    def reset(self):
        """Reset the session"""
        if self.is_recording:
            self.stop_recording()
        
        self.is_session_ready = False
        self.session = None
        
        # Reinitialize session
        asyncio.create_task(self.init_session())
        self.update_status("Session reset.")
    
    def cleanup(self):
        """Clean up resources"""
        self.stop_recording()
        if self.audio:
            self.audio.terminate()

# Example usage
async def main():
    # Set your Gemini API key
    os.environ['GEMINI_API_KEY'] = 'AIzaSyDSNSPPE2L7Ah3elfAsnCvERTRByz0XNXg'
    
    # Create audio chat instance
    audio_chat = GeminiLiveAudio()
    
    try:
        # Start recording
        audio_chat.start_recording()
        
        # Keep running
        while True:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        audio_chat.cleanup()

if __name__ == "__main__":
    # Install required packages:
    # pip install pyaudio numpy google-generativeai websockets
    asyncio.run(main())
