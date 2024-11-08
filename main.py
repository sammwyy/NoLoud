import sounddevice as sd
from df.enhance import enhance, init_df
import numpy as np
import torch
import threading
from queue import Queue, Empty
import signal
import sys

# Initialize DeepFilterNet model for noise cancellation
model, df_state, _ = init_df()

# Audio configuration
SAMPLE_RATE = df_state.sr()
BLOCK_SIZE = 1024 * 8  # Smaller block size for lower latency
BUFFER_SIZE = BLOCK_SIZE * 10  # Buffer size for smoother audio flow

# Circular buffer to smooth out audio glitches
circular_buffer = np.zeros(BUFFER_SIZE, dtype=np.float32)
buffer_index = 0  # Start index for buffer writing

# Queue for handling audio processing asynchronously
audio_queue = Queue(maxsize=5)

# Event to stop processing on Ctrl+C
stop_event = threading.Event()

# Callback for capturing, processing, and playing audio
def audio_callback(indata, outdata, frames, time, status):
    if status:
        print(status)
        
    global buffer_index  # Declare as global here
    
    # Convert input data to numpy array if not already
    audio_mono = indata[:, 0] if indata.ndim > 1 else indata.flatten()
    # Add audio block to processing queue
    if not audio_queue.full():
        audio_queue.put((audio_mono, frames))

    # Pull from the circular buffer into the output stream
    outdata[:, 0] = circular_buffer[buffer_index:buffer_index + frames]
    # Update buffer index, wrapping if necessary
    buffer_index = (buffer_index + frames) % BUFFER_SIZE

# Background thread for real-time noise cancellation
def process_audio():
    global buffer_index  # Declare as global here
    while not stop_event.is_set():
        try:
            # Fetch block from queue or skip if empty
            audio_mono, frames = audio_queue.get(timeout=0.1)
            # Convert to PyTorch tensor and add batch dimension
            audio_tensor = torch.from_numpy(audio_mono).float().unsqueeze(0)
            # Apply noise cancellation
            enhanced_audio = enhance(model, df_state, audio_tensor)
            # Flatten and convert back to numpy
            enhanced_audio_np = enhanced_audio.squeeze(0).detach().numpy()
            
            # Write the processed audio to the circular buffer
            end_index = buffer_index + frames
            if end_index <= BUFFER_SIZE:
                circular_buffer[buffer_index:end_index] = enhanced_audio_np[:frames]
            else:
                # Wrap around circular buffer
                part1_len = BUFFER_SIZE - buffer_index
                circular_buffer[buffer_index:] = enhanced_audio_np[:part1_len]
                circular_buffer[:end_index % BUFFER_SIZE] = enhanced_audio_np[part1_len:]

            audio_queue.task_done()
        except Empty:
            continue

# Signal handler for Ctrl+C
def signal_handler(sig, frame):
    print("\nTerminating noise cancellation process...")
    stop_event.set()
    sys.exit(0)

# Register signal handler
signal.signal(signal.SIGINT, signal_handler)

# Start background processing thread
processing_thread = threading.Thread(target=process_audio, daemon=True)
processing_thread.start()

# Set up the audio stream using Stream
with sd.Stream(callback=audio_callback, samplerate=SAMPLE_RATE, blocksize=BLOCK_SIZE, dtype='float32', channels=1):
    print("Starting real-time noise cancellation. Press Ctrl+C to stop.")
    while not stop_event.is_set():
        sd.sleep(100)
