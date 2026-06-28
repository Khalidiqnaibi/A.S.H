// renderer.js
const user =  window.USERNAME || null;

const chatLog = document.getElementById('chat-log');
const inputField = document.getElementById('input-field');
const sendButton = document.getElementById('send-button');
const micButton = document.getElementById('mic-button'); // Form component bind

const socket = io(); // auto-connects

function logAdd(txt , from = "ai") {
  const logEntry = document.createElement('div'); // use div for long content
  logEntry.classList.add('log-entry');
  if (from === "user") logEntry.classList.add('user-entry');
  if (from === "system") logEntry.classList.add('sys-entry');

  // Convert markdown to HTML
  logEntry.innerHTML = marked.parse(txt);

  chatLog.appendChild(logEntry);
  chatLog.scrollTop = chatLog.scrollHeight;
}


function kprint(txt) {
  logAdd(`ASH: ${txt}`);
}


/* ---------- SOCKET EVENTS ---------- */

socket.on("connect", () => {
  console.log("Connected to server");
});

socket.on("disconnect", (reason) => {
  console.warn("Disconnected:", reason);
  logAdd("⚠️ Connection lost. Reconnecting...", "system");
});

socket.on("reconnect", () => {
  console.log("Reconnected to server");
  logAdd("✅ Reconnected!", "system");
});

socket.on("connect_error", (err) => {
  console.error("Socket error:", err);
});

// Server pushed text response sequence
socket.on("ash_response", (data) => {
  // Clear the "thinking..." animation block if it exists
  const thinking = document.getElementById('ash-thinking');
  if (thinking) thinking.remove();

  logAdd(data.text, "ai");
  sendButton.disabled = false; // unlock input form controls
});

// Server pushed raw STT voice transcript to sync text frames
socket.on("voice_transcript", (data) => {
  const systemMessages = chatLog.querySelectorAll('.sys-entry');
    systemMessages.forEach(msg => {
      if (msg.textContent.includes('🎙️ Processing your voice entry...')) {
        msg.remove();
      }
    });
  const thinking = document.getElementById('ash-thinking');
  if (thinking) thinking.remove();
  
  logAdd(data.text, "user");

  // Inject a fresh thinking indicator while LLM finishes generating stream
  const thinkingDiv = document.createElement('div');
  thinkingDiv.id = 'ash-thinking';
  thinkingDiv.classList.add('log-entry', 'sys-entry');
  thinkingDiv.textContent = 'ASH is thinking...';
  chatLog.appendChild(thinkingDiv);
  chatLog.scrollTop = chatLog.scrollHeight;
});

socket.on("system", (data) => {
  logAdd(data.msg, "system");
});


/* ---------- USER INPUT HANDLING ---------- */

sendButton.addEventListener('click', () => {
  const message = inputField.value.trim();
  if (!message) return;

  // Render immediately locally
  logAdd(message, "user");
  inputField.value = '';

  // Append a lightweight systemic visual anchor for generation lifecycle tracking
  const thinkingDiv = document.createElement('div');
  thinkingDiv.id = 'ash-thinking';
  thinkingDiv.classList.add('log-entry', 'sys-entry');
  thinkingDiv.textContent = 'ASH is thinking...';
  chatLog.appendChild(thinkingDiv);
  chatLog.scrollTop = chatLog.scrollHeight;

  // Disable send button while processing
  sendButton.disabled = true;

  socket.emit("user_message", {
    user: user,
    message: message
  });

  // Re-enable after a safety timeout (response handler also re-enables)
  setTimeout(() => { sendButton.disabled = false; }, 30000);
});

inputField.addEventListener('keyup', (event) => {
  if (event.key === 'Enter') {
    sendButton.click();
  }
});


/* ---------- VOICE I/O (STT & LOCAL KOKORO TTS) ---------- */

// 1. Local TTS Controls & Event State Trackers
let ttsEnabled = false;
const ttsToggle = document.getElementById('tts-toggle');

const audioQueue = [];
let isPlayingAudio = false;
let currentAudio = null;

if (ttsToggle) {
    ttsToggle.addEventListener('click', () => {
        ttsEnabled = !ttsEnabled;
        ttsToggle.textContent = ttsEnabled ? "🔊 TTS: ON" : "🔊 TTS: OFF";
        ttsToggle.style.background = ttsEnabled ? "#042b46" : "transparent";
        
        // Dynamic Intercept: Flush buffers if disabled mid-stream
        if (!ttsEnabled) {
            audioQueue.length = 0;
            if (currentAudio) {
                currentAudio.pause();
                isPlayingAudio = false;
                currentAudio = null;
            }
        }
    });
}

// 2. Process Binary WAV Chunks from Socket Server Pipeline
socket.on("ash_audio", (data) => {
    if (!ttsEnabled) return; 

    // Convert raw binary network payload into a playable audio blob container
    const blob = new Blob([data.audio], { type: 'audio/wav' });
    const url = URL.createObjectURL(blob);
    
    audioQueue.push(url);
    playNextChunk();
});

// 3. Sequential Back-to-Back Audio Queue Executor
function playNextChunk() {
    if (isPlayingAudio || audioQueue.length === 0) return;

    isPlayingAudio = true;
    const nextAudioUrl = audioQueue.shift();
    currentAudio = new Audio(nextAudioUrl);
    
    currentAudio.onended = () => {
        isPlayingAudio = false;
        URL.revokeObjectURL(nextAudioUrl); // Memory release guard
        playNextChunk(); // Recursively execute subsequent chunks
    };

    currentAudio.play().catch(e => {
        console.error("Audio playback execution blocked or dropped:", e);
        isPlayingAudio = false;
        playNextChunk();
    });
}


/* ---------- HARDWARE MICROPHONE INPUT CONTROLLER (LOCAL STT) ---------- */

let isRecording = false;
let mediaRecorder = null;
let audioChunks = [];

if (micButton) {
    micButton.addEventListener('click', async () => {
        if (isRecording) {
            if (mediaRecorder && mediaRecorder.state !== "inactive") {
                mediaRecorder.stop();
            }
        } else {
            audioChunks = [];
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                isRecording = true;
                micButton.classList.add('recording');
                
                mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });
                
                mediaRecorder.ondataavailable = (event) => {
                    if (event.data.size > 0) {
                        audioChunks.push(event.data);
                    }
                };
                
                mediaRecorder.onstart = () => {
                    inputField.placeholder = "Listening... Press mic again to stop.";
                };
                
                mediaRecorder.onstop = async () => {
                    isRecording = false;
                    micButton.classList.remove('recording');
                    inputField.placeholder = "Processing voice payload...";
                    
                    // Kill device stream tracks to turn off recording hardware lights
                    stream.getTracks().forEach(track => track.stop());
                    
                    // Combine audio buffer frames to assemble raw binary chunks
                    const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
                    
                    // Unpack immutable data stream into an ArrayBuffer container block
                    const arrayBuffer = await audioBlob.arrayBuffer();
                    
                    // Stream binary elements directly via WebSocket pipelines
                    socket.emit("user_voice", arrayBuffer);
                    
                    inputField.placeholder = "Talk to ASH...";
                };
                
                mediaRecorder.start();
                
            } catch (err) {
                console.error("WebSocket microphone acquisition failure:", err);
                logAdd("⚠️ Could not claim mic hardware. Check browser rules.", "system");
            }
        }
    });
}