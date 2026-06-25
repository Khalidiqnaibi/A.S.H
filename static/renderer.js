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
  console.error("Connection error:", err.message);
});

socket.on("system", (data) => {
  logAdd(data.msg, "system");
});

socket.on("ash_response", (data) => {
  // Remove thinking indicator if present
  const thinking = document.getElementById('thinking-indicator');
  if (thinking) thinking.remove();
  
  // Re-enable send button
  sendButton.disabled = false;
  kprint(data.text);
  
  speakText(data.text); // Preserved TTS output engine invocation loop
});

// Appends user text log bubble during hands-free WebSocket transcription returns
socket.on("voice_transcript", (data) => {
    logAdd(`${user ? user : 'User'}: ${data.text}`, 'user');
    
    // Inject identical processing indicators to match core click flow architectures
    const thinkingDiv = document.createElement('div');
    thinkingDiv.id = 'thinking-indicator';
    thinkingDiv.classList.add('log-entry', 'sys-entry');
    thinkingDiv.textContent = 'ASH is thinking...';
    chatLog.appendChild(thinkingDiv);
    chatLog.scrollTop = chatLog.scrollHeight;
    
    sendButton.disabled = true;
});


/* ---------- SEND MESSAGE ---------- */

sendButton.addEventListener('click', () => {

  const message = inputField.value.trim();
  inputField.value = '';

  if (!message) return;

  logAdd(`${user ? user : 'User'}: ${message}`,'user');

  // Show thinking indicator
  const thinkingDiv = document.createElement('div');
  thinkingDiv.id = 'thinking-indicator';
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


/* ---------- VOICE I/O (STT & TTS) ---------- */

// 1. Text-to-Speech (TTS) Setup - COMPLETELY PRESERVED
let ttsEnabled = false;
const ttsToggle = document.getElementById('tts-toggle');
const synth = window.speechSynthesis;

if (ttsToggle) {
    ttsToggle.addEventListener('click', () => {
        ttsEnabled = !ttsEnabled;
        ttsToggle.textContent = ttsEnabled ? "🔊 TTS: ON" : "🔊 TTS: OFF";
        ttsToggle.style.background = ttsEnabled ? "#042b46" : "transparent";
    });
}

function speakText(text) {
    if (!ttsEnabled || !synth) return;
    const cleanText = text.replace(/[*_~`#>-]/g, '');
    const utterance = new SpeechSynthesisUtterance(cleanText);
    utterance.rate = 1.0;
    utterance.pitch = 1.0;
    
    const voices = synth.getVoices();
    const preferredVoice = voices.find(v => v.lang.includes('en-GB') || v.lang.includes('en-US'));
    if (preferredVoice) utterance.voice = preferredVoice;

    synth.speak(utterance);
}

// 2. Binary Socket.IO Audio Recorder Pipeline
let mediaRecorder = null;
let audioChunks = [];
let isRecording = false;

if (micButton) {
    micButton.addEventListener('click', async () => {
        if (isRecording) {
            mediaRecorder.stop();
        } else {
            audioChunks = [];
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                mediaRecorder = new MediaRecorder(stream);
                
                mediaRecorder.ondataavailable = (event) => {
                    if (event.data.size > 0) audioChunks.push(event.data);
                };
                
                mediaRecorder.onstart = () => {
                    isRecording = true;
                    micButton.classList.add('recording');
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