const user =  window.USERNAME || null;

const chatLog = document.getElementById('chat-log');
const inputField = document.getElementById('input-field');
const sendButton = document.getElementById('send-button');

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
