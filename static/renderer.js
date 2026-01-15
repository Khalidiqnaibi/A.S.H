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

socket.on("system", (data) => {
  logAdd(data.msg, "system");
});

socket.on("ash_response", (data) => {
  kprint(data.text);
});


/* ---------- SEND MESSAGE ---------- */

sendButton.addEventListener('click', () => {

  const message = inputField.value.trim();
  inputField.value = '';

  if (!message) return;

  logAdd(`${user ? user : 'User'}: ${message}`,'user');

  socket.emit("user_message", {
    user: user,
    message: message
  });
});

inputField.addEventListener('keyup', (event) => {
  if (event.key === 'Enter') {
    sendButton.click();
  }
});
