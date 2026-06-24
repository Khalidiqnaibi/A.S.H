# Executive summary (one-paragraph)

Build ASH as a set of modular services running on a high-end local PC (offline-first) and a set of real-time microcontrollers for actuation. Use ROS2 for robot IPC, containerized model servers (vLLM/TGI or local ggml fallback), whisper.cpp for ASR, Coqui/Bark for TTS, and a local vector DB for memory. Provide a `face-runtime` (Godot or WebGL) for expressions and lip-sync. Protect hardware with watchdogs, E-stop, and a separate power distribution PCB. Start by proving personality (speech + face + single joint) then incrementally add sensors, more DoFs, and production-grade PCBs/OS pieces.

---

# Top-level architecture (text diagram)

```
Sensors (mic array, cameras, touch)  -> Perception Layer -> Router/Intent Classifier -> Agent (ASH core)
                                                              |
                                                              v
                            Memory / Vector DB  <-  Agent  -> Tools (calculator, time, vision tools)
                                                              |
                                                              v
                                                      Planner / Action
                                                              |
                                                              v
                ROS2 Topics/Actions -> MCU (Teensy/STM32) -> Servo Drivers / Motor Controllers
                                                              |
                                                              v
                     Face Runtime (Godot) <-> TTS Node -> Audio Out + Speakers
                                                              |
                                                              v
                                        Power Management + Safety (E-stop, watchdog)
```

---

# Hardware: parts & BOM (3 tiers)

### Minimum-viable extraordinary (recommended starter to be extraordinary)

* **Main PC (onboard or base)**: Ryzen 9 7900X or Intel i9-13900K, **RTX 4080** GPU, 64GB RAM, 2TB NVMe.
* **MCU (real-time)**: Teensy 4.1 or STM32F4 dev board.
* **Servos (expressive)**: 2 × Dynamixel XM430 (for head tilt + arm) — high quality, feedback.
* **Camera**: Intel RealSense D435i (RGB + depth + IMU).
* **Microphone**: ReSpeaker 4-mic HAT (or USB 4-mic array).
* **Display**: 7" IPS HDMI touch.
* **Speakers**: 2W–10W small amp + speaker.
* **LEDs**: NeoPixel ring (12–24 pixels).
* **Power**: 12–24V Li-ion battery pack + BMS + DC-DC converters (12→5V, 12→3.3V).
* **Servo driver (if using hobby servos)**: PCA9685 or Dynamixel bus.
* **Chassis / body**: 3D printed shell + mounting hardware.
* **Cooling**: small fan(s) and heat-sinks for GPU/CPU if internalized.

### Pro (more robust / mobile / advanced)

* RTX 4090 or dual GPU setup, Jetson Orin NX for edge nodes, extra Dynamixel for arms, LIDAR (Hokuyo/RPLidar), UPS/UPS HAT.

### Budget estimates (ballpark)

* Minimum-viable extraordinary: **$5k–8k** (GPU + PC are biggest cost).
* Pro/production prototype: **$12k–30k+** depending on parts and servos.

---

# Electrical & mechanical guidance (PCB & wiring)

* **Power distribution board (PDB)**: design a simple PCB with:

  * Main battery input → BMS connector.
  * Fused rails for servos (5–12V) and PC (ATX-style supply).
  * Current sense shunt for motor rail (for stall detection).
  * E-stop relay that cuts servo power (mechanical relay or high-current MOSFET).
  * JST connectors for servos, sensors, and MCU power.
* **MCU connection**:

  * MCU to PC via USB (serial) for dev; use UART or CAN for production (CAN recommended for robust multi-actuator).
  * MCU heartbeat to PC + PC heartbeat back — if either missed, MCU kills motor power.
* **Signal isolation**:

  * Use logic-level shifting if interfacing 3.3V MCU with 5V servos.
  * Add opto-isolation for motor drivers if you expect noisy environments.
* **Cable management**: keep power and signal wiring separate; use ferrite beads and decoupling capacitors near servos.

---

# Software stack (services & runtimes)

### Base OS

* **Ubuntu LTS** (24.04 recommended) — best GPU driver and toolchain support.
* Optional: Ubuntu Core/Yocto if you later move to custom OS.

### Container & orchestration

* **Docker** or **Podman** — run each service isolated:

  * `agent-service` (ASH/LangGraph wrapper)
  * `model-server` (vLLM/TGI or local ggml/llama.cpp)
  * `asr-service` (whisper.cpp wrapper)
  * `tts-service` (Coqui/Bark)
  * `vision-service` (OpenCV/YOLO)
  * `memory-service` (Milvus/Chroma)
  * `face-runtime` service (if using Web face)
* Manage startup with `systemd` or `docker-compose`.

### Robot IPC

* **ROS2 Humble/Galactic** for topics/actions (hardware-level), ROS2 for sensors & actuators.
* Use *DDS* in ROS2 for local low-latency comms.

### Inference servers

* GPU host: **vLLM** or **Text Generation Inference (TGI)** for large models.
* CPU fallback: **llama.cpp** / **ggml** quantized models.
* For embeddings: use a compact transformer embedding (all-MiniLM) on CPU.

### Storage & DB

* **Milvus** or **Chroma** local for vector memory (store embeddings & metadata).
* **Postgres** or **SQLite** for config, logs, and structured events.
* **S3-compatible** storage (minio) for large model files and backups.

---

# Core modules & responsibilities

1. **Perception Layer**

   * ASR: whisper.cpp (offline) with VAD + beamforming from mic array.
   * Vision: YOLOv8 / OpenCV for object detection; RealSense for depth/SLAM.
   * OCR: PaddleOCR for reading screens/labels.

2. **Router / Intent Classifier**

   * Small classifier that chooses mode: `chat`, `game`, `story`, `utility`, `vision_scan`, etc.
   * Lightweight (tiny transformer or rule-based + small LLM fallback).

3. **Agent (ASH core)**

   * The personality LLM + tool invoker + planner. Wrap this as `agent-service`.
   * Accepts `POST /think` with context and tools allowed, returns `Final Answer` + `Actions`.

4. **Emotion Engine**

   * Maintains `emotion_state` (mood, energy, attachment). Tools to `get_emo`, `update_emo`.
   * Drives face expressions and modifies prompt/response tone.

5. **Memory Service**

   * Stores events, embeddings, and meta. Provides `recall(query)` that returns top-K memories to prepend to LLM prompt.

6. **Face Runtime**

   * Godot-based engine or HTML5 WebGL. Listens on WebSocket for:

     * `set_expression`, `play_animation`, `phoneme_sync`, `eye_direction`.
   * Receives viseme timings from TTS to lip-sync.

7. **Planner + Actuation**

   * Translate high-level actions → ROS2 actions → MCU commands (servo position, speed).
   * Action safety checks & pre-flight simulation (limit ranges, torque).

8. **Safety**

   * E-stop hardware, watchdog MCU, current-limits, and soft-limits in software.

---

# Personality & prompt design (BMO-like persona)

Use prompt injection carefully; keep safety constraints.

**Persona template (compact)**:

```
You are "ASH" — friendly, curious, playful, childlike but wise. 
Rules:
- Always call the user "friend" or nickname if known.
- Stay cheerful and imaginative; use short playful sentences.
- If asked for a factual answer, be accurate and cite tools (if available).
- Do not fabricate facts — if unsure, say "I don't know".
- Use emotion_state to modulate language: {mood}, {energy}.
- If needed, call tools using "Action:" / "Action Input:" format.
```

You can keep this as a `bmo_personality.txt` loaded into the LLM prompt template with top-p/temperature tuning.

---

# Emotion engine (code sketch)

```python
# emotion_engine.py
from enum import Enum
import json, time

class Mood(Enum):
    HAPPY='happy'
    SAD='sad'
    CURIOUS='curious'
    ANGRY='angry'
    TIRED='tired'

class EmotionEngine:
    def __init__(self, filepath='emotion_state.json'):
        self.filepath = filepath
        try:
            with open(self.filepath,'r') as f:
                self.state = json.load(f)
        except:
            self.state = {'mood':Mood.HAPPY.value,'energy':80,'attachment':50,'last':time.time()}
            self._save()

    def _save(self):
        with open(self.filepath,'w') as f:
            json.dump(self.state,f)

    def get(self):
        return self.state

    def update(self, delta):
        # delta is dict like {'energy': -10, 'attachment': +5}
        for k,v in delta.items():
            if k in self.state:
                self.state[k] = max(0, min(100, self.state[k]+v))
        self.state['last'] = time.time()
        self._save()
        return self.state

    def set_mood_by_rules(self):
        # simple rule-based override
        if self.state['energy'] < 20: self.state['mood'] = Mood.TIRED.value
        elif self.state['attachment'] > 80: self.state['mood'] = Mood.HAPPY.value
        self._save()
```

Expose `get_emo`, `update_emo` as tools callable by the agent.

---

# Memory schema & example usage

* **Memory entry**:

```json
{
  "id":"uuid",
  "type":"event|fact|preference",
  "text":"Went to the park with friend; likes chocolate ice cream.",
  "embedding":[...],
  "timestamp":"ISO8601",
  "importance": 0-1
}
```

* **Workflow**:

  * On each conversation turn, embed the user utterance → store in memory with `importance` based on emotional weight or explicit user request to remember.
  * On new queries, recall `top_k` relevant memories and include them in prompt.

---

# Minimal docker-compose (offline stack)

```yaml
version: '3.8'
services:
  agent:
    image: yourorg/ash-agent:latest
    restart: unless-stopped
    volumes:
      - ./models:/models
      - ./data:/data
    ports:
      - "9000:9000"

  asr:
    image: ghcr.io/your/whisper-wrapper:latest
    volumes:
      - ./models/whisper:/models/whisper

  tts:
    image: yourorg/coqui-tts:latest
    ports:
      - "5002:5002"

  memory:
    image: milvusdb/milvus:latest
    ports:
      - "19530:19530"
    volumes:
      - ./milvus_data:/var/lib/milvus
```

Run with `docker compose up -d` on your robot network (no egress).

---

# Essential ROS2 nodes (skeleton)

### ROS2 Agent Node — `bmo_agent_node.py`

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from your_agent import ash  # import your ASH instance

class BMONode(Node):
    def __init__(self):
        super().__init__('bmo_agent_node')
        self.sub = self.create_subscription(String,'/bmo/transcript',self.on_transcript,10)
        self.pub_speak = self.create_publisher(String,'/bmo/speak',10)
        self.pub_action = self.create_publisher(String,'/bmo/action',10)

    def on_transcript(self, msg):
        query = msg.data
        self.get_logger().info(f"Transcript: {query}")
        response = ash.run(query)  # must return dict {text, actions}
        # publish speech
        out = String(); out.data = response.get('text','')
        self.pub_speak.publish(out)
        # publish action command if exists
        act = String(); act.data = response.get('action','')
        self.pub_action.publish(act)

def main(args=None):
    rclpy.init(args=args)
    node = BMONode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
```

---

# Development workflow & CI

* Git repo with services as subfolders and Dockerfiles.
* Use `pre-commit` and linting for Python.
* Automated tests: unit tests for emotion engine, memory recall, and fake-agent runs.
* Hardware-in-the-loop (HIL) tests: simulate motor responses in software before running on real hardware.

---

# Safety, privacy & security (must-haves)

* E-stop physical button that kills motor power instantly.
* Watchdog on MCU that requires periodic heartbeat from host.
* System network policy: **block egress by default** (iptables), allow local-only traffic.
* Encrypt stored audio/video logs on disk; provide user commands to delete logs.
* Signed updates: use GPG or secure boot for production OTA.
* Privacy UI: toggles to disable voice/video logging, cloud features, and network access.

---

# Testing & measurement checklist

* Power: measure stall current of servos; size fuses accordingly.
* Thermal: monitor CPU/GPU temps under long-run demos.
* Latency: measure ASR → agent → TTS end-to-end; aim <500ms for conversational feel.
* Safety: test watchdog by killing agent process and verifying MCU kills motors.
* Memory: test recall accuracy via unit tests and human evaluation.

---

# Build roadmap (12–week focused plan, high-velocity)

### Week 0–1 — Prep & baseline

* Order PC + core parts. Install Ubuntu, Docker, ROS2. Place sketch image in docs: `/mnt/data/rn_image_picker_lib_temp_2dea8f29-17c8-468e-a925-0ba9613f7b0d.jpg`.
* Create repo skeleton and Docker Compose.

### Week 2–3 — Face + TTS + ASR (demo)

* Implement Godot face that listens on WebSocket. Hook to Coqui TTS for phoneme timing.
* Set up whisper.cpp for mic input; publish transcripts to ROS2 topic.

### Week 4–5 — Single-DOF actuation & MCU

* MCU firmware: serial command handler, heartbeat, E-stop handler.
* ROS2 bridge to MCU; demo head-tilt + blink + speak reaction.

### Week 6–7 — Agent & memory

* Wrap ASH into `agent-service` with personality prompt and emotion engine tools.
* Implement local Milvus/Chroma and memory save/recall.

### Week 8–9 — Vision & depth behaviors

* Integrate RealSense; simple object following or approach behavior.

### Week 10–12 — Polish + safety + demo video

* Polish animations, tune prompts, bake sample games, record high-quality demo video.

