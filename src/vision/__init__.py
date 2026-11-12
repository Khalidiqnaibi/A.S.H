"""
src/vision -- what ASH can recognise through the camera, kept as labels and numbers, never as pictures.

    memory     who and what it has been shown (labels + embeddings), and cautious matching
    backends   the models: faces (OpenCV YuNet/SFace), objects (an ONNX detector), and a light embedder for things
    eye        one frame, on request, from the camera -- never stored
"""
