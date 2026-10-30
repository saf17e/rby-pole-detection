# Coloured pole detection (red / blue / yellow)

An underwater pole detector for Team Nirma AUV.

**Pipeline:** YOLO finds pole centres → SAM segments each pole from those points → each mask is classified by colour, after boosting the red channel and applying CLAHE (water absorbs red first). There are versions for a USB webcam (`logitech_version.py`) and an OAK camera with Depth Anything V2 depth (`main.py`). `wall_depth/` is an experiment in getting metric depth from a known wall.

This is a research prototype and didn't run on the vehicle. Model checkpoints (YOLO, SAM ViT-H, Depth Anything V2) and the Depth Anything V2 code aren't included.
