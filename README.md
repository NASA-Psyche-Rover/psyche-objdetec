# psyche-objdetec

Autonomous rover perception navigation system focused on Psyche asteroid traversal under compute constraints. 

The system combines: 
- A YOLOv8-based computer vision system that detects clusters of obstacles
on asteroid Psyche and decides whether the rover should proceed or stop.
- A MiDaS-based monocular depth estimation pipeline that converts depth maps
into terrain risk signals for real-time rover navigation decision-making under compute constraints.

## Features
- Real-time object detection using YOLOv8 and monocular depth estimation using MiDaS
- Cluster density analysis
- Terrain risk estimation from depth maps
- Supports navigation decisioning (PROCEED / STOP)
- Optimized inference pipeline for edge deployment (CPU-constrained environments)
- Easily extendable to ROS or simulated environments (Unity)

## Current Status
- YOLOv8 object detection pipeline implemented
- MiDaS monocular depth estimation pipeline implemented
- Multithreaded producer-consumer inference architecture completed
- ONNX Runtime optimization completed
- Raspberry Pi benchmarking in progress

## Overview
The system processes live webcam input, runs object detection and depth estimation in parallel and generates navigation decision based on terrain depth and objects detected. 

Live display shows:
- Navigation decision (GO / STOP)  
- Terrain risk score  
- FPS  
- Detected object labels  

A separate benchmarking pipeline evaluates MiDaS inference performance across PyTorch, ONNX Runtime, and OpenVINO backends to analyze tradeoffs between latency, throughput, and model size for edge deployment.


## System Architecture

```text
Camera Stream
      │
      V
Frame Queue
      │
      ├──────────────> Object Detection (YOLOv8)
      │
      └──────────────> Depth Estimation (MiDaS)
                               │
                               V
                       Terrain Risk Analysis
                               │
                               V
                      Navigation Decision

```
## Design Decisions
- YOLOv8n: Selected for its balance of accuracy and inference, compatible with raspberry pi hardware
- MiDAS-small: Selected for relative depth estimation to supprt terrain understanding without requiring metric depth accuracy
- ONNX Runtime: Selected for its ease in improving inference latency and CPU efficiency compared to PyTorch

### Performance Evaluation

Measured on ARM64 (Apple Silicon, dev environment)
- Depth inference latency: 465 ms to 45 ms (ONNX optimization)
- Perception throughput: 2 FPS to 22 FPS
- Frame-drop rate: 92.6% to 25.7%
- Backend comparison: PyTorch, ONNX Runtime, OpenVINO

## Future Work
- Benchmark full system performance on Raspberry Pi 5 hardware
- Evaluate object detection accuracy (mAP, precision, recall)
- Explore additional lightweight depth models for edge deployment
- Integrate LiDAR-based depth validation
- Replace heuristic thresholds with learned navigation policies
- Extend profiling to include power and memory usage metrics

## Usage
```bash
pip install -r requirements.txt
python main.py
