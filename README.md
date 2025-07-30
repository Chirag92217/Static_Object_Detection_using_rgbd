🧠 Static Object Detection using RGB-D Data
This project implements static object detection using RGB and Depth data collected via the Intel RealSense camera. Three different fusion strategies—early, late, and hybrid—are implemented and compared for performance.

📸 Data Collection
Device: Intel RealSense depth camera

Modalities Captured:

RGB frames (.png)

Depth frames (.png)

Structure:

markdown
Copy
Edit
dataset/
└── class_name/

      ├── rgb/
      
      └── depth/
🏷️ Annotation
Tool Used: Roboflow

RGB frames annotated with bounding boxes

Labels synced with corresponding depth frames

🔀 Fusion Techniques
Early Fusion: Combines RGB and depth at the input level (4 channels).

Late Fusion: Separate RGB and depth feature extractors, fused before classification.


📊 Comparison
 The comparision have been made on three models which weere trained on rgbd(early fusion, late fusion) and rgb only 
| Model    |   Accuracy(%) |   Precision(%) |   Recall(%) |   F1(%) |   Lat(ms/img) |   FPS |   VRAM(GB) |   Params(M) |   FLOPs(G) |
|:---------|--------------:|---------------:|------------:|--------:|--------------:|------:|-----------:|------------:|-----------:|
| RGB-only |          97.3 |           97.3 |        97.4 |    97.3 |          12.2 |  81.8 |       3.94 |        41.1 |      177.6 |
| Early    |          97.9 |           97.9 |        98   |    97.9 |          11.5 |  86.6 |       3.95 |        41.1 |      178.3 |
| Mid      |          97.9 |           98   |        98   |    97.9 |          15.7 |  63.6 |       5.21 |        65.9 |      245.5 |
