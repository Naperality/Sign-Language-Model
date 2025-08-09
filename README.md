## Sign Language Model Project

Uses LSTM for ASL action recognition

## 📁 Dataset Folder Structure in Google

dataset/
   ├── hello/
   │     ├── hello_1.mp4
   │     ├── hello_2.mp4
   │     ├── ...
   │
   ├── thank_you/
   │     ├── thank_you_1.mp4
   │     ├── thank_you_2.mp4
   │     ├── ...
   │
   ├── yes/
   │     ├── yes_1.mp4
   │     ├── yes_2.mp4
   │     ├── ...
   │
   └── no/
         ├── no_1.mp4
         ├── no_2.mp4
         ├── ...

### NOTE
- 🎥 How Many Videos per Action?
-- Minimum: 30–50 videos per action
- ⏱ Video Length
-- Ideal length: 2–4 seconds per video
-- Keep frame rate at 30 FPS (default in most devices)
- 📌 Tips for Recording
-- Vary background and lighting slightly to make the model robust.
-- Record from slightly different positions/angles.
-- Include different people performing the sign if possible.
-- Keep hands fully in frame — especially important for two-hand signs.
-- For two-hand detection, make sure both hands are clearly visible in most frames.