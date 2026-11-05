# Driver Drowsiness Detection using MediaPipe and LSTM

This project is being built in milestones for issue [#1199](https://github.com/abhisheks008/DL-Simplified/issues/1199).
The first two milestones provide a local facial-landmark demo and a cautious
EAR/MAR temporal rule baseline. It uses MediaPipe Tasks Face Landmarker with
the bundled `models/face_landmarker.task` asset.

## Setup

From the repository root, activate the existing environment:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r ".\Driver Drowsiness Detection using DL\requirements.txt"
```

The model asset is not downloaded by the application. Keep it at
`models/face_landmarker.task`. Its provenance and license should be checked
before redistribution; MediaPipe Tasks is distributed by Google under the
Apache 2.0 license.

## Run the landmark demo

```powershell
python ".\Driver Drowsiness Detection using DL\landmark_demo.py"
```

Use a recorded video instead:

```powershell
python ".\Driver Drowsiness Detection using DL\landmark_demo.py" `
  --input "C:\path\to\video.mp4"
```

Press `Q` to stop. Green outlines show the selected eye landmarks and the
orange outline shows the selected mouth landmarks. The status text distinguishes
an available face track from a frame where no reliable face was found. The
displayed FPS is measured processing throughput, not a claim about camera FPS.

The demo also displays:

- **EAR**: the average eye aspect ratio from six landmarks per eye.
- **MAR**: `(distance(0,17) + distance(13,14)) / (2 * distance(61,291))`.
- A provisional rule status. Sustained eye closure is reported as a rule alert,
  while sustained mouth opening is explicitly not treated as confirmed yawning.

Thresholds and durations are configurable, for example:

```powershell
python ".\Driver Drowsiness Detection using DL\landmark_demo.py" `
  --ear-threshold 0.21 --mar-threshold 0.60 `
  --closure-seconds 1.5 --mouth-seconds 1.0 --alert-cooldown 5
```

EAR and MAR are calculated after converting normalized coordinates to pixels,
which avoids distorting distances when the camera frame is not square. Missing
or invalid measurements produce `UNABLE TO ASSESS`; they are not replaced with
zero and are not carried forward. Tracking gaps reset sustained-condition
timers. The thresholds are provisional demonstration values, not clinical
limits, and the rule output is not a diagnosis or safety guarantee.

## Tests

```powershell
python -m unittest discover -s ".\Driver Drowsiness Detection using DL\tests"
```

The existing `webcam_test.py` remains a minimal camera-only smoke test.
No webcam footage or dataset is recorded by this project.
