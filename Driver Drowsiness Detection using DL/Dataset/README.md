# UTA-RLDD dataset notes

The official UTA-RLDD page describes approximately 30 hours of RGB video from
60 healthy participants: one approximately 10-minute video per participant for
each of three participant-provided video-level classes:

- `alert` (`0`)
- `low vigilant` (`5`)
- `drowsy` (`10`)

The official page reports 180 videos, five subject folds of 12 participants,
and a total size of 111.3 GB. It provides a Google Drive access link, but this
project does not download the dataset automatically. Obtain access and place
the data locally yourself.

Sources:

- [Official UTA-RLDD documentation](https://sites.google.com/prod/view/utarldd/home)
- [UTA-RLDD paper](https://arxiv.org/abs/1904.07312)

The official documentation does **not** define a guaranteed folder/file naming
schema in the published page, nor does it provide frame-level yawning or
microsleep event annotations. Therefore, preprocessing requires a manifest
created from the files you obtained. Do not infer event labels from the
video-level drowsiness class.

Required manifest columns:

```csv
subject_id,video_id,label,video_path
subject-001,subject-001-alert,alert,fold1/subject-001-alert.mp4
```

`video_path` is relative to `--dataset-root` or may be absolute. Labels must
be `alert`, `low vigilant`, or `drowsy` (the numeric aliases `0`, `5`, and `10`
from the official page are also accepted and canonicalized). Event flags are
intentionally not accepted by this pipeline.

## Preprocessing

Run the synthetic smoke fixture first:

```powershell
.\.venv\Scripts\python.exe ".\Driver Drowsiness Detection using DL\preprocess_dataset.py" `
  --smoke `
  --output ".\Driver Drowsiness Detection using DL\smoke_features.npz"
```

For locally obtained data:

```powershell
.\.venv\Scripts\python.exe ".\Driver Drowsiness Detection using DL\preprocess_dataset.py" `
  --dataset-root "D:\datasets\uta-rldd" `
  --manifest "D:\datasets\uta-rldd\manifest.csv" `
  --model ".\Driver Drowsiness Detection using DL\models\face_landmarker.task" `
  --output ".\Driver Drowsiness Detection using DL\features.npz"
```

The pipeline uses one detector, extracts timestamped EAR/MAR values, resets
windows across invalid tracking gaps, samples windows at an explicit target
FPS, assigns subjects before feature normalization, fits normalization on
training windows only, and writes `features.npz` plus `features.json`.
