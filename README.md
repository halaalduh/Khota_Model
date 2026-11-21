# Khota ML

Khota is our graduation project at King Saud University. The app lets people report sidewalk problems in Riyadh. This repo contains the model code that checks the photos submitted with those reports.

## How it works

1. The YOLO detectors look for two types of problems in the full photo: curb ramp issues and surface damage.
2. We combine detections that overlap. Each detected area is cropped from the original photo, with 20% extra space on each side.
3. The ResNet-18 classifiers use the crop to rate the defect as low/medium or high severity.

The final version combines seven YOLO detectors and six ResNet-18 classifiers.

## Test results

The test photos came from locations that were not used for training.

| What we measured | Result |
| --- | ---: |
| Curb ramp issue detection (AP@0.5) | 61% |
| Surface damage detection (AP@0.5) | 17% |
| Severity classification (Macro-F1) | 75% |
| High-severity recall | 71% (17 out of 24) |
| Photos with a labeled defect that got at least one detection | 56% |
| Photos without a labeled defect that got a false alarm | 9% |

The model did better at finding curb ramp issues than surface damage. Surface damage is the main area we want to improve.

## Files

- [`khota/`](khota/) has the code for detection, combining predictions, cropping, and severity classification.
- [`weights/pipeline.json`](weights/pipeline.json) has the model settings and thresholds.
- [`examples/sample_response.json`](examples/sample_response.json) shows what the model returned for one photo. It found a curb ramp issue and a surface defect and rated both as high severity.

Running the model requires the trained `.pt` weight files.
