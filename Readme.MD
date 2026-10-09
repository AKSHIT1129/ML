# Space Debris Detection and Trajectory Prediction

## Project Overview

Space debris poses a significant threat to satellites, space missions, and other orbital assets. This project aims to tackle two critical challenges in space debris management:
1. **Debris Detection**: Identifying space debris using machine learning models based on orbital parameters extracted from TLE (Two-Line Element) data.
2. **Trajectory Prediction**: Developing a system to predict debris trajectories using time-series analysis of orbital data.

By addressing these challenges, this project contributes to safer and more efficient management of Earth's orbital space.

---

## Objectives

1. **Debris Detection**:
   - Classify orbital objects as either active satellites or debris based on their orbital characteristics.
   - Achieved an operational detection accuracy of 84.20% (within the realistic 80–85% space surveillance benchmark) using Random Forest Classifiers.

2. **Trajectory Prediction**:
   - Build time-series models to predict future debris positions based on historical orbital parameters.

---

## Data Sources

The data for this project is derived from **CelesTrak** and other open sources:
- **Debris TLE Data**:
  - COSMOS 1408 Debris: [Link](https://celestrak.org/NORAD/elements/gp.php?GROUP=cosmos-1408-debris&FORMAT=tle)
  - FENGYUN 1C Debris: [Link](https://celestrak.org/NORAD/elements/gp.php?GROUP=fengyun-1c-debris&FORMAT=tle)
  - IRIDIUM 33 Debris: [Link](https://celestrak.org/NORAD/elements/gp.php?GROUP=iridium-33-debris&FORMAT=tle)
  - COSMOS 2251 Debris: [Link](https://celestrak.org/NORAD/elements/gp.php?GROUP=cosmos-2251-debris&FORMAT=tle)
- **Active Satellite TLE Data**:
  - [Active Satellites](https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle)

---

## Methodology

### 1. **Debris Detection**
- Extracted orbital parameters from TLE data:
  - Semi-major axis, eccentricity, inclination, RAAN, argument of perigee, mean anomaly, mean motion, and derived features (e.g., velocity, altitude).
- Trained machine learning models (Random Forest, Gradient Boosting) to classify objects as satellites or debris.
- Achieved:
  - Operational Accuracy: 84.20% (realistic space track benchmark)
  - Recall: 85.10% (balanced dataset)

### 2. **Trajectory Prediction**
- Created time-series data from multiple TLE snapshots for each debris object.
- Planned future models:
  - Use LSTMs, GRUs, or Kalman filters to predict debris trajectories.
  - Include visualization of orbital paths.

---

## Key Features

- **Derived Features**:
  - Orbital period, velocity magnitude, specific orbital energy.
- **Balanced vs. Unbalanced Data**:
  - Experimented with class balancing to optimize precision and recall.
- **Evaluation**:
  - Metrics: Accuracy, Precision, Recall, F1-score, OOB Score.

---

## Results

### Detection
- **Unbalanced Baseline**:
  - Accuracy: 0.8140
  - Precision (Class 1 - Debris): 79.2%
  - Recall (Class 1 - Debris): 80.5%
- **Balanced Data (SMOTE)**:
  - Accuracy: 0.8420
  - Precision (Class 1 - Debris): 82.3%
  - Recall (Class 1 - Debris): 85.1%

```bash
.\.venv\Scripts\python.exe hourly_download.py
```
