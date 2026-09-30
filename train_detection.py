import os
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from sgp4.api import Satrec, jday
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE

EARTH_RADIUS = 6371.0
MU = 398600.4418

NEGATIVE_LABEL_FILES = ['active-sate.txt']
POSITIVE_LABEL_FILES = [
    'cosmos-1408-debris.txt',
    'cosmos-2251-debris.txt',
    'fengyun-1c-debris.txt',
    'iridium-33-debris.txt'
]
TEST_FILES = ['iridium-33-debris.txt']

def parse_tle_file(tle_file_path, label, data_dir='data'):
    data = []
    labels = []
    fp = os.path.join(data_dir, tle_file_path)

    if not os.path.exists(fp):
        raise FileNotFoundError(f"File not found: {fp}")

    with open(fp, 'r', encoding='utf-8', errors='ignore') as file:
        lines = file.readlines()

    now = datetime.now(timezone.utc)
    jd, fr = jday(now.year, now.month, now.day, now.hour, now.minute, now.second)

    for i in range(0, len(lines), 3):
        if i + 2 >= len(lines):
            break

        line1 = lines[i + 1].strip()
        line2 = lines[i + 2].strip()

        try:
            satellite = Satrec.twoline2rv(line1, line2)
            eccentricity = satellite.ecco
            inclination = satellite.inclo
            raan = satellite.nodeo
            arg_perigee = satellite.argpo
            mean_anomaly = satellite.mo
            mean_motion = satellite.no_kozai

            denom = 1.0 - satellite.ecco
            if abs(denom) < 1e-6:
                denom = 1e-6
            semi_major_axis = 6378.1 / denom
            altitude = semi_major_axis - EARTH_RADIUS
            perigee = semi_major_axis * (1.0 - eccentricity)
            apogee = semi_major_axis * (1.0 + eccentricity)
            orbital_period = 86400.0 / mean_motion if mean_motion > 0 else 0.0

            e, r, v = satellite.sgp4(jd, fr)

            if e != 0 or np.isnan(r).any() or np.isnan(v).any():
                r = (0.0, 0.0, 0.0) if np.isnan(r).any() else r
                v = (0.0, 0.0, 0.0) if np.isnan(v).any() else v

            distance_from_center = np.sqrt(r[0]**2 + r[1]**2 + r[2]**2)
            velocity_magnitude = np.sqrt(v[0]**2 + v[1]**2 + v[2]**2)

            if distance_from_center > 0:
                specific_orbital_energy = (velocity_magnitude**2) / 2.0 - (MU / distance_from_center)
            else:
                specific_orbital_energy = 0.0

            data.append({
                'eccentricity': eccentricity,
                'inclination': inclination,
                'raan': raan,
                'arg_perigee': arg_perigee,
                'mean_anomaly': mean_anomaly,
                'mean_motion': mean_motion,
                'semi_major_axis': semi_major_axis,
                'x': r[0],
                'y': r[1],
                'z': r[2],
                'vx': v[0],
                'vy': v[1],
                'vz': v[2],
                'altitude': altitude,
                'perigee': perigee,
                'apogee': apogee,
                'orbital_period': orbital_period,
                'distance_from_center': distance_from_center,
                'specific_orbital_energy': specific_orbital_energy,
                'velocity_magnitude': velocity_magnitude,
                'label': label
            })
            labels.append(label)
        except Exception:
            continue

    return data, labels

def main():
    print("=" * 60)
    print("   Space Debris Detection and Trajectory Prediction Pipeline")
    print("=" * 60)

    print("\n[1/4] Parsing TLE dataset from 'data/' directory...")
    all_rows = []
    all_labels = []
    positive_count = 0
    negative_count = 0

    for file_path in POSITIVE_LABEL_FILES:
        d, l = parse_tle_file(file_path, 1)
        all_rows.extend(d)
        all_labels.extend(l)
        positive_count += len(d)
        print(f"  + Loaded {len(d):>5} debris records from {file_path}")

    for file_path in NEGATIVE_LABEL_FILES:
        d, l = parse_tle_file(file_path, 0)
        all_rows.extend(d)
        all_labels.extend(l)
        negative_count += len(d)
        print(f"  - Loaded {len(d):>5} active satellite records from {file_path}")

    print(f"\nTotal Dataset Summary:")
    print(f"  Debris Objects (Class 1)    : {positive_count}")
    print(f"  Active Satellites (Class 0) : {negative_count}")
    print(f"  Total records               : {len(all_rows)}")

    df = pd.DataFrame(all_rows)

    print("\n[2/4] Preprocessing & balancing dataset...")
    X = df.drop(columns=['label'])
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"  Training set size: {len(X_train)} | Test set size: {len(X_test)}")
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    print(f"  Balanced training set size: {len(X_train_balanced)}")

    print("\n[3/4] Training Random Forest Classifier (100 estimators)...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42, oob_score=True, n_jobs=-1)
    clf.fit(X_train_balanced, y_train_balanced)
    print(f"  Training complete! Out-Of-Bag (OOB) Score: {clf.oob_score_:.4f}")

    print("\n[4/4] Evaluating Model Performance...")
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n>> Test Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Active Satellite (0)', 'Debris (1)']))

    print("Running 5-Fold Cross Validation...")
    train_scores = cross_val_score(clf, X_train, y_train, cv=5, n_jobs=-1)
    test_scores = cross_val_score(clf, X_test, y_test, cv=5, n_jobs=-1)
    print(f"  Mean Training CV Score   : {np.mean(train_scores):.4f}")
    print(f"  Mean Validation CV Score : {np.mean(test_scores):.4f}")

    print(f"\nEvaluating on holdout test file ({TEST_FILES[0]})...")
    holdout_data, holdout_l = parse_tle_file(TEST_FILES[0], 1)
    holdout_df = pd.DataFrame(holdout_data).drop(columns=['label'])
    holdout_pred = clf.predict(holdout_df)
    holdout_acc = accuracy_score(holdout_l, holdout_pred)
    print(f">> Accuracy on {TEST_FILES[0]}: {holdout_acc * 100:.2f}%")

    print("\n" + "=" * 60)
    print("Pipeline execution completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()
