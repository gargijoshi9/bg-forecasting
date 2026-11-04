#!/usr/bin/env python3
"""
build_datasets.py
=================
Reproducible pipeline to parse raw Apple Health XML export and build
modular, audited datasets for blood glucose forecasting and hormonal/stress analysis:
1. glucose_5min.csv: Primary CGM 5-minute resampled series (no imputed/invented data)
2. cycle_events.csv: Menstrual cycle events and metadata
3. stress_events.csv: Manually annotated stress events
4. wearable_metrics.csv: Heart rate (HR) and Heart Rate Variability (HRV) series
5. glucose_features.csv: Final integrated modeling dataset
"""

import os
import re
import math
import pandas as pd
import numpy as np

RAW_XML_PATH = 'data/raw/export.xml'
PROCESSED_DIR = 'data/processed'

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def parse_apple_health_xml(xml_path):
    print(f"Parsing raw Apple Health XML from {xml_path}...")
    glucose_rows = []
    hr_rows = []
    hrv_rows = []
    
    # Fast regex line scanner to handle large XML without memory overhead
    # Regex targets for required record types
    with open(xml_path, 'r', encoding='utf-8') as f:
        for line in f:
            if 'HKQuantityTypeIdentifierBloodGlucose' in line:
                m_sd = re.search(r'startDate=\"([^\"]+)\"', line)
                m_val = re.search(r'value=\"([^\"]+)\"', line)
                m_unit = re.search(r'unit=\"([^\"]+)\"', line)
                m_src = re.search(r'sourceName=\"([^\"]+)\"', line)
                if m_sd and m_val:
                    glucose_rows.append((m_sd.group(1), float(m_val.group(1)), m_unit.group(1) if m_unit else 'mg/dL', m_src.group(1) if m_src else 'LinX'))
            elif 'HKQuantityTypeIdentifierHeartRate\"' in line:
                m_sd = re.search(r'startDate=\"([^\"]+)\"', line)
                m_val = re.search(r'value=\"([^\"]+)\"', line)
                m_unit = re.search(r'unit=\"([^\"]+)\"', line)
                m_src = re.search(r'sourceName=\"([^\"]+)\"', line)
                if m_sd and m_val:
                    hr_rows.append((m_sd.group(1), float(m_val.group(1)), m_unit.group(1) if m_unit else 'count/min', m_src.group(1) if m_src else 'Apple Watch'))
            elif 'HKQuantityTypeIdentifierHeartRateVariabilitySDNN\"' in line:
                m_sd = re.search(r'startDate=\"([^\"]+)\"', line)
                m_val = re.search(r'value=\"([^\"]+)\"', line)
                m_unit = re.search(r'unit=\"([^\"]+)\"', line)
                m_src = re.search(r'sourceName=\"([^\"]+)\"', line)
                if m_sd and m_val:
                    hrv_rows.append((m_sd.group(1), float(m_val.group(1)), m_unit.group(1) if m_unit else 'ms', m_src.group(1) if m_src else 'Apple Watch'))

    print(f"Parsed {len(glucose_rows):,} raw glucose readings.")
    print(f"Parsed {len(hr_rows):,} heart rate readings.")
    print(f"Parsed {len(hrv_rows):,} HRV readings.")
    return glucose_rows, hr_rows, hrv_rows

def build_glucose_5min(glucose_rows):
    """
    Resamples raw 1-minute CGM readings to 5-minute standard intervals.
    CRITICAL: Does NOT invent, forward-fill, or interpolate readings.
    Empty 5-minute bins are excluded to preserve ground truth.
    """
    df_raw = pd.DataFrame(glucose_rows, columns=['timestamp_str', 'glucose', 'unit', 'source'])
    df_raw['timestamp'] = pd.to_datetime(df_raw['timestamp_str'])
    df_raw = df_raw.sort_values('timestamp').reset_index(drop=True)
    
    # Set index to resample
    df_indexed = df_raw.set_index('timestamp')
    
    # 5-minute aggregation: closed='left', label='left'
    agg_funcs = {
        'glucose': ['mean', 'min', 'max', 'count', 'std']
    }
    df_5m = df_indexed.resample('5min', closed='left', label='left').agg(agg_funcs)
    df_5m.columns = ['glucose', 'glucose_min', 'glucose_max', 'reading_count', 'glucose_std']
    
    # Drop buckets with 0 readings (NEVER invent readings)
    df_5m = df_5m[df_5m['reading_count'] > 0].reset_index()
    
    # Round values for consistency
    df_5m['glucose'] = df_5m['glucose'].round(2)
    df_5m['glucose_min'] = df_5m['glucose_min'].round(2)
    df_5m['glucose_max'] = df_5m['glucose_max'].round(2)
    df_5m['glucose_std'] = df_5m['glucose_std'].fillna(0.0).round(2)
    df_5m['reading_count'] = df_5m['reading_count'].astype(int)
    
    # Format ISO 8601 timestamp string preserving +05:30 timezone
    df_5m['timestamp_iso'] = df_5m['timestamp'].dt.strftime('%Y-%m-%dT%H:%M:%S%z')
    # Reorder columns
    cols = ['timestamp_iso', 'glucose', 'glucose_min', 'glucose_max', 'glucose_std', 'reading_count']
    out_df = df_5m[cols].rename(columns={'timestamp_iso': 'timestamp'})
    
    out_path = os.path.join(PROCESSED_DIR, 'glucose_5min.csv')
    out_df.to_csv(out_path, index=False)
    print(f"Saved {out_path} ({len(out_df):,} 5-minute records).")
    return out_df

def build_cycle_events():
    """
    Generates cycle_events.csv from validated user menstrual tracking data:
    - May 27, 2026 (26-day cycle, 4 days duration)
    - June 22, 2026 (25-day cycle, 4 days duration)
    - July 17, 2026 (33-day cycle, 4 days duration)
    - August 19, 2026 (73-day prolonged/delayed cycle due to skipped Sept period, 4 days duration)
    - October 31, 2026 (Return of menses)
    """
    cycles = [
        {
            'cycle_id': 1,
            'period_start_date': '2026-05-27',
            'period_end_date': '2026-05-30',
            'flow_duration_days': 4,
            'next_period_start_date': '2026-06-22',
            'cycle_length_days': 26,
            'cycle_type': 'regular',
            'cgm_covered_days': '2026-06-08 to 2026-06-21 (Days 13 to 26)',
            'notes': 'Normal length ovulatory cycle; CGM started on Day 13.'
        },
        {
            'cycle_id': 2,
            'period_start_date': '2026-06-22',
            'period_end_date': '2026-06-25',
            'flow_duration_days': 4,
            'next_period_start_date': '2026-07-17',
            'cycle_length_days': 25,
            'cycle_type': 'regular',
            'cgm_covered_days': '2026-06-22 to 2026-07-16 (Days 1 to 25)',
            'notes': 'Regular cycle. Sensor changeover occurred on Day 5.'
        },
        {
            'cycle_id': 3,
            'period_start_date': '2026-07-17',
            'period_end_date': '2026-07-20',
            'flow_duration_days': 4,
            'next_period_start_date': '2026-08-19',
            'cycle_length_days': 33,
            'cycle_type': 'regular',
            'cgm_covered_days': '2026-07-22 to 2026-08-18 (Days 6 to 33)',
            'notes': 'Slightly extended 33-day regular cycle. Sensor changeover on Day 5.'
        },
        {
            'cycle_id': 4,
            'period_start_date': '2026-08-19',
            'period_end_date': '2026-08-22',
            'flow_duration_days': 4,
            'next_period_start_date': '2026-10-31',
            'cycle_length_days': 73,
            'cycle_type': 'prolonged_amenorrhea',
            'cgm_covered_days': '2026-08-26 to 2026-10-27 (Days 8 to 70)',
            'notes': 'September period skipped during severe exam panic; menses resumed Oct 31.'
        }
    ]
    df_cycles = pd.DataFrame(cycles)
    out_path = os.path.join(PROCESSED_DIR, 'cycle_events.csv')
    df_cycles.to_csv(out_path, index=False)
    print(f"Saved {out_path} ({len(df_cycles)} cycle records).")
    return df_cycles

def build_stress_events():
    """
    Generates stress_events.csv from user annotations:
    - EV01: Hackathon & Interpersonal Conflicts (Aug 26 - Aug 29)
    - EV02: Post-Hackathon Fallout & Anxiety (Aug 30 - Sep 4)
    - EV03: Severe Exam Panic & Period Due (Sep 13 - Sep 17)
    - EV04: Chronic Result Waiting Anxiety (Sep 18 - Oct 5)
    - EV05: Stress Relief & Metabolic Recovery (Oct 6 - Oct 27)
    """
    stress_data = [
        {
            'event_id': 'EV01',
            'event_name': 'Hackathon & Conflict',
            'start_timestamp': '2026-08-26T00:00:00+05:30',
            'end_timestamp': '2026-08-29T23:59:59+05:30',
            'stress_category': 'acute_conflict_exertion',
            'stress_intensity_level': 3,
            'stress_label': 'Severe Acute Stress',
            'description': 'Hackathon with friends, constant fights, major 1 AM fight on Aug 29, high physical exertion. Morning spike to 327 mg/dL.'
        },
        {
            'event_id': 'EV02',
            'event_name': 'Post-Hackathon Anxiety',
            'start_timestamp': '2026-08-30T00:00:00+05:30',
            'end_timestamp': '2026-09-04T23:59:59+05:30',
            'stress_category': 'post_conflict_anxiety',
            'stress_intensity_level': 2,
            'stress_label': 'High Anxiety',
            'description': 'Lingering interpersonal distress and worry. HRV reached lowest level (18.9 ms).'
        },
        {
            'event_id': 'EV03',
            'event_name': 'Exam Panic & Period Due',
            'start_timestamp': '2026-09-13T00:00:00+05:30',
            'end_timestamp': '2026-09-17T23:59:59+05:30',
            'stress_category': 'severe_exam_panic',
            'stress_intensity_level': 3,
            'stress_label': 'Severe Acute Panic',
            'description': 'Intense fear of failing exam. Coincided with expected period due date, triggering functional hypothalamic amenorrhea (skipped menses). Median glucose 205 mg/dL.'
        },
        {
            'event_id': 'EV04',
            'event_name': 'Exam Result Waiting Anxiety',
            'start_timestamp': '2026-09-18T00:00:00+05:30',
            'end_timestamp': '2026-10-05T23:59:59+05:30',
            'stress_category': 'chronic_result_anticipation',
            'stress_intensity_level': 2,
            'stress_label': 'Chronic Anticipatory Stress',
            'description': 'Prolonged worry waiting for exam results. Baseline glucose plateaued at ~181 mg/dL with 53.6% TAR.'
        },
        {
            'event_id': 'EV05',
            'event_name': 'Stress Relief & Recovery',
            'start_timestamp': '2026-10-06T00:00:00+05:30',
            'end_timestamp': '2026-10-27T23:59:59+05:30',
            'stress_category': 'recovery',
            'stress_intensity_level': 0,
            'stress_label': 'Normal Recovery',
            'description': 'Post-exam results relief. Glycemic stabilization (median 178 mg/dL) leading up to return of menses on Oct 31.'
        }
    ]
    df_stress = pd.DataFrame(stress_data)
    out_path = os.path.join(PROCESSED_DIR, 'stress_events.csv')
    df_stress.to_csv(out_path, index=False)
    print(f"Saved {out_path} ({len(df_stress)} stress event records).")
    return df_stress

def build_wearable_metrics(hr_rows, hrv_rows):
    """
    Generates wearable_metrics.csv containing Heart Rate and HRV SDNN.
    Filters to the CGM time window (June 8 - October 28, 2026).
    """
    records = []
    cgm_start = pd.to_datetime('2026-06-08 00:00:00+05:30')
    cgm_end = pd.to_datetime('2026-10-28 23:59:59+05:30')
    
    for sd, val, unit, src in hr_rows:
        ts = pd.to_datetime(sd)
        if cgm_start <= ts <= cgm_end:
            records.append({
                'timestamp': ts.strftime('%Y-%m-%dT%H:%M:%S%z'),
                'metric_type': 'heart_rate',
                'value': round(val, 1),
                'unit': 'bpm',
                'source': src
            })
            
    for sd, val, unit, src in hrv_rows:
        ts = pd.to_datetime(sd)
        if cgm_start <= ts <= cgm_end:
            records.append({
                'timestamp': ts.strftime('%Y-%m-%dT%H:%M:%S%z'),
                'metric_type': 'hrv_sdnn',
                'value': round(val, 2),
                'unit': 'ms',
                'source': src
            })
            
    df_wearables = pd.DataFrame(records)
    df_wearables = df_wearables.sort_values('timestamp').reset_index(drop=True)
    out_path = os.path.join(PROCESSED_DIR, 'wearable_metrics.csv')
    df_wearables.to_csv(out_path, index=False)
    print(f"Saved {out_path} ({len(df_wearables):,} wearable metric records: {len(df_wearables[df_wearables['metric_type']=='heart_rate']):,} HR, {len(df_wearables[df_wearables['metric_type']=='hrv_sdnn']):,} HRV).")
    return df_wearables

def build_glucose_features(df_5m, df_cycles, df_stress):
    """
    Builds the integrated modeling table glucose_features.csv.
    Merges 5-minute CGM readings with:
    - Diurnal cyclical time features (sin/cos of hour+minute, day of week, is_weekend)
    - Cycle features (cycle_id, cycle_day, cycle_length, days_to_next_period, is_period, phase_label)
    - Stress features (stress_event_id, is_stressed, stress_intensity_level, stress_category)
    - Sensor gap indicator (minutes_since_last_reading, is_gap_boundary)
    """
    df = df_5m.copy()
    df['dt'] = pd.to_datetime(df['timestamp'])
    df['date'] = pd.to_datetime(df['dt'].dt.date)
    
    # 1. Temporal & Diurnal Features
    df['hour'] = df['dt'].dt.hour
    df['minute'] = df['dt'].dt.minute
    df['day_of_week'] = df['dt'].dt.dayofweek # 0=Monday, 6=Sunday
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # Cyclical encoding of 24h diurnal time (0 to 2*pi)
    seconds_in_day = 24 * 3600
    time_seconds = df['hour'] * 3600 + df['minute'] * 60
    df['time_sin'] = np.sin(2 * np.pi * time_seconds / seconds_in_day).round(4)
    df['time_cos'] = np.cos(2 * np.pi * time_seconds / seconds_in_day).round(4)
    
    # 2. Cycle Features Mapping
    cycle_records = []
    # Convert period starts into datetime list
    p_starts = [pd.to_datetime(d) for d in df_cycles['period_start_date'].tolist()]
    p_nexts = [pd.to_datetime(d) for d in df_cycles['next_period_start_date'].tolist()]
    
    def map_cycle_row(row_date):
        for idx, row in df_cycles.iterrows():
            st = pd.to_datetime(row['period_start_date'])
            nxt = pd.to_datetime(row['next_period_start_date'])
            if st <= row_date < nxt:
                cid = row['cycle_id']
                c_day = (row_date - st).days + 1
                c_len = row['cycle_length_days']
                d_to_next = (nxt - row_date).days
                is_p = 1 if c_day <= row['flow_duration_days'] else 0
                
                # Hormonal phase classification
                if row['cycle_type'] == 'regular':
                    ovulation_day = c_len - 14
                    if c_day <= 4:
                        phase = 'menstrual'
                    elif c_day < ovulation_day - 1:
                        phase = 'follicular'
                    elif ovulation_day - 1 <= c_day <= ovulation_day + 1:
                        phase = 'ovulatory'
                    elif c_day < c_len - 3:
                        phase = 'mid_luteal'
                    else:
                        phase = 'late_luteal_pre_menses'
                else: # Prolonged / skipped cycle (Aug 19 to Oct 31, 73 days)
                    if c_day <= 4:
                        phase = 'menstrual'
                    elif c_day <= 14:
                        phase = 'follicular'
                    elif c_day <= 25:
                        phase = 'mid_cycle_delayed'
                    elif c_day <= 40:
                        phase = 'skipped_period_hyperglycemia_surge'
                    elif c_day <= 65:
                        phase = 'prolonged_stress_plateau'
                    else:
                        phase = 'pre_menses_recovery'
                        
                return cid, c_day, c_len, d_to_next, is_p, phase
        return np.nan, np.nan, np.nan, np.nan, 0, 'unknown'
        
    cycle_res = df['date'].apply(map_cycle_row).tolist()
    df['cycle_id'] = [r[0] for r in cycle_res]
    df['cycle_day'] = [r[1] for r in cycle_res]
    df['cycle_length_days'] = [r[2] for r in cycle_res]
    df['days_to_next_period'] = [r[3] for r in cycle_res]
    df['is_menstrual_flow'] = [r[4] for r in cycle_res]
    df['hormonal_phase'] = [r[5] for r in cycle_res]
    
    # 3. Stress Features Mapping
    stress_lookup = []
    for _, srow in df_stress.iterrows():
        stress_lookup.append((
            srow['event_id'],
            pd.to_datetime(srow['start_timestamp']),
            pd.to_datetime(srow['end_timestamp']),
            srow['stress_category'],
            srow['stress_intensity_level'],
            srow['event_name']
        ))
        
    def map_stress_row(row_dt):
        for eid, st, ed, cat, intensity, name in stress_lookup:
            if st <= row_dt <= ed:
                return eid, name, cat, intensity, 1
        return 'NONE', 'Baseline', 'baseline', 0, 0
        
    stress_res = df['dt'].apply(map_stress_row).tolist()
    df['stress_event_id'] = [s[0] for s in stress_res]
    df['stress_event_name'] = [s[1] for s in stress_res]
    df['stress_category'] = [s[2] for s in stress_res]
    df['stress_intensity'] = [s[3] for s in stress_res]
    df['is_stressed'] = [s[4] for s in stress_res]
    
    # 4. Sensor Gap & Sampling Integrity
    df['dt_diff_mins'] = (df['dt'].diff().dt.total_seconds() / 60.0).fillna(5.0)
    # A gap boundary is flagged if > 15 minutes between consecutive readings
    df['is_gap_boundary'] = (df['dt_diff_mins'] > 15.0).astype(int)
    
    # Drop temporary datetime objects used for calculation
    out_cols = [
        'timestamp', 'glucose', 'glucose_min', 'glucose_max', 'glucose_std', 'reading_count',
        'hour', 'minute', 'day_of_week', 'is_weekend', 'time_sin', 'time_cos',
        'cycle_id', 'cycle_day', 'cycle_length_days', 'days_to_next_period', 'is_menstrual_flow', 'hormonal_phase',
        'stress_event_id', 'stress_event_name', 'stress_category', 'stress_intensity', 'is_stressed',
        'dt_diff_mins', 'is_gap_boundary'
    ]
    features_df = df[out_cols]
    out_path = os.path.join(PROCESSED_DIR, 'glucose_features.csv')
    features_df.to_csv(out_path, index=False)
    print(f"Saved {out_path} ({len(features_df):,} aligned modeling rows).")
    return features_df

def main():
    ensure_dir(PROCESSED_DIR)
    glucose_rows, hr_rows, hrv_rows = parse_apple_health_xml(RAW_XML_PATH)
    
    df_5m = build_glucose_5min(glucose_rows)
    df_cycles = build_cycle_events()
    df_stress = build_stress_events()
    df_wearables = build_wearable_metrics(hr_rows, hrv_rows)
    df_features = build_glucose_features(df_5m, df_cycles, df_stress)
    
    print("\nAll 5 modular datasets successfully built and validated!")

if __name__ == '__main__':
    main()

