


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, MultiHeadAttention, LayerNormalization, Concatenate, Multiply
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import AdamW
import matplotlib.pyplot as plt



# --- DATASET PY ----
def load_and_split_data():
    print("\n[INFO] Loading datasets...")
    wide_df = pd.read_csv("/projects/dsci410_510/Aurora/widegpsptsd.csv", low_memory=False)
    long_df = pd.read_csv("/projects/dsci410_510/Aurora/longgpsliwc.csv", low_memory=False)
    
    numeric_cols = wide_df.columns[wide_df.columns.str.contains('PCL5|ChanceofDying|Age|PSS10|PhenX|AnxBank|Dep8b|Pain4a|Somatic|CDRISC10|PCS_Rum|PSQI|SleepImp|ISI|BMI|Stress', regex=True)]
    for col in numeric_cols:
        if col in wide_df.columns:
            wide_df[col] = pd.to_numeric(wide_df[col], errors='coerce')
    
    long_df["Time_Label"] = long_df["day"].apply(lambda x: "PRE" if x <= 0 else "WK2" if x <= 14 else "WK8" if x <= 56 else "M3" if x <= 90 else "M6" if x <= 180 else None)
    long_df = long_df[long_df["Time_Label"].notna()]
    
    train_pids, temp_pids = train_test_split(wide_df["PID"].unique(), test_size=0.30, random_state=42069)
    val_pids, test_pids = train_test_split(temp_pids, test_size=0.50, random_state=42069)
    
    train_wide = wide_df[wide_df["PID"].isin(train_pids)]
    val_wide = wide_df[wide_df["PID"].isin(val_pids)]
    test_wide = wide_df[wide_df["PID"].isin(test_pids)]
    
    train_long = long_df[long_df["PID"].isin(train_pids)]
    val_long = long_df[long_df["PID"].isin(val_pids)]
    test_long = long_df[long_df["PID"].isin(test_pids)]
    
    print(f"Train Wide: {train_wide.shape}, Train Long: {train_long.shape}")
    print(f"Val Wide: {val_wide.shape}, Val Long: {val_long.shape}")
    print(f"Test Wide: {test_wide.shape}, Test Long: {test_long.shape}")
    
    return train_long, train_wide, val_long, val_wide, test_long, test_wide

# Preprocess Data
def preprocess_data(long_df, wide_df):
    ts_features = ['dist_traveled', 'radius', 'num_sig_places', 'av_flight_length', 'LIWC_posemo', 'LIWC_anger', 'LIWC_sad']
    static_features = [
        'ED_ChanceofDying', 'WK2_PROM_AnxBank_RS', 'WK2_SCRN_GAD_RS', 'WK2_BriefPCL', 'WK2_ASD_RS',
        'ED_ChiefPain_MuscSkel', 'ED_ChiefPain_Neck', 'ED_ChiefPain_Headache', 'PRE_PROM_Pain4a_T',
        'WK2_PROM_Pain4a_T', 'PRE_Somatic_YN_COUNT', 'WK2_Somatic_YN_COUNT', 'PRE_PhenX_Tob30d_Freq',
        'WK2_PhenX_Alc30d_Freq', 'PRE_PSS10_Total_RS', 'WK2_CDRISC10_RS', 'ED_RaceEthCode',
        'ED_highestgrade', 'WK2_IncomeCode', 'PRE_PROM_Dep8b_T', 'WK2_TIPI_EmoStability_RS',
        'WK2_CTQSF_EmoAbu_RS', 'PRE_PCL5_RS', 'WK2_PCL5_RS', 'PRE_PCS_Rum', 'WK2_PCS_Rum', 'WK2_PSQI_SlpDur',
        'PRE_PhenX_Alc30d_Quan', 'WK2_PhenX_Alc30d_Quan', 'PRE_PROM_SleepImp8a_RS', 'WK2_ISI_RS',
        'ED_GenderBirthCert', 'ED_GenderNow', 'ED_Marital', 'WK2_EmploymentCode', 
        'ED_VehicleCat', 'ED_VehicleDamage_cat', 'CensusDivision', 'CensusRegion',
        'ED_Event_BroadClass', 'ED_Event_SpecificClass', 'Wk2_Polytrauma',
    ]
    continuous_to_bin = [
        'BMI', 'PRE_Stress_Max', 'WK2_Stress_MAX', 'ED_Age', 
        'PRE_PSS10_Distress_RS', 'PRE_PSS10_CantCope_RS', 'WK8_PSS10_Distress_RS', 'ED_NowSomatic_YN_COUNT_x'
    ]
    targets = ['WK8_PCL5_RS', 'M3_PCL5_RS', 'M6_PCL5_RS']

    print("Raw PCL-5 sample (first 5 rows):")
    print(wide_df[targets + ['PRE_PCL5_RS', 'WK2_PCL5_RS']].head())
    print("Max raw PCL-5 values:", wide_df[targets + ['PRE_PCL5_RS', 'WK2_PCL5_RS']].max())
    print("Min raw PCL-5 values:", wide_df[targets + ['PRE_PCL5_RS', 'WK2_PCL5_RS']].min())

    le = LabelEncoder()
    categorical_cols = [
        'ED_RaceEthCode', 'ED_highestgrade', 'WK2_IncomeCode', 'ED_Event_BroadClass', 'ED_Event_SpecificClass', 
        'Wk2_Polytrauma', 'ED_GenderBirthCert', 'ED_GenderNow', 'ED_Marital', 'WK2_EmploymentCode', 
        'ED_VehicleCat', 'ED_VehicleDamage_cat', 'CensusDivision', 'CensusRegion'
    ]
    for col in categorical_cols:
        if col in wide_df.columns:
            wide_df[col] = le.fit_transform(wide_df[col].astype(str).fillna('Unknown'))

    for col in continuous_to_bin:
        if col in wide_df.columns:
            wide_df[f'{col}_cat'] = pd.qcut(wide_df[col], q=4, labels=False, duplicates='drop').fillna(-1).astype(int)
            static_features.append(f'{col}_cat')

    long_pre_wk2 = long_df[long_df["day"] <= 14]
    agg_dict = {feat: ['mean', 'std'] for feat in ts_features}
    ts_agg = long_pre_wk2.groupby("PID").agg(agg_dict).reset_index()
    ts_agg.columns = ['PID'] + [f"{feat}_{agg}" for feat, agg in ts_agg.columns[1:]]
    wide_df = wide_df.merge(ts_agg, on="PID", how="left")

    static_features.extend([col for col in ts_agg.columns if col != 'PID'])
    static_features.append('PCL5_PRE_WK2_diff')
    wide_df['PCL5_PRE_WK2_diff'] = wide_df['WK2_PCL5_RS'] - wide_df['PRE_PCL5_RS']

    static_df = wide_df[static_features].copy()
    for col in static_features:
        static_df[col] = pd.to_numeric(static_df[col], errors='coerce')

    imputer = KNNImputer(n_neighbors=5)
    static_df = pd.DataFrame(imputer.fit_transform(static_df), columns=static_features, index=wide_df.index)
    scaler_static = StandardScaler()
    X_static = scaler_static.fit_transform(static_df)

    scaler_ts = StandardScaler()
    ts_numeric = long_df[ts_features].select_dtypes(include=[np.number])
    long_df[ts_numeric.columns] = imputer.fit_transform(ts_numeric)
    long_scaled = scaler_ts.fit_transform(long_df[ts_numeric.columns])

    y_raw = wide_df[targets].values
    print(f"y mean (pre-impute): {np.nanmean(y_raw, axis=0)}, SD: {np.nanstd(y_raw, axis=0)}")
    y = imputer.fit_transform(y_raw)
    print(f"y mean (post-impute): {y.mean(axis=0)}, SD: {y.std(axis=0)}")

    max_days = 181
    X_ts = np.zeros((len(wide_df), max_days, len(ts_features)))
    mask = np.zeros((len(wide_df), max_days))

    pid_to_idx = {pid: idx for idx, pid in enumerate(wide_df["PID"])}
    merged_df = long_df[["PID", "day"]].join(pd.DataFrame(long_scaled, columns=ts_features, index=long_df.index))
    for pid, group in merged_df.groupby("PID"):
        idx = pid_to_idx[pid]
        for _, row in group.iterrows():
            day = int(row["day"])
            if day < max_days:
                X_ts[idx, day] = row[ts_features].values
                mask[idx, day] = 1

    return X_ts, X_static, y, mask, ts_features, static_features, targets

