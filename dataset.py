# dataset.py

pip install tensorflow pandas numpy scikit-learn matplotlib

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split

class AuroraDataset:
    def __init__(self, wide_path="/projects/dsci410_510/Aurora/widegpsptsd.csv", 
                 long_path="/projects/dsci410_510/Aurora/longgpsliwc.csv", 
                 test_size=0.3, val_size=0.5, random_state=42069):
        self.wide_path = wide_path
        self.long_path = long_path
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        self.ts_features = ['dist_traveled', 'radius', 'num_sig_places', 'av_flight_length', 
                           'LIWC_posemo', 'LIWC_anger', 'LIWC_sad']
        self.static_features = [
            'ED_ChanceofDying', 'WK2_PROM_AnxBank_RS', 'WK2_SCRN_GAD_RS', 'WK2_BriefPCL', 
            'WK2_ASD_RS', 'ED_ChiefPain_MuscSkel', 'ED_ChiefPain_Neck', 'ED_ChiefPain_Headache',
            'PRE_PROM_Pain4a_T', 'WK2_PROM_Pain4a_T', 'PRE_Somatic_YN_COUNT', 
            'WK2_Somatic_YN_COUNT', 'PRE_PhenX_Tob30d_Freq', 'WK2_PhenX_Alc30d_Freq', 
            'PRE_PSS10_Total_RS', 'WK2_CDRISC10_RS', 'ED_RaceEthCode', 'ED_highestgrade', 
            'WK2_IncomeCode', 'PRE_PROM_Dep8b_T', 'WK2_TIPI_EmoStability_RS', 
            'WK2_CTQSF_EmoAbu_RS', 'PRE_PCL5_RS', 'WK2_PCL5_RS', 'PRE_PCS_Rum', 
            'WK2_PCS_Rum', 'WK2_PSQI_SlpDur', 'PRE_PhenX_Alc30d_Quan', 
            'WK2_PhenX_Alc30d_Quan', 'PRE_PROM_SleepImp8a_RS', 'WK2_ISI_RS',
            'ED_GenderBirthCert', 'ED_GenderNow', 'ED_Marital', 'WK2_EmploymentCode', 
            'ED_VehicleCat', 'ED_VehicleDamage_cat', 'CensusDivision', 'CensusRegion',
            'ED_Event_BroadClass', 'ED_Event_SpecificClass', 'Wk2_Polytrauma'
        ]
        self.continuous_to_bin = ['BMI', 'PRE_Stress_Max', 'WK2_Stress_MAX', 'ED_Age', 
                                 'PRE_PSS10_Distress_RS', 'PRE_PSS10_CantCope_RS', 
                                 'WK8_PSS10_Distress_RS', 'ED_NowSomatic_YN_COUNT_x']
        self.targets = ['WK8_PCL5_RS', 'M3_PCL5_RS', 'M6_PCL5_RS']
        self.load_and_preprocess()

    def load_and_preprocess(self):
        wide_df = pd.read_csv(self.wide_path, low_memory=False)
        long_df = pd.read_csv(self.long_path, low_memory=False)

        # Numeric conversion
        numeric_cols = wide_df.columns[wide_df.columns.str.contains(
            'PCL5|ChanceofDying|Age|PSS10|PhenX|AnxBank|Dep8b|Pain4a|Somatic|CDRISC10|PCS_Rum|PSQI|SleepImp|ISI|BMI|Stress', 
            regex=True)]
        for col in numeric_cols:
            if col in wide_df.columns:
                wide_df[col] = pd.to_numeric(wide_df[col], errors='coerce')

        # Time labels
        long_df["Time_Label"] = long_df["day"].apply(
            lambda x: "PRE" if x <= 0 else "WK2" if x <= 14 else "WK8" if x <= 56 else 
            "M3" if x <= 90 else "M6" if x <= 180 else None)
        long_df = long_df[long_df["Time_Label"].notna()]

        # Split data
        train_pids, temp_pids = train_test_split(wide_df["PID"].unique(), 
                                               test_size=self.test_size, 
                                               random_state=self.random_state)
        val_pids, test_pids = train_test_split(temp_pids, 
                                             test_size=self.val_size, 
                                             random_state=self.random_state)
        self.train_wide = wide_df[wide_df["PID"].isin(train_pids)]
        self.val_wide = wide_df[wide_df["PID"].isin(val_pids)]
        self.test_wide = wide_df[wide_df["PID"].isin(test_pids)]
        self.train_long = long_df[long_df["PID"].isin(train_pids)]
        self.val_long = long_df[long_df["PID"].isin(val_pids)]
        self.test_long = long_df[long_df["PID"].isin(test_pids)]

        # Preprocess
        self.X_ts_train, self.X_static_train, self.y_train = self._preprocess(self.train_long, self.train_wide)
        self.X_ts_val, self.X_static_val, self.y_val = self._preprocess(self.val_long, self.val_wide)
        self.X_ts_test, self.X_static_test, self.y_test = self._preprocess(self.test_long, self.test_wide)

    def _preprocess(self, long_df, wide_df):
        le = LabelEncoder()
        categorical_cols = [
            'ED_RaceEthCode', 'ED_highestgrade', 'WK2_IncomeCode', 'ED_Event_BroadClass', 
            'ED_Event_SpecificClass', 'Wk2_Polytrauma', 'ED_GenderBirthCert', 'ED_GenderNow', 
            'ED_Marital', 'WK2_EmploymentCode', 'ED_VehicleCat', 'ED_VehicleDamage_cat', 
            'CensusDivision', 'CensusRegion'
        ]
        for col in categorical_cols:
            if col in wide_df.columns:
                wide_df[col] = le.fit_transform(wide_df[col].astype(str).fillna('Unknown'))

        # Bin continuous features
        for col in self.continuous_to_bin:
            if col in wide_df.columns:
                wide_df[f'{col}_cat'] = pd.qcut(wide_df[col], q=4, labels=False, duplicates='drop').fillna(-1).astype(int)
                if f'{col}_cat' not in self.static_features:
                    self.static_features.append(f'{col}_cat')

        # Time-series aggregation
        long_pre_wk2 = long_df[long_df["day"] <= 14]
        agg_dict = {feat: ['mean', 'std'] for feat in self.ts_features}
        ts_agg = long_pre_wk2.groupby("PID").agg(agg_dict).reset_index()
        ts_agg.columns = ['PID'] + [f"{feat}_{agg}" for feat, agg in ts_agg.columns[1:]]
        wide_df = wide_df.merge(ts_agg, on="PID", how="left")

        # Extend static features
        static_features_extended = self.static_features + [col for col in ts_agg.columns if col != 'PID']
        static_features_extended.append('PCL5_PRE_WK2_diff')
        wide_df['PCL5_PRE_WK2_diff'] = wide_df['WK2_PCL5_RS'] - wide_df['PRE_PCL5_RS']

        # Static processing
        static_df = wide_df[static_features_extended].copy()
        for col in static_features_extended:
            static_df[col] = pd.to_numeric(static_df[col], errors='coerce')
        imputer = KNNImputer(n_neighbors=5)
        static_df = pd.DataFrame(imputer.fit_transform(static_df), 
                                columns=static_features_extended, 
                                index=wide_df.index)
        scaler_static = StandardScaler()
        X_static = scaler_static.fit_transform(static_df)

        # Time-series processing
        scaler_ts = StandardScaler()
        ts_numeric = long_df[self.ts_features].select_dtypes(include=[np.number])
        long_df[ts_numeric.columns] = imputer.fit_transform(ts_numeric)
        long_scaled = scaler_ts.fit_transform(long_df[ts_numeric.columns])

        # 3D time-series array
        max_days = 181
        X_ts = np.zeros((len(wide_df), max_days, len(self.ts_features)))
        pid_to_idx = {pid: idx for idx, pid in enumerate(wide_df["PID"])}
        merged_df = long_df[["PID", "day"]].join(pd.DataFrame(long_scaled, 
                                                              columns=self.ts_features, 
                                                              index=long_df.index))
        for pid, group in merged_df.groupby("PID"):
            idx = pid_to_idx[pid]
            for _, row in group.iterrows():
                day = int(row["day"])
                if day < max_days:
                    X_ts[idx, day] = row[self.ts_features].values

        # Targets
        y_raw = wide_df[self.targets].values
        y = imputer.fit_transform(y_raw)

        return X_ts, X_static, y

    def get_train_data(self):
        return self.X_ts_train, self.X_static_train, self.y_train

    def get_val_data(self):
        return self.X_ts_val, self.X_static_val, self.y_val

    def get_test_data(self):
        return self.X_ts_test, self.X_static_test, self.y_test

if __name__ == "__main__":
    dataset = AuroraDataset()
    X_ts_train, X_static_train, y_train = dataset.get_train_data()
    print(f"Train shapes: X_ts={X_ts_train.shape}, X_static={X_static_train.shape}, y={y_train.shape}")