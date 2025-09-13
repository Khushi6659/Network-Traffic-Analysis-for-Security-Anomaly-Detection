# ml_model_enhanced_full.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

def train_ml_model(input_file="features_data.csv", model_file="network_intrusion_model_enhanced.pkl"):
    print("🚀 Starting ML Model Training...\n")

    # 1️ Load dataset
    df = pd.read_csv(input_file)
    print(f"✅ Dataset loaded with shape: {df.shape}\n")

    # 2️ Create binary attack column from 'label'
    df['attack'] = df['label'].apply(lambda x: 0 if x == 0 else 1)
    print("📊 Class distribution (before sampling):")
    print(df['attack'].value_counts(), "\n")

    # 3️ Sample dataset safely (approximate stratified)
    sample_size = 1000000
    fractions = sample_size / len(df)
    df_sample = pd.concat([x.sample(frac=fractions, random_state=42) for _, x in df.groupby('attack')])
    print(f"✅ Using {len(df_sample)} rows for training\n")

    # 4️ Drop irrelevant columns
    drop_cols = ['uid', 'ts', 'id.orig_h', 'id.resp_h', 'attack_type', 'label']
    df_sample = df_sample.drop([c for c in drop_cols if c in df_sample.columns], axis=1)

    # 5️ Split features and target
    X = df_sample.drop('attack', axis=1)
    y = df_sample['attack']

    # 6️ Encode categorical columns
    for col in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])

    # 7️ Add extra derived features
    X['pkt_ratio'] = X['fwd_pkts_tot'] / (X['bwd_pkts_tot'] + 1)
    X['data_ratio'] = X['fwd_data_pkts_tot'] / (X['bwd_data_pkts_tot'] + 1)
    X['header_ratio'] = X['fwd_header_size_tot'] / (X['bwd_header_size_tot'] + 1)
    X['flow_per_sec_ratio'] = X['fwd_pkts_per_sec'] / (X['bwd_pkts_per_sec'] + 1)
    X['down_up_ratio_alt'] = X['down_up_ratio']
    X['total_flags'] = (X['flow_fin_flag_count'] + X['flow_syn_flag_count'] +
                        X['flow_rst_flag_count'] + X['fwd_psh_flag_count'] +
                        X['bwd_psh_flag_count'] + X['flow_ack_flag_count'] +
                        X['fwd_urg_flag_count'] + X['bwd_urg_flag_count'] +
                        X['flow_cwr_flag_count'] + X['flow_ece_flag_count'])
    X['payload_per_pkt'] = X['payload_bytes_per_second'] / (X['fwd_pkts_tot'] + X['bwd_pkts_tot'] + 1)
    X['window_size_ratio'] = (X['fwd_init_window_size'] + 1) / (X['bwd_init_window_size'] + 1)

    # 8️ Scale numeric features
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
    scaler = StandardScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    # 9️ Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    print(f"✅ Training data: {X_train.shape}, Testing data: {X_test.shape}\n")

    # 🔟 Train Random Forest with class weight balancing
    print("🤖 Training Random Forest Classifier...")
    model = RandomForestClassifier(
        n_estimators=100, 
        random_state=42, 
        n_jobs=-1,
        class_weight="balanced_subsample"
    )
    model.fit(X_train, y_train)
    print("✅ Model training completed.\n")

    # 1️⃣1️⃣ Predict with threshold tuning
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    thresholds = [0.5, 0.4, 0.3]
    for t in thresholds:
        print(f"\n📈 Model Performance (Threshold = {t}):")
        y_pred = (y_pred_proba >= t).astype(int)
        acc = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {acc:.4f}")
        print(classification_report(y_test, y_pred, target_names=["Normal", "Attack"]))

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        print("Confusion Matrix:\n", cm)

        plt.figure(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Normal","Attack"], yticklabels=["Normal","Attack"])
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title(f"Confusion Matrix (Threshold={t})")
        plt.show()

    # 1️⃣2️⃣ Save model
    joblib.dump(model, model_file)
    print(f"\n💾 Model saved as '{model_file}'")

if __name__ == "__main__":
    train_ml_model()
