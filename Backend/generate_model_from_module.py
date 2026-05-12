import os
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    data_path = 'dataset/Training (1).csv'
    if not os.path.exists(data_path):
        raise FileNotFoundError(f'Training dataset not found: {data_path}')

    df = pd.read_csv(data_path)
    if 'prognosis' in df.columns:
        label_col = 'prognosis'
    else:
        label_col = df.columns[-1]

    X = df.drop(columns=[label_col])
    y = df[label_col].astype(str)

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    score = model.score(X_test, y_test)
    print(f'Trained RandomForestClassifier on {len(X)} samples; test accuracy: {score:.4f}')

    os.makedirs('model', exist_ok=True)
    with open('model/svc.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('model/label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)
    print('Saved trained model to model/svc.pkl and label encoder to model/label_encoder.pkl')
