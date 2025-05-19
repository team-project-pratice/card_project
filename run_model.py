import joblib
from preprocessing import preprocess_feature
import pandas as pd

rf = joblib.load("./model/95randomforest.pkl")

def predict_from_input(input_dict):
    """
    input_dict: 예측에 사용할 하나의 샘플 데이터 (딕셔너리)
    return: 예측 결과와 클래스 1 확률
    """
    # 입력을 DataFrame으로 변환
    df = pd.DataFrame([input_dict])

    # 전처리 수행
    df_processed = preprocess_feature(df)

    # 예측
    pred = rf.predict(df_processed)[0]               # 첫 샘플 예측 결과
    proba = rf.predict_proba(df_processed)[0][1]     # 첫 샘플 클래스 1 확률

    return pred, proba
