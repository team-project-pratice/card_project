import joblib
from preprocessing import preprocess_feature
import pandas as pd
import os
import pymysql
 
base_dir = os.path.dirname(__file__)  # == src/junmodel
model_path = os.path.join(base_dir, 'model', 'k-fold_42.pkl')
 
conn = pymysql.connect(
    host='localhost',
    user='wms',
    password='1234',
    db='card',
    charset='utf8mb4',
    cursorclass=pymysql.cursors.DictCursor
)
 
rf = joblib.load(model_path)

def predict_from_input(input_list):
    """
    input_dict: 예측에 사용할 하나의 샘플 데이터 (딕셔너리)
    return: 예측 결과와 클래스 1 확률
    """
    # 입력을 DataFrame으로 변환
    df = pd.DataFrame(input_list)
 
    # 전처리 수행
    df_processed = preprocess_feature(df)
    
    # 여러 샘플에 대한 클래스 1 확률
    proba = rf.predict_proba(df_processed)[:, 1]

    # 임계값 기준으로 1 또는 0 예측
    pred = rf.predict(df_processed)

    
    print("예측 결과 (pred):", pred)
    print("pred의 자료형:", type(pred))
 
    print("클래스 1일 확률 (proba):", proba)
    print("proba의 자료형:", type(proba))
 
 
    return pred, proba
 
with conn.cursor() as cursor:
    sql = "SELECT * FROM card_leaver"  # 여기서 your_table_name을 실제 테이블 이름으로 바꿔줘
    cursor.execute(sql)
    row = cursor.fetchall()  # dict로 반환됨
 
    if row:
        predict_from_input(row)
    else:
        print("데이터가 없습니다.")