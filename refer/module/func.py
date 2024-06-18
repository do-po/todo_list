import pandas as pd
from refer.module.database import MyDB
from dotenv import load_dotenv
import os

# MyDB 클래스 생성
db = MyDB()

# 테이블에 데이터를 저장하는 함수
def save_data(data, table):
    """
    다양한 형식의 데이터를 받아서 테이블에 저장하는 함수
    data: dict, DataFrame 또는 list 중 하나
    table: 데이터를 저장할 테이블 이름
    """
    if isinstance(data, dict):
        save_dict_data(data, table)  # dict 형식의 데이터를 처리하는 함수 호출
    elif isinstance(data, pd.DataFrame):
        save_dataframe_data(data, table)  # DataFrame 형식의 데이터를 처리하는 함수 호출
    elif isinstance(data, list):
        save_list_data(data, table)  # list 형식의 데이터를 처리하는 함수 호출
    else:
        raise ValueError("Unsupported data type. Please provide a dict, DataFrame, or list.")  # 지원하지 않는 데이터 타입 처리

def save_dict_data(data, table):
    """
    dict 형식의 데이터를 테이블에 저장하는 함수
    data: dict 형태의 데이터
    table: 데이터를 저장할 테이블 이름
    """
    # 특정 테이블(user_weight)에서만 user_id 제거
    if (table == 'user_weight') & ('user_id' in data):
        
        del data['user_id']
    query = f'''
    INSERT INTO `{table}`
    ({','.join(tuple(data.keys()))})
    VALUES
    ({','.join(['%s'] * len(data))})
    '''
    print("sql query : ", query)  # 생성된 쿼리 출력 (디버깅 용도)
    print("sql insert data : ", data)   # 데이터 출력 (디버깅 용도)
    result = db.sql_query(query, *list(data.values()))  # MyDB 모듈을 통해 쿼리 실행
    return result

def save_list_data(data, table):
    """
    list 형식의 데이터를 테이블에 저장하는 함수
    data: list 형태의 데이터 (각 요소가 dict 형태)
    table: 데이터를 저장할 테이블 이름
    """
    for index, record in enumerate(data):
        # 특정 테이블(user_weight)에서만 user_id 제거
        if (table == 'user_weight') and ('user_id' in record):
            del record['user_id']
        try:
            print("save_list_data record : ",type(record))
            save_dict_data(record, table)  # 각 dict 데이터를 저장하는 함수 호출
        except Exception as e:
            print(f"Error inserting record {index}: {e}")

def save_dataframe_data(dataframe, table):
    """
    DataFrame 형식의 데이터를 테이블에 저장하는 함수
    dataframe: DataFrame 형태의 데이터
    table: 데이터를 저장할 테이블 이름
    """
    for index, row in dataframe.iterrows():  # DataFrame의 각 행을 반복
        data_dict = row.to_dict()  # 각 행을 딕셔너리로 변환
        # 특정 테이블(user_weight)에서만 user_id 제거
        if table == 'user_weight' and 'user_id' in data_dict:
            del data_dict['user_id']
        try:
            save_dict_data(data_dict, table)  # 변환된 딕셔너리 데이터를 저장하는 함수 호출
        except Exception as e:
            print(f"Error inserting row {index}: {e}")

# 유저의 속성에 맞는 가중치를 SQL에서 받아오는 함수
def get_data(user, table='base_weight'):
    """
    다양한 형식의 유저 데이터를 받아서 해당 유저의 속성에 맞는 가중치를 SQL에서 받아오는 함수
    user: dict 또는 DataFrame 중 하나
    table: 데이터를 조회할 테이블 이름 (기본값: 'base_weight')
    """
    if isinstance(user, dict):
        return get_dict_data(user, table)  # dict 형식의 유저 데이터를 처리하는 함수 호출
    elif isinstance(user, pd.DataFrame):
        return get_dataframe_data(user, table)  # DataFrame 형식의 유저 데이터를 처리하는 함수 호출
    else:
        raise ValueError("Unsupported user data type. Please provide a dict or a DataFrame.")  # 지원하지 않는 데이터 타입 처리

def get_dict_data(user, table):
    """
    dict 형식의 유저 데이터를 받아서 해당 유저의 속성에 맞는 가중치를 SQL에서 받아오는 함수
    user: dict 형태의 유저 데이터
    table: 데이터를 조회할 테이블 이름
    """
    condition_list = [f'{key} = %s' for key in user.keys()]  # 조건문 리스트 생성
    conditions = " AND ".join(condition_list)  # 조건문 리스트를 " AND "로 연결하여 쿼리 생성
    query = f'''
    SELECT * FROM `{table}`
    WHERE {conditions}
    '''
    result = db.sql_query(query, *list(user.values()))  # MyDB 모듈을 통해 쿼리 실행
    return format_result(result)  # 결과 형식화 함수 호출

def get_dataframe_data(dataframe, table):
    """
    DataFrame 형식의 유저 데이터를 받아서 해당 유저의 속성에 맞는 가중치를 SQL에서 받아오는 함수
    dataframe: DataFrame 형태의 유저 데이터
    table: 데이터를 조회할 테이블 이름
    """
    results = []
    for index, row in dataframe.iterrows():  # DataFrame의 각 행을 반복
        user_dict = row.to_dict()  # 각 행을 딕셔너리로 변환
        results.append(get_dict_data(user_dict, table))  # 변환된 딕셔너리 데이터를 처리하여 결과 리스트에 추가
    return results  # 결과 리스트 반환

def format_result(result):
    """
    쿼리 결과를 형식화하는 함수
    result: 쿼리 결과
    """
    if not result:
        print("No data found matching the criteria.")  # 결과가 없으면 메시지 출력
        return {}
    data = {}
    first_result = result[0]
    for key in first_result.keys():
        data[key] = first_result[key]
    return data  # 형식
