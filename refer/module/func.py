# 필요 라이브러리 로드
from database import MyDB
from dotenv import load_dotenv
import os

# MyDB 클래스 생성

db = MyDB()

# DB에 저장하는 기능을 모아둔 섹션
    
    # 테이블에 데이터를 저장
def save_data(_data:dict, _table):

    # sql 쿼리 작성
    query = f'''
    INSERT
    INTO
    `{_table}`
    ({','.join(tuple(_data.keys()))})
    VALUES
    ({','.join(tuple(_data.values()))})
    '''
    print(query)
    # MyDB 모듈로 query 실행
    result = db.sql_query(query)
    
    return result

# DB에서 데이터를 받아오는 기능들을 모아둔 섹션

    # 유저의 속성에 맞는 가중치를 SQL에서 받아온다
def get_data(_user:dict, _table = 'base_weight'):

    # 빈 리스트를 생성하여 조건문을 저장
    condition_list = []

    # _user의 각 key에 대해 반복
    for key in _user.keys():
        
        # 각 key에 대해 'key = %s' 형식의 문자열을 생성하여 리스트에 추가
        condition = f'{key} = %s'
        condition_list.append(condition)

    # 리스트의 모든 문자열을 " AND "로 연결하여 query 생성
    conditions = " AND ".join(condition_list)

    # sql 쿼리 실행
    query = f'''
    SELECT
    {','.join(_user.keys())}
    FROM
    `{_table}`
    WHERE
    {conditions}
    '''

    # MyDB 모듈로 query 실행
    result = db.sql_query(query, *list(_user.values()))

    # 가중치를 저장할 dict와 list 생성
    data = {}
    weights = []

    # query 결과를 가중치 list에 저장
    weights.append(result)

        # query 결과를 처리
    if result:  # 결과가 존재하면
        # 첫 번째 행의 결과만 사용, _user의 각 키에 대응하는 값을 저장
        first_result = result[0]
        for key in _user.keys():
            data[key] = first_result[key]
    else:
        # 결과가 없는 경우, 기본값 또는 오류 처리를 할 수 있음
        print("No data found matching the criteria.")

    # 만들어진 가중치 dict를 반환
    return data