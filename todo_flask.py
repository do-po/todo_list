
# 필요한 라이브러리 로드
import pandas as pd
from flask import Flask, render_template, request, redirect, session
from datetime import timedelta

# 함수가 저장된 모듈 로드
from refer.module import func
from refer.module.database import MyDB


# SQL 연결을 위한 MyDB 클래스 생성
db = MyDB()

# Flask로 서버 구축
app = Flask(__name__)

# secret_key 설정 (session data 암호화 키) **이것도 암호화 필요해 (아마 SHA로)
app.secret_key = 'ABC'

# 세션 데이터의 생명주기(지속시간)을 설정 **이거 시간이 아니라 종료시로 바꿔야 해 (세션 종료시 스케쥴 업데이트 위해서)
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(minutes= 15)

# 로그인 페이지
    # 로그인을 위해 post로 보내진 회원 가입 데이터 가져오기
@app.route('/sign_up', methods=['post'])
def sign_up():

    # post로 form에 보내진 로그인 데이터 가져오기
    user_info = request.form

    _id = user_info['user_id']
    _pass = user_info['password']

    # 로그로그
    print(f'유저가 보낸 id : {_id}, 유저가 보낸 비밀번호 : {_pass}')

    # DB 데이터와 비교하기 위해 user_id와 password 정보를 담은 query문 작성
    query = '''
        select
        `user_id`
        and
        `password`
        from
        `user_info`
        where
        `user_id` = %s and `password` = %s
        '''
    
    # DB로 query와 데이터 전송
    result = db.sql_query(query, _id, _pass)

                                        # ** 앞으로 여기 파트에 session, 로그인 실패시 정보를 보여주기 위해 데이터 담아주기 필요 <- 이게 무슨 말이었더라....??
    # 로그인 성공시 /home으로 이동
    if result:
        return redirect('/home')
    
    # 로그인 실패시 /sign_up으로 다시 이동
    else:
        return redirect('/sign_up')

# 회원가입 페이지
    # id 존재 유무를 판단
@app.route('/check_id', methods = ['post'])
def check_id():
    # front-end에서 비동기 통신(ajax)으로 보내는 id 값을 변수에 저장
    _id = request.form['user_id']
    
    # 유저에게 받은 데이터 확인 (로그)

    print(f"/check_id[post] -> 유저 id : {_id}")

    # 유저가 보낸 id 값이 사용 가능한가?
        # 조건문 == 해당하는 id로 table에 데이터가 존재하는가 ?

    query = """
        SELECT
        `user_id`
        from
        `user_info`
        where
        id = %s
    """

    result = db(query, _id)

    # id가 사용 가능한 경우 == idx 값이 없을 때
        # id가 이미 있는 경우 idx 값으로 0을 받는다
    if result:
        idx = "0"
        # id가 없는 경우 idx 값으로 1을 받는다.
    else:
        idx = "1"

    return idx

    # id가 없을 경우 회원 정보에 추가
@app.route('/sign_in', method=['post'])
def sign_in():

    # post로 받아온 user_info:dict를 `user_info` 테이블에 저장
    user_info = request.form
    
    try:
        result = func.save_data(user_info, 'user_info')

    except:
        idx = 0
        return redirect('/sign_up')
    
    # 저장시 오류가 발생한다면 회원가입 화면으로 되돌아간다
        # 오류 발생 문구를 보여주기 위해 idx 값을 받는다
    if idx == "0":
        return redirect(f'/sign_up?state={idx}')
    # 오류가 나지 않는다면 로그인 화면으로 돌아간다.
    else:
        return redirect('/sign_in')

# 로그아웃
@app.route('/log_out')
def log_out():
    
    # 세션 데이터를 제거

    # session.pop('user_id', None)
    # session.pop('user_pass', None)
    session.clear()

    # **여기에 현재 스케쥴 가중치를 `schedule_2`에 업데이트 하는 코드가 들어가야 할듯??

    # 로그인 페이지로 이동

    return redirect('/sign_up')


# 서버 구동 (현재 디버깅 모드)
app.run(debug=True)