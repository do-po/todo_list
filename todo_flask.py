
# 필요한 라이브러리 로드
import pandas as pd
from flask import Flask, render_template, request, redirect

# 함수가 저장된 모듈 로드
from refer.module import func
from refer.module.database import MyDB

# SQL 연결을 위한 MyDB 클래스 생성
db = MyDB()

# Flask로 서버 구축
app = Flask(__name__)


# 로그인을 위해 post로 보내진 회원 가입 데이터 가져오기

@app.route('/sign_up', methods=['post'])

# 로그인 페이지
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

                                        # 앞으로 여기 파트에 session, 로그인 실패시 정보를 보여주기 위해 데이터 담아주기 필요
    # 로그인 성공시 /home으로 이동
    if result:
        return redirect('/home')
    
    # 로그인 실패시 /sign_up으로 다시 이동
    else:
        return redirect('/sign_up')
    
@app.route()

# 서버 구동 (현재 디버깅 모드)
app.run(debug=True)