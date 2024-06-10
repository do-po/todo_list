import pymysql

class MyDB:
    # 생성자 함수 생성 (DB 서버의 정보를 입력)
    def __init__(
            self,
            _host = 'localhost',
            _port = 3306,
            _user = 'root',
            _pw = '0000',
            _db = 'todo'
    ):
        self.host = _host
        self.port = _port
        self.user = _user
        self.pw = _pw
        self.db = _db

    def sql_query(self, _sql, *_values):
        ## DB 서버와의 연결
        mydb = pymysql.connect(
            host = self.host,
            port = self.port,
            user = self.user,
            password = self.pw,
            db = self.db
        )
        # cursor 생성
        cursor = mydb.cursor(pymysql.cursors.DictCursor)
        # _sql, _values를 이용하여 cursor에게 질의를 보낸다.
        cursor.execute(_sql, _values)
        # _sql이 select문인지 확인
        if _sql.strip().lower().startswith('select'):
            result = cursor.fetchall()
        else:
            mydb.commit()
            result = 'Query Done'
        # 데이터베이스 서버와의 연결 종료
        mydb.close()
        return result
            