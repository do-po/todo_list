-- 데이터베이스 선택
USE todo;

-- user_weight 테이블에 데이터 삽입 및 외래 키 설정
CREATE TABLE IF NOT EXISTS user_weight (
    user_weight_no INT PRIMARY KEY AUTO_INCREMENT,
    user_id VARCHAR(256),
    work DOUBLE,
    edu DOUBLE,
    free_time DOUBLE,
    health DOUBLE,
    chores DOUBLE,
    category_else DOUBLE,
    FOREIGN KEY (user_id) REFERENCES user_info(user_id)
);

-- 기존 프로시저 삭제
DROP PROCEDURE IF EXISTS drop_fk_if_exists;

-- 프로시저를 생성하여 외래 키 제약 조건을 확인하고 삭제
DELIMITER //

CREATE PROCEDURE drop_fk_if_exists()
BEGIN
    DECLARE fk_name VARCHAR(255);

    -- history 테이블 외래 키 확인 및 삭제
    SELECT CONSTRAINT_NAME INTO fk_name 
    FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE 
    WHERE TABLE_NAME = 'history' 
    AND COLUMN_NAME = 'user_weight_no' 
    AND CONSTRAINT_NAME != 'PRIMARY' 
    LIMIT 1;

    IF fk_name IS NOT NULL THEN
        SET @sql = CONCAT('ALTER TABLE history DROP FOREIGN KEY ', fk_name);
        PREPARE stmt FROM @sql;
        EXECUTE stmt;
        DEALLOCATE PREPARE stmt;
    END IF;

    -- schedule_2 테이블 외래 키 확인 및 삭제
    SELECT CONSTRAINT_NAME INTO fk_name 
    FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE 
    WHERE TABLE_NAME = 'schedule_2' 
    AND COLUMN_NAME = 'user_weight_no' 
    AND CONSTRAINT_NAME != 'PRIMARY' 
    LIMIT 1;

    IF fk_name IS NOT NULL THEN
        SET @sql = CONCAT('ALTER TABLE schedule_2 DROP FOREIGN KEY ', fk_name);
        PREPARE stmt FROM @sql;
        EXECUTE stmt;
        DEALLOCATE PREPARE stmt;
    END IF;
END //

DELIMITER ;

-- 외래 키 제약 조건을 삭제하는 프로시저 호출
CALL drop_fk_if_exists();

-- history 테이블 외래 키 추가
ALTER TABLE history
ADD CONSTRAINT fk_history_user_weight_no FOREIGN KEY (user_weight_no) REFERENCES user_weight(user_weight_no);

-- schedule_2 테이블 외래 키 추가
ALTER TABLE schedule_2
ADD CONSTRAINT fk_schedule_2_user_weight_no FOREIGN KEY (user_weight_no) REFERENCES user_weight(user_weight_no);

-- user_weight 삽입 후 history 업데이트 트리거
DELIMITER //

CREATE TRIGGER after_insert_user_weight
AFTER INSERT ON user_weight
FOR EACH ROW
BEGIN
    DECLARE user_id_value VARCHAR(256);
    
    -- NEW.user_weight_no의 user_id 값을 가져오기 위해 user_weight 테이블 조회
    SELECT user_id INTO user_id_value FROM user_weight WHERE user_weight_no = NEW.user_weight_no;
    
    -- history 테이블에 데이터 삽입
    INSERT INTO history (sc1, user_id, record_time, user_goal, goal_complexity, goal_start_time, goal_end_time, base_weight_no, user_weight_no)
    SELECT s.sc1, s.user_id, s.record_time, s.user_goal, s.goal_complexity, s.goal_start_time, s.goal_end_time, s.base_weight_no, NEW.user_weight_no
    FROM schedule_1 s
    WHERE s.user_id = user_id_value;
END //

DELIMITER ;

-- schedule_2 테이블에 마지막 history 값 삽입 쿼리 (Python 코드에서 호출)
-- 아래는 참고용으로 Python 코드에서 실행하는 예시입니다.
-- INSERT INTO schedule_2 (sc1, user_id, record_time, user_goal, goal_complexity, goal_start_date, goal_end_date, user_weight_no)
-- SELECT h.sc1, h.user_id, h.record_time, h.user_goal, h.goal_complexity, h.goal_start_time, h.goal_end_time, h.user_weight_no
-- FROM history h
-- WHERE h.user_id = 'user_id_value'
-- ORDER BY h.hi1 DESC
-- LIMIT 1;

-- 예시: Python 코드에서 실행
-- cursor.execute("""
-- INSERT INTO schedule_2 (sc1, user_id, record_time, user_goal, goal_complexity, goal_start_date, goal_end_date, user_weight_no)
-- SELECT h.sc1, h.user_id, h.record_time, h.user_goal, h.goal_complexity, h.goal_start_time, h.goal_end_time, h.user_weight_no
-- FROM history h
-- WHERE h.user_id = %s
-- ORDER BY h.hi1 DESC
-- LIMIT 1
-- """, (user_id_value,))
