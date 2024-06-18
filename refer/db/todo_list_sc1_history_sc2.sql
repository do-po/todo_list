-- 데이터베이스 선택
USE todo;

-- 기존 프로시저 삭제
DROP PROCEDURE IF EXISTS add_column_if_not_exists;

-- 새로운 프로시저 생성
DELIMITER //

CREATE PROCEDURE add_column_if_not_exists()
BEGIN
    DECLARE col_exists INT;

    -- user_weight 테이블에 history_no 컬럼 존재 여부 확인
    SELECT COUNT(*) INTO col_exists
    FROM INFORMATION_SCHEMA.COLUMNS
    WHERE TABLE_NAME='user_weight' AND COLUMN_NAME='history_no';

    -- 컬럼이 존재하지 않을 경우 추가
    IF col_exists = 0 THEN
        ALTER TABLE user_weight ADD COLUMN history_no INT;
    END IF;

    -- history 테이블에 user_weight_no 컬럼 존재 여부 확인
    SELECT COUNT(*) INTO col_exists
    FROM INFORMATION_SCHEMA.COLUMNS
    WHERE TABLE_NAME='history' AND COLUMN_NAME='user_weight_no';

    -- 컬럼이 존재하지 않을 경우 추가
    IF col_exists = 0 THEN
        ALTER TABLE history ADD COLUMN user_weight_no INT;
    END IF;
END //

DELIMITER ;

-- 프로시저 실행
CALL add_column_if_not_exists();

-- history 테이블에 외래 키 설정
ALTER TABLE history
ADD CONSTRAINT fk_history_schedule_1 FOREIGN KEY (sc1) REFERENCES schedule_1(sc1),
ADD CONSTRAINT fk_history_user_weight FOREIGN KEY (user_weight_no) REFERENCES user_weight(user_weight_no);

-- schedule_2 테이블에 외래 키 설정
ALTER TABLE schedule_2
ADD CONSTRAINT fk_schedule_2_history FOREIGN KEY (history_no) REFERENCES history(history_no),
ADD CONSTRAINT fk_schedule_2_user_weight FOREIGN KEY (user_weight_no) REFERENCES user_weight(user_weight_no);

-- schedule_1 데이터 수정 시 history 테이블에 데이터 추가 트리거 생성
DELIMITER //

CREATE TRIGGER after_update_schedule_1_to_history
AFTER UPDATE ON schedule_1
FOR EACH ROW
BEGIN
    DECLARE v_user_weight_no INT;

    -- user_weight 테이블에서 user_weight_no 가져오기
    SELECT user_weight_no INTO v_user_weight_no
    FROM user_weight
    WHERE user_id = NEW.user_id;

    -- history 테이블에 데이터 삽입
    INSERT INTO history (sc1, user_id, record_time, user_goal, goal_complexity, goal_start_time, goal_end_time, user_weight_no)
    VALUES (NEW.sc1, NEW.user_id, NEW.record_time, NEW.user_goal, NEW.goal_complexity, NEW.goal_start_time, NEW.goal_end_time, v_user_weight_no);
END //

DELIMITER ;

-- history 테이블에서 가장 마지막 데이터를 schedule_2 테이블로 추가 트리거 생성
DELIMITER //

CREATE TRIGGER after_insert_history_to_schedule_2
AFTER INSERT ON history
FOR EACH ROW
BEGIN
    DECLARE v_history_no INT;

    -- history 테이블에서 가장 마지막으로 추가된 history_no 가져오기
    SELECT history_no INTO v_history_no
    FROM history
    WHERE user_id = NEW.user_id AND user_goal = NEW.user_goal
    ORDER BY record_time DESC
    LIMIT 1;

    -- schedule_2 테이블에 데이터 삽입
    INSERT INTO schedule_2 (history_no, sc1, user_id, record_time, user_goal, goal_complexity, goal_start_date, goal_end_date, user_weight_no)
    VALUES (v_history_no, NEW.sc1, NEW.user_id, NEW.record_time, NEW.user_goal, NEW.goal_complexity, NEW.goal_start_time, NEW.goal_end_time, NEW.user_weight_no);
END //

DELIMITER ;
