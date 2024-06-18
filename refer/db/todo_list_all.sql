-- 데이터베이스 선택
USE todo;

-- 외래 키 제약 조건 존재 여부 확인 및 삭제
SET @fk_name = (SELECT CONSTRAINT_NAME 
                FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE 
                WHERE TABLE_NAME = 'history' AND CONSTRAINT_NAME = 'fk_history_schedule_1');

SET @query = IF(@fk_name IS NOT NULL, CONCAT('ALTER TABLE history DROP FOREIGN KEY ', @fk_name), 'SELECT 1');
PREPARE stmt FROM @query;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

SET @fk_name = (SELECT CONSTRAINT_NAME 
                FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE 
                WHERE TABLE_NAME = 'history' AND CONSTRAINT_NAME = 'fk_history_user_weight');

SET @query = IF(@fk_name IS NOT NULL, CONCAT('ALTER TABLE history DROP FOREIGN KEY ', @fk_name), 'SELECT 1');
PREPARE stmt FROM @query;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

SET @fk_name = (SELECT CONSTRAINT_NAME 
                FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE 
                WHERE TABLE_NAME = 'history' AND CONSTRAINT_NAME = 'fk_history_schedule_1_new');

SET @query = IF(@fk_name IS NOT NULL, CONCAT('ALTER TABLE history DROP FOREIGN KEY ', @fk_name), 'SELECT 1');
PREPARE stmt FROM @query;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

SET @fk_name = (SELECT CONSTRAINT_NAME 
                FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE 
                WHERE TABLE_NAME = 'history' AND CONSTRAINT_NAME = 'fk_history_user_weight_new');

SET @query = IF(@fk_name IS NOT NULL, CONCAT('ALTER TABLE history DROP FOREIGN KEY ', @fk_name), 'SELECT 1');
PREPARE stmt FROM @query;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

SET @fk_name = (SELECT CONSTRAINT_NAME 
                FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE 
                WHERE TABLE_NAME = 'schedule_2' AND CONSTRAINT_NAME = 'fk_schedule_2_history');

SET @query = IF(@fk_name IS NOT NULL, CONCAT('ALTER TABLE schedule_2 DROP FOREIGN KEY ', @fk_name), 'SELECT 1');
PREPARE stmt FROM @query;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

SET @fk_name = (SELECT CONSTRAINT_NAME 
                FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE 
                WHERE TABLE_NAME = 'schedule_2' AND CONSTRAINT_NAME = 'fk_schedule_2_user_weight');

SET @query = IF(@fk_name IS NOT NULL, CONCAT('ALTER TABLE schedule_2 DROP FOREIGN KEY ', @fk_name), 'SELECT 1');
PREPARE stmt FROM @query;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

SET @fk_name = (SELECT CONSTRAINT_NAME 
                FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE 
                WHERE TABLE_NAME = 'schedule_2' AND CONSTRAINT_NAME = 'fk_schedule_2_history_new');

SET @query = IF(@fk_name IS NOT NULL, CONCAT('ALTER TABLE schedule_2 DROP FOREIGN KEY ', @fk_name), 'SELECT 1');
PREPARE stmt FROM @query;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

SET @fk_name = (SELECT CONSTRAINT_NAME 
                FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE 
                WHERE TABLE_NAME = 'schedule_2' AND CONSTRAINT_NAME = 'fk_schedule_2_user_weight_new');

SET @query = IF(@fk_name IS NOT NULL, CONCAT('ALTER TABLE schedule_2 DROP FOREIGN KEY ', @fk_name), 'SELECT 1');
PREPARE stmt FROM @query;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;

-- 기존 트리거 삭제
DROP TRIGGER IF EXISTS after_update_schedule_1_to_history;
DROP TRIGGER IF EXISTS after_insert_history_to_schedule_2;

-- 기존 프로시저 삭제
DROP PROCEDURE IF EXISTS add_column_if_not_exists;
DROP PROCEDURE IF EXISTS update_base_weight_no;

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

CREATE PROCEDURE update_base_weight_no()
BEGIN
    DECLARE done INT DEFAULT 0;
    DECLARE v_sc1 INT;
    DECLARE v_user_id VARCHAR(256);
    DECLARE v_gender INT;
    DECLARE v_job INT;
    DECLARE v_mbti INT;
    DECLARE v_age INT;
    DECLARE v_base_weight_no INT;

    DECLARE cur CURSOR FOR
        SELECT sc1, user_id FROM schedule_1 WHERE base_weight_no IS NULL;
    DECLARE CONTINUE HANDLER FOR NOT FOUND SET done = 1;

    OPEN cur;

    read_loop: LOOP
        FETCH cur INTO v_sc1, v_user_id;
        IF done THEN
            LEAVE read_loop;
        END IF;

        -- user_info 테이블에서 사용자 정보 가져오기
        SELECT gender, job, mbti, age INTO v_gender, v_job, v_mbti, v_age
        FROM user_info
        WHERE user_id = v_user_id;

        -- base_weight 테이블에서 일치하는 no 값 가져오기
        SELECT no INTO v_base_weight_no
        FROM base_weight
        WHERE gender = v_gender AND job = v_job AND mbti = v_mbti AND age = v_age
        LIMIT 1;

        -- 일치하는 값이 없으면 base_weight에 새로운 항목 추가
        IF v_base_weight_no IS NULL THEN
            INSERT INTO base_weight (gender, job, mbti, age)
            VALUES (v_gender, v_job, v_mbti, v_age);

            -- 방금 추가한 항목의 no 값 가져오기
            SELECT no INTO v_base_weight_no
            FROM base_weight
            WHERE gender = v_gender AND job = v_job AND mbti = v_mbti AND age = v_age
            LIMIT 1;
        END IF;

        -- schedule_1 테이블의 base_weight_no 업데이트
        UPDATE schedule_1
        SET base_weight_no = v_base_weight_no
        WHERE sc1 = v_sc1;
    END LOOP;

    CLOSE cur;
END //

DELIMITER ;

-- 프로시저 실행
CALL add_column_if_not_exists();

-- history 테이블에 외래 키 설정 (이름 변경)
ALTER TABLE history
ADD CONSTRAINT fk_history_schedule_1_new FOREIGN KEY (sc1) REFERENCES schedule_1(sc1),
ADD CONSTRAINT fk_history_user_weight_new FOREIGN KEY (user_weight_no) REFERENCES user_weight(user_weight_no);

-- schedule_2 테이블에 외래 키 설정 (이름 변경)
ALTER TABLE schedule_2
ADD CONSTRAINT fk_schedule_2_history_new FOREIGN KEY (history_no) REFERENCES history(history_no),
ADD CONSTRAINT fk_schedule_2_user_weight_new FOREIGN KEY (user_weight_no) REFERENCES user_weight(user_weight_no);

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
    INSERT INTO schedule_2 (history_no, sc1, user_id, record_time, user_goal, goal_complexity, goal_start_time, goal_end_time, user_weight_no)
    VALUES (v_history_no, NEW.sc1, NEW.user_id, NEW.record_time, NEW.user_goal, NEW.goal_complexity, NEW.goal_start_time, NEW.goal_end_time, NEW.user_weight_no);
END //

DELIMITER ;
