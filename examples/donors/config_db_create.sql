DROP TABLE IF EXISTS raw CASCADE;
DROP TABLE IF EXISTS raw_projects CASCADE;
DROP TABLE IF EXISTS results CASCADE;

CREATE TABLE raw (
    projectid VARCHAR(50) PRIMARY KEY UNIQUE,
    teacher_acctid VARCHAR,
    schoolid VARCHAR(50),
    school_ncesid DECIMAL,
    school_latitude DECIMAL,
    school_longitude DECIMAL,
    school_city VARCHAR(50),
    school_state VARCHAR(2),
    school_metro VARCHAR(50),
    school_district VARCHAR(200),
    school_county VARCHAR(50),
    school_charter VARCHAR(50),
    school_magnet VARCHAR(50),
    teacher_prefix VARCHAR(50),
    primary_focus_subject VARCHAR(50),
    primary_focus_area VARCHAR(50),
    secondary_focus_subject VARCHAR(50),
    secondary_focus_area VARCHAR(50),
    resource_type VARCHAR(50),
    poverty_level VARCHAR(50),
    grade_level VARCHAR(50),
    total_price_including_optional_support DECIMAL,
    students_reached INT,
    eligible_double_your_impact_match VARCHAR(2),
    date_posted TIMESTAMP,
    datefullyfunded TIMESTAMP
    );

CREATE TABLE raw_projects (
    projectid VARCHAR(50) PRIMARY KEY UNIQUE,
    teacher_acctid VARCHAR,
    -- schoolid VARCHAR(50),
    -- school_ncesid DECIMAL,
    -- school_latitude DECIMAL,
    -- school_longitude DECIMAL,
    school_city VARCHAR(50),
    school_state VARCHAR(2),
    -- school_metro VARCHAR(50),
    -- school_district VARCHAR(200),
    school_county VARCHAR(50),
    school_charter VARCHAR(50),
    school_magnet VARCHAR(50),
    -- teacher_prefix VARCHAR(50),
    primary_focus_subject VARCHAR(50),
    -- primary_focus_area VARCHAR(50),
    -- secondary_focus_subject VARCHAR(50),
    -- secondary_focus_area VARCHAR(50),
    resource_type VARCHAR(50),
    poverty_level VARCHAR(50),
    grade_level VARCHAR(50),
    total_price_including_optional_support DECIMAL,
    students_reached INT,
    eligible_double_your_impact_match VARCHAR,
    date_posted TIMESTAMP,
    datefullyfunded TIMESTAMP
    );

CREATE TABLE results (
    train_start TIMESTAMP,
    train_end TIMESTAMP,
    test_start TIMESTAMP,
    test_end TIMESTAMP,
    model_name VARCHAR,
    params JSONB,
    metric VARCHAR,
    threshold DECIMAL,
    metric_value DECIMAL
    );
