-- This runs automatically when PostgreSQL container starts for the first time
CREATE SCHEMA IF NOT EXISTS nasa;

CREATE TABLE IF NOT EXISTS nasa.raw_train (
    id        SERIAL PRIMARY KEY,
    unit_id   INTEGER,
    cycles    INTEGER,
    setting_1 FLOAT, setting_2 FLOAT, setting_3 FLOAT,
    s1  FLOAT, s2  FLOAT, s3  FLOAT, s4  FLOAT, s5  FLOAT,
    s6  FLOAT, s7  FLOAT, s8  FLOAT, s9  FLOAT, s10 FLOAT,
    s11 FLOAT, s12 FLOAT, s13 FLOAT, s14 FLOAT, s15 FLOAT,
    s16 FLOAT, s17 FLOAT, s18 FLOAT, s19 FLOAT, s20 FLOAT,
    s21 FLOAT
);

CREATE TABLE IF NOT EXISTS nasa.raw_test (
    id        SERIAL PRIMARY KEY,
    unit_id   INTEGER,
    cycles    INTEGER,
    setting_1 FLOAT, setting_2 FLOAT, setting_3 FLOAT,
    s1  FLOAT, s2  FLOAT, s3  FLOAT, s4  FLOAT, s5  FLOAT,
    s6  FLOAT, s7  FLOAT, s8  FLOAT, s9  FLOAT, s10 FLOAT,
    s11 FLOAT, s12 FLOAT, s13 FLOAT, s14 FLOAT, s15 FLOAT,
    s16 FLOAT, s17 FLOAT, s18 FLOAT, s19 FLOAT, s20 FLOAT,
    s21 FLOAT
);

CREATE TABLE IF NOT EXISTS nasa.test_rul (
    unit_id  INTEGER PRIMARY KEY,
    true_rul INTEGER
);