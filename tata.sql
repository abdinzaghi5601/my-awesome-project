CREATE DATABASE TATABANK;

USE TATABANK;

SELECT *
FROM delinquency
WHERE income IS NULL;
   

SHOW COLUMNS FROM delinquency;

set sql_safe_updates = 0 ;

ALTER TABLE delinquency CHANGE COLUMN `ï»¿Customer_ID`  Customer_ID varchar(255);

ALTER TABLE delinquency CHANGE COLUMN `ï»¿Customer_ID` Customer_ID VARCHAR(255);


SELECT *
FROM delinquency
WHERE income IS NULL;

SELECT Customer_ID FROM delinquency WHERE credit_score IS NULL;

SELECT Customer_ID
FROM delinquency
WHERE credit_score = '';

SELECT Customer_ID
FROM delinquency
WHERE TRIM(credit_score) = '';

SELECT *
FROM delinquency;
	
    
SELECT *
FROM delinquency
WHERE credit_score = 'N/A'
   OR credit_score = 'NA'
   OR credit_score = '-';    
   
   
DESCRIBE delinquency;
-- OR
SHOW COLUMNS FROM delinquency LIKE 'credit_score';   	


SELECT *
FROM delinquency
WHERE credit_score IS NULL         -- Catches true NULLs
   OR credit_score = ''             -- Catches empty strings
   OR TRIM(credit_score) = '';      -- Catches strings with only spaces
   
   
SELECT Customer_ID
FROM delinquency
WHERE credit_score IS NULL
   OR credit_score = ''             -- This might implicitly cast '' to 0 for numeric types, but explicit is clearer
   OR TRIM(credit_score) = ''
   OR credit_score = 0;             -- Specifically for numeric 0s   
   
SELECT COUNT(*)
FROM delinquency
WHERE Income IS NULL;   

SELECT COUNT(*)
FROM delinquency
WHERE Credit_Score IS NULL;



use tatabank;


select * from dataset;


create table payments as 
select 	age,
		income,
        missed_payments,
        debt_to_income_ratio,
        employment_status
from dataset;



select * from payments;        


select distinct(employment_status) from payments;


UPDATE 	payments
SET	employment_status = 'employed'
where employment_status = 'Employed';

