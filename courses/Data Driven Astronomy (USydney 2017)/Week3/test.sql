SELECT koi_name, radius FROM Planet 
WHERE radius < 2;

SELECT radius, t_eff FROM Star WHERE radius > 1;

SELECT radius FROM  Star
WHERE radius BETWEEN 1 AND 2;

SELECT kepler_id, t_eff FROM  Star
WHERE t_eff BETWEEN 5000 AND 6000;

SELECT kepler_name, radius FROM Planet 
WHERE (kepler_name <> NULL or status = 'CONFIRMED') AND (radius BETWEEN 1 AND 3);

SELECT koi_name, radius FROM Planet 
ORDER BY radius DESC
LIMIT 5;

SELECT MIN(radius), MAX(radius), AVG(radius), STDDEV(radius) FROM Planet 
WHERE kepler_name IS NULL;

SELECT radius, COUNT(koi_name) 
FROM Planet 
GROUP BY radius
HAVING COUNT(koi_name) > 1;

SELECT radius, COUNT(koi_name) 
FROM Planet 
WHERE t_eq BETWEEN 500 AND 1000
GROUP BY radius
HAVING COUNT(koi_name) > 1;

SELECT kepler_id, COUNT(koi_name) 
FROM Planet 
GROUP BY kepler_id
HAVING COUNT(koi_name) > 1
ORDER BY COUNT(koi_name) DESC;

INSERT INTO Star (kepler_id, t_eff, radius) VALUES
  (2713050, 5000, 0.956),
  (2713051, 3100, 1.321);
  
DELETE FROM Planet;

UPDATE Star
SET t_eff = 6000
WHERE kepler_id = 2713049;

INSERT INTO Star (kepler_id, t_eff, radius) VALUES
(7115384,	3789,	27.384),
(8106973,	5810,	0.811),
(9391817,	6200,	0.958);

UPDATE Planet
SET kepler_name = NULL
WHERE status <> 'CONFIRMED';

DELETE FROM Planet
WHERE radius < 0;

CREATE TABLE Star (
  kepler_id INTEGER,
  t_eff INTEGER,
  radius FLOAT
);

INSERT INTO Star VALUES
  (10341777, 6302, 0.815);
  
CREATE TABLE Star (
  kepler_id INTEGER PRIMARY KEY,
  t_eff INTEGER CHECK (t_eff > 3000),
  radius FLOAT
);

INSERT INTO Star VALUES
  (10341777, 6302, 0.815);

CREATE TABLE Star (
  kepler_id INTEGER
);

INSERT INTO Star VALUES (3.141);
SELECT * FROM Star;

INSERT INTO Star VALUES ('a string');
SELECT * FROM Star;

CREATE TABLE Star (
  kepler_id INTEGER CHECK(kepler_id > 10)
);

INSERT INTO Star VALUES (3);
SELECT * FROM Star;

CREATE TABLE Planet (
  kepler_id INTEGER NOT NULL,
  koi_name VARCHAR(15) UNIQUE NOT NULL,
  kepler_name VARCHAR(15),
  status VARCHAR(20) NOT NULL,
  radius FLOAT NOT NULL
);

INSERT INTO Planet (kepler_id, koi_name, kepler_name, status, radius) VALUES
(6862328,'K00865.01',NULL,'CANDIDATE',119.021),
(10187017,'K00082.05','Kepler-102 b','CONFIRMED',5.286),
(10187017,'K00082.04','Kepler-102 c','CONFIRMED',7.071);

\d Planet;

CREATE TABLE Star (
  kepler_id INTEGER PRIMARY KEY 
);
  
CREATE TABLE Planet (
  kepler_id INTEGER REFERENCES Star (kepler_id)
);
  
INSERT INTO Star VALUES (10341777);
INSERT INTO Planet VALUES (10341777);

CREATE TABLE Star (
  kepler_id INTEGER PRIMARY KEY,
  t_eff INTEGER,
  radius FLOAT
);

COPY Star (kepler_id, t_eff, radius) 
  FROM 'stars.csv' CSV;

SELECT * FROM Star;

CREATE TABLE Star (
  kepler_id INTEGER PRIMARY KEY,
  t_eff INTEGER NOT NULL,
  radius FLOAT NOT NULL
);

CREATE TABLE Planet (
  kepler_id INTEGER REFERENCES Star (kepler_id),
  koi_name VARCHAR(20) PRIMARY KEY,
  kepler_name VARCHAR(20),
  status VARCHAR(20) NOT NULL,
  period FLOAT,
  radius FLOAT,
  t_eq INTEGER  
);

COPY Star (kepler_id, t_eff, radius) 
  FROM 'stars.csv' CSV;

COPY Planet (kepler_id, koi_name, kepler_name, status, period, radius, t_eq) 
  FROM 'planets.csv' CSV;


SELECT * FROM Star LIMIT 1;

ALTER TABLE Star
ADD COLUMN ra FLOAT,
ADD COLUMN decl FLOAT;
 
SELECT * FROM Star LIMIT 1;

ALTER TABLE Star
DROP COLUMN ra, 
DROP COLUMN decl;
 
SELECT * FROM Star LIMIT 1;

\d Star;

ALTER TABLE Star
 ALTER COLUMN t_eff SET DATA TYPE FLOAT;
 
ALTER TABLE Star
  ADD CONSTRAINT radius CHECK(radius > 0);
 
\d Star;

SELECT * FROM Star;

ALTER TABLE Star
ADD COLUMN ra FLOAT,
ADD COLUMN decl FLOAT;
 
DELETE FROM Star;

COPY Star (kepler_id, t_eff, radius, ra, decl) 
  FROM 'stars_full.csv' CSV;
  
SELECT * FROM Star;
  
SELECT Star.kepler_id, Planet.koi_name
FROM Star, Planet
WHERE Star.kepler_id = Planet.kepler_id;

SELECT Star.kepler_id, Planet.koi_name
FROM Star, Planet

SELECT Star.radius as sun_radius, Planet.radius as planet_radius
FROM Star, Planet
WHERE Star.kepler_id = Planet.kepler_id AND Star.radius / Planet.radius > 1
ORDER BY sun_radius DESC;

SELECT Star.kepler_id, Planet.koi_name
FROM Star
JOIN Planet USING (kepler_id);

SELECT Star.kepler_id, Planet.koi_name
FROM Star
JOIN Planet ON Star.kepler_id = Planet.kepler_id;

SELECT Star.kepler_id, Planet.koi_name
FROM Star
JOIN Planet ON Star.radius > 1.5 AND Planet.t_eq > 2000;

SELECT Star.radius, COUNT(1)
FROM Star
JOIN Planet ON Star.kepler_id = Planet.kepler_id AND Star.radius > 1
GROUP BY Star.kepler_id HAVING COUNT(1) > 1
ORDER BY Star.radius DESC;

SELECT S.kepler_id, P.koi_name
FROM Star S
LEFT OUTER JOIN Planet P USING(kepler_id);

SELECT S.kepler_id, P.koi_name
FROM Star S
RIGHT OUTER JOIN Planet P USING(kepler_id);

SELECT S.kepler_id, S.t_eff, S.radius
FROM Star S
LEFT OUTER JOIN Planet P USING(kepler_id) 
WHERE P.koi_name IS NULL
ORDER BY S.t_eff DESC;

SELECT * FROM Star
WHERE Star.radius > (
  SELECT AVG(radius) FROM Star
);

\timing
-- Join with subqueries
SELECT s.kepler_id 
FROM Star s
WHERE s.kepler_id IN (
  SELECT p.kepler_id FROM Planet p
  WHERE p.radius < 1
);

-- Join with JOIN operator
SELECT DISTINCT(s.kepler_id)
FROM Star s
JOIN Planet p USING (kepler_id)
WHERE p.radius < 1;

--SELECT ROUND(p.t_eq, 1), min(s.t_eff), max(s.t_eff) 
--FROM Star s
--WHERE s.kepler_id IN (
--  SELECT p.kepler_id FROM Planet p
--  WHERE s.t_eff > (SELECT avg(t_eff) FROM Star)
--);
;
SELECT ROUND(avg(p.t_eq), 1), min(s.t_eff), max(s.t_eff) 
FROM Star s
JOIN Planet p USING (kepler_id)
WHERE s.t_eff > (SELECT avg(t_eff) FROM Star);
--GROUP BY s.kepler_id, p.kepler_id

SELECT p.koi_name, p.radius, s.radius
FROM Star s
JOIN Planet p USING (kepler_id)
WHERE s.radius in
(SELECT radius FROM Star
ORDER BY radius DESC
LIMIT 5);