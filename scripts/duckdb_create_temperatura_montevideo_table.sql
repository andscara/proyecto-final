CREATE TABLE temperatura_montevideo AS
SELECT *
FROM read_csv('open-meteo-34.90S56.19W31m.csv',
    delim = ',',
    header = true,
    columns = {
        'time': 'TIMESTAMP',
        'temperature': 'DOUBLE'
    });