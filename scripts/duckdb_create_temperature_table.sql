CREATE TABLE temperatura_estacion AS
SELECT *
FROM read_csv('temperatura_diaria_todas_estaciones_filled.csv',
    delim = ',',
    header = true,
    columns = {
        'dia': 'DATE',
        'temp_max': 'DOUBLE',
        'temp_min': 'DOUBLE',
        'temp_media': 'DOUBLE',
        'estacion': 'VARCHAR'
    });