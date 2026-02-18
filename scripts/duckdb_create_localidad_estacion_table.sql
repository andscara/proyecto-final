CREATE TABLE localidad_estacion AS
SELECT *
FROM read_csv('localidades_con_estacion_proxima.csv',
    delim = ',',
    header = true,
    columns = {
        'departamento': 'VARCHAR',
        'localidad': 'VARCHAR',
        'estacion': 'VARCHAR'
    });


