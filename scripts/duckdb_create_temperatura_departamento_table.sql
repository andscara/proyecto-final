CREATE TABLE temperatura_departamento AS
SELECT dia, departamento, avg(temp_max) as temp_max, avg(temp_min) as temp_min, avg(temp_media) as temp_media
FROM temperatura_estacion te INNER JOIN localidad_estacion le ON te.estacion=le.estacion
GROUP BY dia, departamento
ORDER BY departamento, dia;