package main

import (
	"database/sql"
	"errors"
	"fmt"
	"log"
	"os"
	"strconv"
	"strings"
	"time"

	_ "github.com/duckdb/duckdb-go/v2"
)

func dirExists(path string) bool {
	info, err := os.Stat(path)
	if os.IsNotExist(err) {
		return false
	}

	return err == nil && info.IsDir() 
}

func fileExists(filename string) bool {
	_, err := os.Stat(filename)

	if err == nil {
		return true
	}

	if errors.Is(err, os.ErrNotExist) {
		return false
	}

	return false 
}

func Cleanup(resultDir string, rootDataDir string) {

	for year:=2022; year <= 2025; year++ {
		for month:=1; month <= 12; month++ {
			rawDataPath := fmt.Sprintf("%s\\year=%d\\month=%d", rootDataDir, year, month)

			if !dirExists(rawDataPath) {
				continue
			}

			if fileExists(fmt.Sprintf("%s\\year=%d\\month=%d\\data_0.parquet", resultDir, year, month)) {
				continue
			}

			duckdbFilePath := fmt.Sprintf("%d_%d", year, month)

			now := time.Now()
			CleanupFile(fmt.Sprintf("%s\\*.parquet", rawDataPath), duckdbFilePath, resultDir, year, month)
			log.Printf("cleanup year=%d, month=%d, time=%v\n", year, month, time.Since(now))

			os.Remove(duckdbFilePath)

			if singleBatchExecution {
				return
			}
		}
	}
}

func CleanupFile(parquetFilePath string, duckdbFilePath string, resultsDir string, year int, month int) {
	db, err := sql.Open("duckdb", duckdbFilePath)
	if err != nil {
		log.Fatalf("sql.Open: %v", err)
	}
	defer db.Close()

	query := fmt.Sprintf(`
		create table data as (
			select * from read_parquet('%s')
		)`, parquetFilePath,
	)
	if err := ExecuteQuery(db, query); err != nil {
		log.Fatalf("create data table failed: %v", err)
	}
	log.Println("data loaded")

	query = fmt.Sprintf(`
		create table holes as (
			select * from read_parquet('holes.parquet') where year=%d and month=%d
		)`, year, month,
	)
	if err := ExecuteQuery(db, query); err != nil {
		log.Fatalf("create holes table failed: %v", err)
	}
	log.Println("holes loaded")

	query = `create table ids as (
			select * from read_parquet('ids.parquet')
	)`
	if err := ExecuteQuery(db, query); err != nil {
		log.Fatalf("create ids table failed: %v", err)
	}
	log.Println("ids loaded")

	query = `DELETE FROM data WHERE id NOT IN (select id from ids)`
	if err := ExecuteQuery(db, query); err != nil {
		log.Fatalf("delete ids failed: %v", err)
	}
	log.Println("ids deleted")

	query = `DELETE from data where dia < '2022-08-13'`
	if err := ExecuteQuery(db, query); err != nil {
		log.Fatalf("delete data < failed: %v", err)
	}
	log.Println("range < deleted")

	query = `DELETE from data where dia > '2025-08-19'`
	if err := ExecuteQuery(db, query); err != nil {
		log.Fatalf("delete data > failed: %v", err)
	}
	log.Println("range > deleted")

	query = `
		DELETE FROM data
		WHERE rowid IN (
			SELECT rowid
			FROM (
				SELECT rowid,
					ROW_NUMBER() OVER (
						PARTITION BY id, dia, hora
						ORDER BY rowid
					) AS rn
				FROM data
			)
			WHERE rn > 1
		)	
	`
	if err := ExecuteQuery(db, query); err != nil {
		log.Fatalf("delete duplicates failed: %v", err)
	}
	log.Println("duplicates deleted")

	query = `
		MERGE INTO data AS d
		USING holes AS h
		ON d.id = h.id AND d.dia = h.dia AND d.hora = h.hora
		WHEN NOT MATCHED AND h.dia >= '2022-08-13' AND h.dia <= '2025-08-19' THEN
			INSERT (id, dia, hora, valor, tarifa, departamento, localidad, oficina, origen, year, month)
			VALUES (h.id, h.dia, h.hora, h.valor, h.tarifa, h.departamento, h.localidad, h.oficina, h.origen, h.year, h.month)
	`
	if err := ExecuteQuery(db, query); err != nil {
		log.Fatalf("merge data failed: %v", err)
	}
	log.Println("holes fixed")

	query = fmt.Sprintf(`
		COPY (
			SELECT *
			FROM data
			ORDER BY id, dia, hora
		) TO '%s'
		(
			FORMAT PARQUET,
			PARTITION_BY (year, month),
			OVERWRITE_OR_IGNORE
		);
	`, resultsDir,
	)
	if err := ExecuteQuery(db, query); err != nil {
		log.Fatalf("create parquet file failed: %v", err)
	}
}

func ExecuteQuery(db *sql.DB, query string) error {
	_, err := db.Exec(query)
	return err
}


func FilterIdsCleanup(metaDB *sql.DB, parquetFilePath string, resultFilePath string) error {
	now := time.Now()

	// select ids to keep in result
	subQuery := `
		SELECT a.id
		FROM analisis a INNER JOIN range r ON a.id = r.id
		WHERE a.zeros / (date_diff('hour', min, max)) <= 0.91 AND a.daily_average >= 1 AND 
			(a.max_streak_missing <= 4 OR (a.max_streak_missing <= 24 AND a.daily_average >= 20))
	`
	query := fmt.Sprintf(`
		COPY (
			SELECT *
			FROM read_parquet('%s')
			WHERE id IN (%s)
		) TO '%s' (FORMAT PARQUET)
	`, parquetFilePath,  subQuery, resultFilePath)

	_, err := metaDB.Exec(query)
	if err != nil {
		return fmt.Errorf("failed to delete entries from parquet: %v", err)
	}

	log.Printf("DeleteEntriesFromParquet completed in %v", time.Since(now))

	return nil
}

type UpdateMissingResult struct {
	ID int64
	Value float32
}

func processBatchFixMissing(data []CompleteRow, resultsCh chan<- CompleteRow) {
	now := time.Now();
	if len(data) == 0 {
		return
	}

	var startNewId bool
	var currentIdStart int
	var currentId int64 = -1
	var diff time.Duration

	fixed := 0
	fixedWith0 := 0
	notFixed := 0

	for i, row := range data {
		diff = time.Hour
		startNewId = currentId != row.ID
		if startNewId {
			currentIdStart = i
			currentId = row.ID
		} else {
			diff = CompleteRowDiff(row, data[i-1])

			if (diff % time.Hour != 0) {
				panic("safety check violated")
			}

			if diff <= time.Hour {
				continue
			}

			missingTimestamp := data[i-1].Dia.Add(time.Duration(data[i-1].Hora - 1) * time.Hour)
			for range int32(diff / time.Hour) - 1 {
				missingTimestamp = missingTimestamp.Add(time.Hour)

				if res, ok := fixValue(missingTimestamp, data, currentIdStart, i); ok {
					fixed++
					if res.Valor == 0 {
						fixedWith0++
					}
					resultsCh <- res
				} else {
					notFixed++
				}
			}
		}
	}

	log.Printf("processBatch time: %v, fixed: %d, fixed with zeros: %d, not fixed %d", time.Since(now), fixed, fixedWith0, notFixed)
}

func fixValue(missingTimestamp time.Time, data []CompleteRow, idStart int, currentIndex int) (CompleteRow, bool) {
	// try fix value going 4 weeks backward or forward
	for week := range 5 {
		if res, ok := fixWithPreviousWeekValue(missingTimestamp, data, idStart, currentIndex, week + 1); ok {
			return res, true
		} else if res, ok := fixWithNextWeekValue(missingTimestamp, data, idStart, currentIndex, week + 1); ok {
			return res, true
		}
	}

	return CompleteRow{}, false
}

func fixWithPreviousWeekValue(missingTimestamp time.Time, data []CompleteRow, idStart int, currentIndex int, weekCount int) (CompleteRow, bool) {
	timestamp := missingTimestamp.Add(-time.Duration(weekCount * 7 * 24) * time.Hour)
	year, month, day := timestamp.Date()
	date := time.Date(year, month, day, 0, 0, 0, 0, timestamp.Location())
	hour := timestamp.Hour() + 1

	for j := currentIndex; j>=idStart; j-- {
		if data[j].Dia.Equal(date) && data[j].Hora == int32(hour) {
			return CompleteRow{
				ID: data[idStart].ID,
				Dia: time.Date(missingTimestamp.Year(), missingTimestamp.Month(), missingTimestamp.Day(), 0, 0, 0, 0, missingTimestamp.Location()),
				Hora: int32(missingTimestamp.Hour()) + 1,
				Valor: data[j].Valor,
				Tarifa: data[currentIndex].Tarifa,
				Departamento: data[currentIndex].Departamento,
				Localidad: data[currentIndex].Localidad,
				Oficina: data[currentIndex].Oficina,
				Origen: data[currentIndex].Origen,
			}, true
		}
	}

	return CompleteRow{}, false
}

func fixWithNextWeekValue(missingTimestamp time.Time, data []CompleteRow, idStart int, currentIndex int, weekCount int) (CompleteRow, bool) {
	timestamp := missingTimestamp.Add(time.Duration(weekCount * 7 * 24) * time.Hour)
	year, month, day := timestamp.Date()
	date := time.Date(year, month, day, 0, 0, 0, 0, timestamp.Location())
	hour := timestamp.Hour() + 1
	for j := currentIndex; j < len(data) && data[j].ID == data[idStart].ID; j++ {
		if data[j].Dia.Equal(date) && data[j].Hora == int32(hour) {
			return CompleteRow{
				ID: data[idStart].ID,
				Dia: time.Date(missingTimestamp.Year(), missingTimestamp.Month(), missingTimestamp.Day(), 0, 0, 0, 0, missingTimestamp.Location()),
				Hora: int32(missingTimestamp.Hour()) + 1,
				Valor: data[j].Valor,
				Tarifa: data[currentIndex].Tarifa,
				Departamento: data[currentIndex].Departamento,
				Localidad: data[currentIndex].Localidad,
				Oficina: data[currentIndex].Oficina,
				Origen: data[currentIndex].Origen,
			}, true
		}
	}

	return CompleteRow{}, false
}

func createTempTableFixMissing(metaDB *sql.DB, dataDB *sql.DB) error {

	stmt := `
		CREATE TABLE holes (
			id DECIMAL(19,0),
			dia DATE,
			hora INT,
			valor DOUBLE,
			tarifa VARCHAR(255),
			departamento VARCHAR(255),
			localidad VARCHAR(255),
			oficina VARCHAR(255),
			origen VARCHAR(255),
			year INT,
			month INT
		)
	`

	_, err := dataDB.Exec(stmt)
	
	return err
}

func insertTempDataFixMissing(metaDB *sql.DB, dataDB *sql.DB, data []CompleteRow) error {
    tx, err := dataDB.Begin()
    if err != nil {
		log.Fatalf("insertTempDataFixMissing failed: %v", err)
        return err
    }
    defer tx.Rollback() // The rollback will be ignored if the commit happens

    stmt, err := tx.Prepare(`
		INSERT INTO holes (id, dia, hora, valor, tarifa, departamento, localidad, oficina, origen, year, month)
		VALUES (?,?,?,?,?,?,?,?,?,?,?)
	`)
    if err != nil {
        return err
    }
    defer stmt.Close()

    for _, row := range data {

        _, err = stmt.Exec(row.ID, row.Dia, row.Hora, row.Valor, row.Tarifa, row.Departamento,
			row.Localidad, row.Oficina, row.Origen, row.Dia.Year(), row.Dia.Month(),
		)
        if err != nil {
			log.Fatalf("insertTempDataFixMissing failed: %v", err)
            return err
        }
    }

    return tx.Commit()
}

func getIdsToProcessFixMissing(metaDB *sql.DB, ids *[]int64) {
	query := `
		SELECT a.id
		FROM analisis a INNER JOIN range r ON a.id = r.id
		WHERE a.zeros / (date_diff('hour', min, max)) <= 0.91 AND a.daily_average >= 1 AND 
			(a.max_streak_missing <= 4 OR (a.max_streak_missing <= 24 AND a.daily_average >= 20)) AND missing <> 0
	`;

	rows, err := metaDB.Query(query)
	if err != nil {
		log.Fatalf("getIdsToProcessFixMissing failed: %v", err)
	}
	defer rows.Close()

	var id int64
	for rows.Next() {
		if err := rows.Scan(&id); err != nil {
			log.Fatalf("getIdsToProcessFixMissing failed: %v", err)
		}
		*ids = append(*ids, id)
	}
}

func readDataFixMissing(metaDB *sql.DB, dataDB *sql.DB, ids []int64, data *[]CompleteRow) {
	now := time.Now();
	strs := make([]string, len(ids))
	for i, id := range ids {
		strs[i] = strconv.FormatInt(id, 10)
	}

	query := fmt.Sprintf(`
		select id, valor, dia, hora, tarifa, departamento, localidad, oficina, origen
		from read_parquet('%s')
		where id in (%s)
		order by id, dia, hora
	`, residencialesParquetFilePath, strings.Join(strs, ", "))

	rows, err := dataDB.Query(query)
	if err != nil {
		log.Fatalf("readDataFixMissing failed: %v", err)
	}
	defer rows.Close()

	var (
		id    int64
		valor float64
		dia time.Time
		hora int32
		tarifa string
		departamento string
		localidad *string
		oficina string
		origen string
	)

	for rows.Next() {
		if err := rows.Scan(&id, &valor, &dia, &hora, &tarifa, &departamento, &localidad, &oficina, &origen); err != nil {
			log.Fatalf("read id failed: %v", err)
		}
		*data = append(*data, CompleteRow{
			ID: id, Valor: valor, Dia: dia, Hora: hora,
			Tarifa: tarifa, Departamento: departamento, Localidad: localidad,
			Oficina: oficina, Origen: origen,
		})
	}

	log.Printf("readDataFixMissing time: %v", time.Since(now))
}
