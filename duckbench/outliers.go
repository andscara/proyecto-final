package main

import (
	"database/sql"
	"fmt"
	"log"
	"os"
	"strconv"
	"strings"
	"time"

	_ "github.com/duckdb/duckdb-go/v2"
)

type OutlierRow struct {
	ID    int64
	Valor float32
	Dia time.Time
	Hora int32
	IsOutlier bool
}

func FixOutliers(analisisFilePath string, dataRootDir string) {
	//outliersParquet := "outliers.parquet"

	metaDB, err := sql.Open("duckdb", analisisFilePath)
	if err != nil {
		log.Fatalf("sql.Open: %v", err)
	}
	defer metaDB.Close()

	outliersParquet := "outliers.parquet"

	if !fileExists(outliersParquet) {
		ComputeOutliers(metaDB, dataRootDir)

		mainProcess(
			metaDB,
			nil,
			getIdsToProcessOutliers,
			readDataOutliers,
			processBatchOutliers,
			insertDataOutliers,
			27000 * customersBatch,
			5 * customersBatch,
			1000,
			nil,
			afterBatchOutliers,
		)
	}

	UpdateAllWithOutliers(
		"C:\\Users\\andres\\Documents\\ute\\cleanup\\res-outliers",
		"C:\\Users\\andres\\Documents\\ute\\cleanup\\res",
		outliersParquet,
	)
}

func UpdateAllWithOutliers(resultDir string, rootDataDir string, outliersParquet string) {

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
			UpdateSingleParquetWithOutliers(fmt.Sprintf("%s\\*.parquet", rawDataPath), duckdbFilePath, resultDir, outliersParquet, year, month)
			log.Printf("cleanup year=%d, month=%d, time=%v\n", year, month, time.Since(now))

			os.Remove(duckdbFilePath)

			if singleBatchExecution {
				return
			}
		}
	}
}

func UpdateSingleParquetWithOutliers(
	parquetFilePath string,
	duckdbFilePath string,
	resultsDir string,
	outliersParquet string,
	year int,
	month int,
) {
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
		create table outliers as (
			select * from read_parquet('%s')
		)`, outliersParquet,
	)
	if err := ExecuteQuery(db, query); err != nil {
		log.Fatalf("create holes table failed: %v", err)
	}
	log.Println("outliers loaded")

	query = `
		MERGE INTO data AS d
		USING outliers AS o
		ON d.id = o.id AND d.dia = o.dia AND d.hora = o.hora
		WHEN MATCHED THEN
			UPDATE SET valor = o.valor
	`
	if err := ExecuteQuery(db, query); err != nil {
		log.Fatalf("merge data failed: %v", err)
	}
	log.Println("outliers fixed")

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

func afterBatchOutliers(metaDB *sql.DB, dataDB *sql.DB) error {
	query := fmt.Sprintf(`
		COPY (
			SELECT *
			FROM outliers
			ORDER BY id, dia, hora
		) TO '%s'
		(
			FORMAT PARQUET,
			OVERWRITE_OR_IGNORE
		);
	`, "outliers.parquet",
	)

	return ExecuteQuery(metaDB, query)
}

func getIdsToProcessOutliers(db *sql.DB, ids *[]int64) {
	query := `
		select distinct id
		from outliers
		where updated=0
	`;

	rows, err := db.Query(query)
	if err != nil {
		log.Fatalf("getIdsToProcessOutliers failed: %v", err)
	}
	defer rows.Close()

	var id int64
	for rows.Next() {
		if err := rows.Scan(&id); err != nil {
			log.Fatalf("getIdsToProcessOutliers failed: %v", err)
		}
		*ids = append(*ids, id)
	}
}

func insertDataOutliers(metaDB *sql.DB, dataDB *sql.DB, data []Row) error {
    tx, err := metaDB.Begin()
    if err != nil {
		log.Fatalf("insertDataOutliers failed: %v", err)
        return err
    }
    defer tx.Rollback() // The rollback will be ignored if the commit happens

    stmt, err := tx.Prepare(`
		UPDATE outliers SET valor = ?, updated = 1 WHERE id = ? AND dia = ? AND hora = ?
	`)
    if err != nil {
        return err
    }
    defer stmt.Close()

    for _, row := range data {

        _, err = stmt.Exec(row.Valor, row.ID, row.Dia, row.Hora)
        if err != nil {
			log.Fatalf("insertDataOutliers failed: %v", err)
            return err
        }
    }

    return tx.Commit()
}

func processBatchOutliers(data []OutlierRow, resultsCh chan<- Row) {
	now := time.Now();
	if len(data) == 0 {
		return
	}

	var startNewId bool
	var currentIdStart int
	var currentId int64 = -1
	var processed = 0
	var notProcessed = 0

	for i, row := range data {
		startNewId = currentId != row.ID
		if startNewId {
			currentIdStart = i
			currentId = row.ID
		}

		if !row.IsOutlier {
			continue
		}

		if res, ok := getPreviousWeekValue(data, currentIdStart, i, 1); ok {
			processed++;
			resultsCh <- res
		} else {
			notProcessed++
		}
	}

	log.Printf("processBatch time: %v, processed=%d, notProcessed=%d", time.Since(now), processed, notProcessed)
}

func getPreviousWeekValue(data []OutlierRow, idStart int, currentIndex int, weekCount int) (Row, bool) {
	timestamp := data[currentIndex].Dia.Add(time.Duration(data[currentIndex].Hora - 1) * time.Hour)
	timestamp = timestamp.Add(-time.Duration(weekCount * 7 * 24) * time.Hour)
	year, month, day := timestamp.Date()
	date := time.Date(year, month, day, 0, 0, 0, 0, timestamp.Location())
	hour := timestamp.Hour() + 1

	for j := currentIndex - 1; j>=idStart; j-- {
		if data[j].Dia.Equal(date) && data[j].Hora == int32(hour) {
			if !data[j].IsOutlier {
				return Row{
					ID: data[currentIndex].ID,
					Dia: data[currentIndex].Dia,
					Hora: data[currentIndex].Hora,
					Valor: data[j].Valor,
				}, true
			} else {
				date = date.Add(-time.Duration(weekCount * 7 * 24) * time.Hour)
			}
		}
	}

	return Row{}, false
}

func readDataOutliers(metaDB *sql.DB, dataDB *sql.DB, ids []int64, data *[]OutlierRow) {
	now := time.Now();
	strs := make([]string, len(ids))
	for i, id := range ids {
		strs[i] = strconv.FormatInt(id, 10)
	}

	query := fmt.Sprintf(`
		select p.id, p.valor, p.dia, p.hora, case when o.id is null then 0 else 1 end as outlier
		from read_parquet('%s') p left join outliers o on p.id=o.id and p.dia=o.dia and p.hora=o.hora
		where p.id in (%s)
		order by p.id, p.dia, p.hora
	`, residencialesParquetFilePath, strings.Join(strs, ", "))

	rows, err := metaDB.Query(query)
	if err != nil {
		log.Fatalf("readDataOutliers failed: %v", err)
	}
	defer rows.Close()

	var (
		id    int64
		valor float32
		dia time.Time
		hora int32
		outlier int32
	)

	log.Println("reading...")
	for rows.Next() {
		if err := rows.Scan(&id, &valor, &dia, &hora, &outlier); err != nil {
			log.Fatalf("processId failed: %v", err)
		}
		*data = append(*data, OutlierRow{ID: id, Valor: valor, Dia: dia, Hora: hora, IsOutlier: outlier > 0})
	}

	log.Printf("readDataOutliers time: %v", time.Since(now))
}

func ComputeOutliers(db *sql.DB, dataRootDir string) {
	query := "DROP TABLE IF EXISTS outliers"
	_, err := db.Exec(query)

	if err != nil {
		log.Fatalf("error dropping outliers table")
	}
	
	query = fmt.Sprintf(`
		CREATE TABLE outliers AS (
		select p.id, p.dia, p.hora, p.valor, 0 AS updated
		from read_parquet('%s') p
			inner join mad m on p.id=m.id and p.hora=m.hora
		where (0.6745 * (p.valor - m.median)/m.mad >= 6) and p.valor > 40 )
	`, dataRootDir)
	_, err = db.Exec(query)

	if err != nil {
		log.Fatalf("error creating outliers parquet")
	}
}
