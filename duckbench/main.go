package main

import (
	"database/sql"
	"fmt"
	"log"
	"strconv"
	"strings"
	"sync"
	"time"

	_ "github.com/duckdb/duckdb-go/v2"
)

type Row struct {
	ID    int64
	Valor float32
	Dia time.Time
	Hora int32
}

type CompleteRow struct {
	ID    int64
	Valor float64
	Dia time.Time
	Hora int32
	Tarifa string
	Departamento string
	Localidad *string
	Oficina string
	Origen string
}

type Segment struct {
	StartDate time.Time
	StartHour int32
	EndDate time.Time
	EndHour int32
	IsZero int32
	NextValue float32
	NextComputed float32
	NextExpected float32
	Count int32
}

type CustomerIdResult struct {
	ID int64
	Missing int32
	MaxStreakMissing int32
	SingleStreakMissing int32
	Zeros int32
	SingleStreakZeros int32
	MaxStreakZeros int32
	Repeated int32
	RepeatedSameValue bool
	Segments []Segment
}

type IdValueResult struct {
	ID int64
	Hora int32
	Value float32
}

var residencialesParquetFilePath = "C:\\Users\\andres\\Documents\\ute\\cleanup\\res"
var cleanupResidencialesPath = "C:\\Users\\andres\\Documents\\ute\\processed\\residenciales.cleanup.parquet"
var analisisResidencialesDuckDbFilePath = "C:\\Users\\andres\\Documents\\analisis"
var residencialesDuckDbFilePath = "C:\\Users\\andres\\Documents\\test-data"
var customersBatch = 5000
var singleBatchExecution = true

func main() {
	metaDB, err := sql.Open("duckdb", analisisResidencialesDuckDbFilePath)
	if err != nil {
		log.Fatalf("sql.Open: %v", err)
	}
	defer metaDB.Close()

	var dataDB *sql.DB;

	dataDB, err = sql.Open("duckdb", residencialesDuckDbFilePath)
	if err != nil {
		log.Fatalf("sql.Open: %v", err)
	}
	defer dataDB.Close()

	// Cleanup de datos

	//Cleanup("C:\\Users\\andres\\Documents\\ute\\cleanup\\res1", "C:\\Users\\andres\\Documents\\ute\\cleanup\\res")
	//FilterIdsCleanup(metaDB, residencialesParquetFilePath, cleanupResidencialesPath)
	FixOutliers(analisisResidencialesDuckDbFilePath, residencialesParquetFilePath)

	//Arregla los agujeros
	// mainProcess(
	// 	metaDB,
	// 	dataDB,
	// 	getIdsToProcessFixMissing,
	// 	readDataFixMissing,
	// 	processBatchFixMissing,
	// 	insertTempDataFixMissing,
	// 	30000 * customersBatch, // daily data for each id in the batch
	//  	2 * customersBatch,
	// 	2000,
	// 	createTempTableFixMissing,
	// 	nil,
	// )

	// Analisis de datos

	// Analiza ceros, agujeros y segmentos
	// mainProcess(
	// 	metaDB,
	// 	dataDB,
	// 	getIdsToProcess,
	// 	readData,
	// 	processBatchCh,
	// 	insertData,
	// 	27000 * customersBatch, // daily data for each id in the batch
	//  	5 * customersBatch,
	// 	1000,
	// 	nil,
	// 	nil,
	// )

	// Calcula daily average
	// mainProcess(
	// 	metaDB,
	// 	dataDB,
	// 	getIdsToProcessDailyAverage,
	// 	readData,
	// 	processBatchDailyAverage,
	// 	insertDataDailyAverage,
	// 	27000 * customersBatch, // daily data for each id in the batch
	//  	5 * customersBatch,
	// 	1000,
	// 	nil,
	// 	nil,
	// )

	//Calcula mediana por hora por id
	// mainProcess[IdValueResult, IdValueResult](
	// 	metaDB,
	// 	dataDB,
	// 	getIdsToProcessMedianHourly,
	// 	insertDataMedianHourly,
	// 	nil,
	// 	nil,
	// 	0,
	//  	0,
	//  	0,
	// 	nil,
	// 	nil,
	// )

	//Calcula mean absolute deviation
	// mainProcess[IdValueResult, IdValueResult](
	// 	metaDB,
	// 	nil,
	// 	getIdsToProcessMadHourly,
	// 	updateDataMadHourly,
	// 	nil,
	// 	nil,
	// 	0,
	// 	0,
	// 	0,
	// 	nil,
	// 	nil,
	// )

	//testAverage(db)
}

func mainProcess[TRead any, TRes any](
	metaDB *sql.DB,
	dataDB *sql.DB,
	getIdsFunc func(db *sql.DB, ids *[]int64),
	readData func (metaDB *sql.DB, dataDB *sql.DB, ids []int64, data *[]TRead),
	processFunc func(data []TRead, resultsCh chan<- TRes),
	insertFunc func (metaDB *sql.DB, dataDB *sql.DB, data []TRes) error,
	readBufferCap int,
	resultsChannelSize int,
	insertBufferSize int,
	beforeFunc func (metaDB *sql.DB, dataDB *sql.DB) error,
	afterFunc func (metaDB *sql.DB, dataDB *sql.DB) error,
) {
	ids := make([]int64, 0, 150000)
	data := make([]TRead, 0, readBufferCap)
	resultsCh := make(chan TRes, resultsChannelSize)
	var wg sync.WaitGroup

	if insertFunc != nil {
		insertBuffer := make([]TRes, 0, insertBufferSize)
		wg.Add(1)
		go insertWorker(metaDB, dataDB, resultsCh, &insertBuffer, &wg, insertFunc)
	}

	if beforeFunc != nil {
		if err := beforeFunc(metaDB, dataDB); err != nil {
			log.Fatalf("beforeFunc failed: %v", err)
		}
	}

	getIdsFunc(metaDB, &ids)

	for len(ids) > 0 {
		batchSize := min(len(ids), customersBatch)

		batch := ids[:batchSize]
		readData(metaDB, dataDB, batch, &data)

		if processFunc != nil {
			processFunc(data, resultsCh)
		}

		ids = ids[batchSize:]
		data = data[:0]

		if singleBatchExecution {
			break
		}
	}
	close(resultsCh)
	wg.Wait()
	if afterFunc != nil {
		if err := afterFunc(metaDB, dataDB); err != nil {
			log.Fatalf("afterFunc failed: %v", err)
		}
	}
}

func insertData(metaDB *sql.DB, dataDB *sql.DB, data []CustomerIdResult) error {
    tx, err := metaDB.Begin()
    if err != nil {
		log.Fatalf("insertData failed: %v", err)
        return err
    }
    defer tx.Rollback() // The rollback will be ignored if the commit happens

    stmtAnalisis, err := tx.Prepare(`
		INSERT INTO analisis (id, missing, zeros, max_streak_zeros, max_streak_missing, departamento,
		localidad, single_streak_zeros, single_streak_missing, repeated, repeated_same_value) VALUES
		(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
	`)
    if err != nil {
        return err
    }
    defer stmtAnalisis.Close()

	stmtSegments, err := tx.Prepare(`
		INSERT INTO segmentos (customer_id, start_date, start_hour, end_date, end_hour, is_zero,
		next_value, next_computed, next_expected, count) VALUES
		(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
	`)
    if err != nil {
        return err
    }
    defer stmtSegments.Close()

	repeatedSameValue := 1
    for _, row := range data {
		if row.RepeatedSameValue {
			repeatedSameValue = 1
		} else {
			repeatedSameValue = 0
		}

        _, err = stmtAnalisis.Exec(row.ID, row.Missing, row.Zeros, row.MaxStreakZeros, row.MaxStreakMissing, "", "",
			row.SingleStreakZeros, row.SingleStreakMissing, row.Repeated, repeatedSameValue,
		)
        if err != nil {
			log.Fatalf("insertData row failed: %v", err)
            return err
        }

		for _, segmentRow := range row.Segments {
			_, err = stmtSegments.Exec(row.ID, segmentRow.StartDate, segmentRow.StartHour, segmentRow.EndDate,
				segmentRow.EndHour, segmentRow.IsZero, segmentRow.NextValue, segmentRow.NextComputed, segmentRow.NextExpected,
				segmentRow.Count,
			)
			if err != nil {
				log.Fatalf("insertData segment failed: %v", err)
				return err
			}
		}
    }

    return tx.Commit()
}

func insertDataDailyAverage(metaDB *sql.DB, dataDB *sql.DB, data []IdValueResult) error {
    tx, err := metaDB.Begin()
    if err != nil {
		log.Fatalf("insertDataDailyAverage failed: %v", err)
        return err
    }
    defer tx.Rollback() // The rollback will be ignored if the commit happens

    stmt, err := tx.Prepare(`
		UPDATE analisis SET daily_average = ? WHERE id = ? 
	`)
    if err != nil {
        return err
    }
    defer stmt.Close()

    for _, row := range data {

        _, err = stmt.Exec(row.Value, row.ID)
        if err != nil {
			log.Fatalf("insertDataDailyAverage  failed: %v", err)
            return err
        }
    }

    return tx.Commit()
}

func insertWorker[T any](
	metaDB *sql.DB,
	dataDB *sql.DB,
	resultsCh <-chan T,
	buf *[]T,
	wg *sync.WaitGroup,
	insertFunc func(metaDB *sql.DB, dataDB *sql.DB, data []T) error,
) {
	defer wg.Done()

	count := 0
	now := time.Now()
	timer := time.NewTimer(time.Second)
	defer timer.Stop()

	for {
		select {
		case result, ok := <-resultsCh:
			if !ok {
				// Channel closed, insert remaining buffer and exit
				if len(*buf) > 0 {
					err := insertFunc(metaDB, dataDB, *buf)
					if err != nil {
						log.Fatalf("insertData commit failed: %v", err)
					}
					*buf = (*buf)[:0]
				}
				return
			}

			*buf = append(*buf, result)

			if len(*buf) == cap(*buf) {
				err := insertFunc(metaDB, dataDB, *buf)
				if err != nil {
					log.Fatalf("insertData commit failed: %v", err)
				} else {
					log.Printf("insertData time: %v", time.Since(now))
					now = time.Now()
				}
				*buf = (*buf)[:0]
			} else if len(*buf) == 1 {
				now = time.Now()
			}

			timer.Reset(time.Second)
			count++

		case <-timer.C:
			// Timeout occurred, insert buffer if it has data
			if len(*buf) > 0 {
				err := insertFunc(metaDB, dataDB, *buf)
				if err != nil {
					log.Fatalf("insertData commit failed: %v", err)
				} else {
					log.Printf("insertData time (timeout): %v", time.Since(now))
					now = time.Now()
				}
				*buf = (*buf)[:0]
			}
			// Don't reset timer here - it will be reset when next item arrives
		}
	}
}


func RowDiff(row1 Row, row2 Row) time.Duration {
	return time.Date(row1.Dia.Year(), row1.Dia.Month(), row1.Dia.Day(), int(row1.Hora-1), 0, 0, 0, row1.Dia.Location()).Sub(
		time.Date(row2.Dia.Year(), row2.Dia.Month(), row2.Dia.Day(), int(row2.Hora-1), 0, 0, 0, row2.Dia.Location()),
	)
}

func CompleteRowDiff(row1 CompleteRow, row2 CompleteRow) time.Duration {
	return time.Date(row1.Dia.Year(), row1.Dia.Month(), row1.Dia.Day(), int(row1.Hora-1), 0, 0, 0, row1.Dia.Location()).Sub(
		time.Date(row2.Dia.Year(), row2.Dia.Month(), row2.Dia.Day(), int(row2.Hora-1), 0, 0, 0, row2.Dia.Location()),
	)
}

func HoursDiff(row1 Row, row2 Row) int32 {
	return int32(RowDiff(row1, row2) / time.Hour)
}

func GetSeason(date time.Time) int {

	switch(date.Month()) {
	case 12,1,2:
		return 0
	case 3,4,5:
		return 1
	case 6,7,8:
		return 2
	default:
		return 3
	}
}

type Key struct {
	season int
	weekDay int
	hour int
}

type Value struct {
	valor float32
	count int
}

func computeAverages(data []Row, id int64, idx int, averages *map[Key]Value) {
	for i:=idx; i < len(data) && (data[i].ID == id); i++ {
		row := data[i]
		if row.Valor != 0 {
			if i > idx && data[i-1].Valor == 0 && HoursDiff(row, data[i-1]) == 1 {
				continue
			}

			mapKey := Key{weekDay:int(row.Dia.Weekday()), hour:int(row.Hora-1), season: GetSeason(row.Dia)}
			if value, ok := (*averages)[mapKey]; ok {
				(*averages)[mapKey] = Value{count: value.count+1, valor: row.Valor + value.valor}
			} else {
				(*averages)[mapKey] = Value{count: 1, valor: row.Valor}
			}
		}
	}
}

func getAverageStreak(averages map[Key]Value, from time.Time, fromHour int, count int) (float32, bool) {
	start := time.Date(from.Year(), from.Month(), from.Day(), fromHour-1, 0, 0, 0, from.Location())

	var result float32 = 0
	totalCount := 0
	for i := range count {
		date := start.Add(time.Duration(i) * time.Hour)
		mapKey := Key{weekDay:int(date.Weekday()), hour: date.Hour(), season: GetSeason(date)}
		if value, ok := averages[mapKey]; ok {
			totalCount++
			result += value.valor / float32(value.count)
		}
	}

	return result, count != totalCount
}

func processBatchCh(data []Row, resultsCh chan<- CustomerIdResult) {
	now := time.Now();
	if len(data) == 0 {
		return
	}

	idx := 0

	var startNewId bool
	var currentIdStart int
	var currentId int64 = -1
	var zeros int32
	var missing int32
	var maxStreakMissing int32
	var maxStreakZeros int32
	var singleStreakMissing int32
	var singleStreakZeros int32
	var streakZerosStart int32
	var repeated int32
	var repeatedSameValue bool
	var diff time.Duration
	var segments []Segment

	averages := make(map[Key]Value)

	for i, row := range data {
		diff = time.Hour
		startNewId = currentId != row.ID
		if startNewId {
			currentIdStart = i
			clear(averages)
			if i > 0 {
				resultsCh <- CustomerIdResult{
					ID: currentId,
					Missing: missing,
					MaxStreakMissing: maxStreakMissing,
					SingleStreakMissing: singleStreakMissing,
					Zeros: zeros,
					MaxStreakZeros: maxStreakZeros,
					SingleStreakZeros: singleStreakZeros,
					Repeated: repeated,
					RepeatedSameValue: repeatedSameValue,
					Segments: segments,
				}

				idx++
			}

			segments = make([]Segment, 0, 1000)
			zeros = 0
			missing = 0
			maxStreakMissing = 0
			maxStreakZeros = 0
			singleStreakMissing = 0
			singleStreakZeros = 0
			repeated = 0
			repeatedSameValue = true
			streakZerosStart = -1

			currentId = row.ID
		} else {
			diff = RowDiff(row, data[i-1])

			if (diff % time.Hour != 0) {
				panic("safety check violated")
			}

			if diff == 0 {
				repeated++
				if repeatedSameValue {
					repeatedSameValue = row.Valor == data[i-1].Valor
				}
				continue
			}

			missingCount := int32(diff / time.Hour) - 1
			missing += missingCount

			if missingCount > maxStreakMissing {
				maxStreakMissing = missingCount
			}

			if missingCount == 1 {
				singleStreakMissing++
			}
		}

		if row.Valor == 0 {
			zeros++

			if diff > time.Hour && streakZerosStart >= 0 {
				zerosCount := HoursDiff(data[i-1], data[streakZerosStart]) + 1
				if zerosCount > maxStreakZeros {
					maxStreakZeros = zerosCount
				}
				if zerosCount == 1 {
					singleStreakZeros++
				}
				streakZerosStart = -1
			}

			if streakZerosStart < 0 {
				streakZerosStart = int32(i)
			}
		} else if streakZerosStart >= 0 {
			zerosCount := HoursDiff(data[i-1], data[streakZerosStart]) + 1
			if zerosCount > maxStreakZeros {
				maxStreakZeros = zerosCount
			}
			if zerosCount == 1 {
				singleStreakZeros++
			}

			segment := Segment{
				StartDate: data[streakZerosStart].Dia,
				StartHour: data[streakZerosStart].Hora,
				EndDate: data[i-1].Dia,
				EndHour: data[i-1].Hora,
				NextValue: 0,
				NextComputed: 0,
				NextExpected: 0,
				IsZero: 1,
				Count: zerosCount,
			}

			if zerosCount > 1 && diff == time.Hour {
				if len(averages) == 0 {
					computeAverages(data, currentId, currentIdStart, &averages)
				}

				//average, warn := getAverageStreak(averages, data[streakZerosStart].Dia, int(data[streakZerosStart].Hora), int(zerosCount))
				average, _ := getAverageStreak(averages, data[streakZerosStart].Dia, int(data[streakZerosStart].Hora), int(zerosCount))
				segment.NextValue = row.Valor
				if value, ok := averages[Key{season: GetSeason(row.Dia), weekDay: int(row.Dia.Weekday()), hour: int(row.Hora-1)}]; ok {
					segment.NextExpected = value.valor / float32(value.count)
				} else {
					segment.NextExpected = -1
				}
				segment.NextComputed = average
				// threshold := 0.9 * average
				// if row.Valor > threshold && !warn {
				// 	expected := ""
				// 	if value, ok := averages[Key{season: GetSeason(row.Dia), weekDay: int(row.Dia.Weekday()), hour: int(row.Hora-1)}]; ok {
				// 		expected = fmt.Sprintf("%f", value.valor / float32(value.count))
				// 	}

				// 	log.Printf("[%d][%d-%d-%d %d] - streak=%d value=%f threshold=%f computed=%f expected=%s",
				// 		row.ID, row.Dia.Year(), row.Dia.Month(), row.Dia.Day(), row.Hora, zerosCount, row.Valor, threshold,
				// 		average, expected,
				// 	)
				// }
			} else if zerosCount > 1 {
				segment.NextValue = -1
				segment.NextExpected = -1
				segment.NextComputed = -1
			}

			if zerosCount > 1 {
				segments = append(segments, segment)
			}

			streakZerosStart = -1
		}
	}

	if streakZerosStart >= 0 {
		zerosCount := HoursDiff(data[len(data)-1], data[streakZerosStart]) + 1
		if zerosCount > maxStreakZeros {
			maxStreakZeros = zerosCount
		}
		if zerosCount == 1 {
			singleStreakZeros++
		}

		segment := Segment{
			StartDate: data[streakZerosStart].Dia,
			StartHour: data[streakZerosStart].Hora,
			EndDate: data[len(data)-1].Dia,
			EndHour: data[len(data)-1].Hora,
			NextValue: 0,
			NextComputed: 0,
			NextExpected: 0,
			IsZero: 1,
			Count: zerosCount,
		}

		if zerosCount > 1 && diff == time.Hour {
			row := data[len(data) -1]
			if len(averages) == 0 {
				computeAverages(data, currentId, currentIdStart, &averages)
			}

			average, _ := getAverageStreak(averages, data[streakZerosStart].Dia, int(data[streakZerosStart].Hora), int(zerosCount))
			segment.NextValue = row.Valor
			if value, ok := averages[Key{season: GetSeason(row.Dia), weekDay: int(row.Dia.Weekday()), hour: int(row.Hora-1)}]; ok {
				segment.NextExpected = value.valor / float32(value.count)
			} else {
				segment.NextExpected = -1
			}
			segment.NextComputed = average
		} else if zerosCount > 1 {
			segment.NextValue = -1
			segment.NextExpected = -1
			segment.NextComputed = -1
		}

		if zerosCount > 1 {
			segments = append(segments, segment)
		}
	}

	resultsCh <- CustomerIdResult{
		ID: currentId,
		Missing: missing,
		MaxStreakMissing: maxStreakMissing,
		SingleStreakMissing: singleStreakMissing,
		Zeros: zeros,
		MaxStreakZeros: maxStreakZeros,
		SingleStreakZeros: singleStreakZeros,
		Repeated: repeated,
		RepeatedSameValue: repeatedSameValue,
		Segments: segments,
	}

	log.Printf("processBatch time: %v", time.Since(now))
}

func processBatchDailyAverage(data []Row, resultsCh chan<- IdValueResult) {
	now := time.Now();
	if len(data) == 0 {
		return
	}

	var startNewId bool
	var currentId int64 = -1
	var diff time.Duration
	var customerCount int32
	var customerSum float32
	var dailyCount int
	var dailySum float32
	var dailyStart int
	

	for i, row := range data {
		diff = time.Hour
		startNewId = currentId != row.ID
		if startNewId {
			if i > 0 {
				if customerCount > 0 {
					resultsCh <- IdValueResult{ID: currentId, Value: customerSum / float32(customerCount)}
				} else {
					// should never happen
					resultsCh <- IdValueResult{ID: currentId, Value: -1}
				}
			}

			customerCount = 0
			customerSum = 0
			dailySum = 0
			dailyCount = 0
			dailyStart = -1
			currentId = row.ID
		} else {
			diff = RowDiff(row, data[i-1])
		}

		if row.Hora == 1 {
			dailyStart = i
			dailySum = row.Valor
			dailyCount = 1
		} else if dailyStart >= 0 && (diff == time.Hour || diff == 0) && row.Dia.Day() == data[i-1].Dia.Day() {
			if (diff != 0) {
				dailySum += row.Valor
				dailyCount++
			}

			if row.Hora == 24 && dailyCount == 24 {
				customerSum += dailySum
				customerCount++
				dailyStart = -1
				dailySum = 0
				dailyCount = 0

				if i == len(data) - 1 {
					resultsCh <- IdValueResult{ID: currentId, Value: customerSum / float32(customerCount)}
				}
			}
		} else {
			dailyStart = -1
			dailySum = 0
			dailyCount = 0
		}
	}

	if customerCount > 0 {
		resultsCh <- IdValueResult{ID: currentId, Value: customerSum / float32(customerCount)}
	} else {
		// should never happen
		resultsCh <- IdValueResult{ID: currentId, Value: -1}
	}

	log.Printf("processBatch time: %v", time.Since(now))
}

func getIdsToProcess(db *sql.DB, ids *[]int64) {
	query := `
		select fid.id as id
		from filtered_ids fid left join analisis a on a.id=fid.id
		where a.id is null
	`;
	

	rows, err := db.Query(query)
	if err != nil {
		log.Fatalf("getIdsToProcess failed: %v", err)
	}
	defer rows.Close()

	var id int64
	for rows.Next() {
		if err := rows.Scan(&id); err != nil {
			log.Fatalf("getIdsToProcess failed: %v", err)
		}
		*ids = append(*ids, id)
	}
}

func getIdsToProcessDailyAverage(db *sql.DB, ids *[]int64) {
	query := `
		select fid.id as id
		from filtered_ids fid left join analisis a on a.id=fid.id
		where a.id is not null and a.daily_average is null
	`;
	

	rows, err := db.Query(query)
	if err != nil {
		log.Fatalf("getIdsToProcess failed: %v", err)
	}
	defer rows.Close()

	var id int64
	for rows.Next() {
		if err := rows.Scan(&id); err != nil {
			log.Fatalf("getIdsToProcess failed: %v", err)
		}
		*ids = append(*ids, id)
	}
}

func getIdsToProcessMedianHourly(db *sql.DB, ids *[]int64) {
	query := `
		select distinct a.id
		from analisis a left join mad m on m.id=a.id
		where m.id is null
	`;

	rows, err := db.Query(query)
	if err != nil {
		log.Fatalf("getIdsToProcess failed: %v", err)
	}
	defer rows.Close()

	var id int64
	for rows.Next() {
		if err := rows.Scan(&id); err != nil {
			log.Fatalf("getIdsToProcess failed: %v", err)
		}
		*ids = append(*ids, id)
	}
}

func getIdsToProcessMadHourly(db *sql.DB, ids *[]int64) {
	query := `
		select distinct m.id
		from analisis a left join mad m on m.id=a.id
		where m.id is not null and m.mad is null
	`;

	rows, err := db.Query(query)
	if err != nil {
		log.Fatalf("getIdsToProcessMadHourly failed: %v", err)
	}
	defer rows.Close()

	var id int64
	for rows.Next() {
		if err := rows.Scan(&id); err != nil {
			log.Fatalf("getIdsToProcessMadHourly failed: %v", err)
		}
		*ids = append(*ids, id)
	}
}

func readData(metaDB *sql.DB, dataDB *sql.DB, ids []int64, data *[]Row) {
	now := time.Now();
	strs := make([]string, len(ids))
	for i, id := range ids {
		strs[i] = strconv.FormatInt(id, 10)
	}

	query := fmt.Sprintf(`
		select id, valor, dia, hora
		from read_parquet('%s')
		where id in (%s)
		order by id, dia, hora
	`, residencialesParquetFilePath, strings.Join(strs, ", "))

	rows, err := metaDB.Query(query)
	if err != nil {
		log.Fatalf("readData failed: %v", err)
	}
	defer rows.Close()

	var (
		id    int64
		valor float32
		dia time.Time
		hora int32
	)

	log.Println("reading...")
	for rows.Next() {
		if err := rows.Scan(&id, &valor, &dia, &hora); err != nil {
			log.Fatalf("processId failed: %v", err)
		}
		*data = append(*data, Row{ID: id, Valor: valor, Dia: dia, Hora: hora})
	}

	log.Printf("readData time: %v", time.Since(now))
}

func insertDataMedianHourly(metaDB *sql.DB, dataDB *sql.DB, ids []int64, data *[]IdValueResult) {
	now := time.Now();
	strs := make([]string, len(ids))
	for i, id := range ids {
		strs[i] = strconv.FormatInt(id, 10)
	}

	stmt := fmt.Sprintf(`
		INSERT INTO mad (id, hora, median)
		SELECT id, hora, median(valor) as median
		FROM read_parquet('%s')
		WHERE id in (%s)
		GROUP BY id, hora
	`, residencialesParquetFilePath, strings.Join(strs, ", "))

	_, err := metaDB.Exec(stmt)
	if err != nil {
		log.Fatalf("insertDataMedianHourly failed: %v", err)
	}

	log.Printf("insertDataMedianHourly time: %v", time.Since(now))
}

func updateDataMadHourly(metaDB *sql.DB, dataDB *sql.DB, ids []int64, data *[]IdValueResult) {
	now := time.Now();
	strs := make([]string, len(ids))
	for i, id := range ids {
		strs[i] = strconv.FormatInt(id, 10)
	}

	stmt := fmt.Sprintf(`
		UPDATE mad as m
		SET mad = source.mad
		FROM (
			select p.id, p.hora, median(abs(p.valor - m.median)) as mad
			from read_parquet('%s') p inner join mad m on p.id=m.id and p.hora=m.hora
			where p.id in (%s)
			group by p.id, p.hora
		) AS source
		WHERE m.id = source.id AND m.hora = source.hora
	`, residencialesParquetFilePath, strings.Join(strs, ", "))

	_, err := metaDB.Exec(stmt)
	if err != nil {
		log.Fatalf("updateDataMadHourly failed: %v", err)
	}

	log.Printf("updateDataMadHourly time: %v", time.Since(now))
}

func readDataSingle(metaDB *sql.DB, dataDB *sql.DB, customerId int64, data *[]Row) {
	now := time.Now();
	str := strconv.FormatInt(customerId, 10)

	query := fmt.Sprintf(`
		select valor, dia, hora
		from read_parquet('%s')
		where id = (%s)
		order by dia, hora
	`, residencialesParquetFilePath, str)

	rows, err := metaDB.Query(query)
	if err != nil {
		log.Fatalf("readData failed: %v", err)
	}
	defer rows.Close()

	var (
		valor float32
		dia time.Time
		hora int32
	)

	for rows.Next() {
		if err := rows.Scan(&valor, &dia, &hora); err != nil {
			log.Fatalf("processId failed: %v", err)
		}
		*data = append(*data, Row{ID: customerId, Valor: valor, Dia: dia, Hora: hora})
	}

	log.Printf("readDataSingle time: %v", time.Since(now))
}

func testAverage(db *sql.DB) {
	data := make([]Row, 0, 35000)

	readDataSingle(db, nil, 15825830, &data)


	var startNewId bool
	var currentId int64 = -1
	var diff time.Duration
	var customerCount int32
	var customerSum float32
	var dailyCount int
	var dailySum float32
	var dailyStart int
	

	for i, row := range data {
		diff = time.Hour
		startNewId = currentId != row.ID
		if startNewId {
			customerCount = 0
			customerSum = 0
			dailySum = 0
			dailyCount = 0
			dailyStart = -1
			currentId = row.ID
		} else {
			diff = RowDiff(row, data[i-1])
		}

		if row.Hora == 1 {
			dailyStart = i
			dailySum = row.Valor
			dailyCount = 1
		} else if dailyStart >= 0 && (diff == time.Hour || diff == 0) && row.Dia.Day() == data[i-1].Dia.Day() {
			if (diff != 0) {
				dailySum += row.Valor
				dailyCount++
			}

			if row.Hora == 24 && dailyCount == 24 {
				customerSum += dailySum
				customerCount++
				dailyStart = -1
				dailySum = 0
				dailyCount = 0
			}
		} else {
			dailyStart = -1
			dailySum = 0
			dailyCount = 0
		}
	}

	log.Printf("average : %f", customerSum / float32(customerCount))
}
