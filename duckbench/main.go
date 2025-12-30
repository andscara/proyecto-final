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
}

var parquetResidencialesPath = "C:\\Users\\andres\\Documents\\ute\\processed\\residenciales.parquet"
var duckdbFilePath = "C:\\Users\\andres\\Documents\\analisis-residenciales"
var customersBatch = 5000

func main() {
	db, err := sql.Open("duckdb", duckdbFilePath)
	if err != nil {
		log.Fatalf("sql.Open: %v", err)
	}
	defer db.Close()

	ids := make([]int64, 0, customersBatch)
	data := make([]Row, 0, 35000 * customersBatch)

	for range 1 {
		getIdsToProcess(db, &ids)

		if len(ids) == 0 {
			log.Printf("early stop")
			break
		}

		readData(db, ids, &data)
		//result := processBatch(len(ids), data)
		processBatch(len(ids), data)
		ids = ids[:0]
		data = data[:0]

		// now := time.Now()
		// insertData(db, result)
		// log.Printf("insertData time: %v", time.Since(now))
	}
}

func insertData(db *sql.DB, data []CustomerIdResult) error {
    tx, err := db.Begin()
    if err != nil {
		log.Fatalf("insertData failed: %v", err)
        return err
    }
    defer tx.Rollback() // The rollback will be ignored if the commit happens

    stmt, err := tx.Prepare(`
		INSERT INTO analisis (id, missing, zeros, max_streak_zeros, max_streak_missing, departamento,
		localidad, single_streak_zeros, single_streak_missing, repeated, repeated_same_value) VALUES
		(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
	`)
    if err != nil {
        return err
    }
    defer stmt.Close()

	repeatedSameValue := 1
    for _, row := range data {
		if row.RepeatedSameValue {
			repeatedSameValue = 1
		} else {
			repeatedSameValue = 0
		}

        _, err = stmt.Exec(row.ID, row.Missing, row.Zeros, row.MaxStreakZeros, row.MaxStreakMissing, "", "",
			row.SingleStreakZeros, row.SingleStreakMissing, row.Repeated, repeatedSameValue,
		)
        if err != nil {
			log.Fatalf("insertData row failed: %v", err)
            return err
        }
    }

    return tx.Commit()
}

func worker(workerId int, db *sql.DB, ids <-chan int64, wg *sync.WaitGroup) {
	defer wg.Done()
	var data []Row = make([]Row, 0, 35000)

	for id := range ids {
		data = data[:0]
		log.Printf("[%d] - Done processing %d", workerId, id)
	}
}



func RowDiff(row1 Row, row2 Row) time.Duration {
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

func processBatch(idCount int, data []Row) []CustomerIdResult {
	now := time.Now();
	if len(data) == 0 {
		return nil
	}

	idx := 0
	result := make([]CustomerIdResult, idCount)

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

	averages := make(map[Key]Value)

	for i, row := range data {
		diff = time.Hour
		startNewId = currentId != row.ID
		if startNewId {
			currentIdStart = i
			clear(averages)
			if i > 0 {
				result[idx].ID = currentId
				result[idx].Missing = missing
				result[idx].MaxStreakMissing = maxStreakMissing
				result[idx].SingleStreakMissing = singleStreakMissing
				result[idx].Zeros = zeros
				result[idx].MaxStreakZeros = maxStreakZeros
				result[idx].SingleStreakZeros = singleStreakZeros
				result[idx].Repeated = repeated
				result[idx].RepeatedSameValue = repeatedSameValue

				idx++
			}

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

			if zerosCount >= 10 && zerosCount <= 100 && diff == time.Hour{
				if len(averages) == 0 {
					computeAverages(data, currentId, currentIdStart, &averages)
				}

				average, warn := getAverageStreak(averages, data[streakZerosStart].Dia, int(data[streakZerosStart].Hora), int(zerosCount))
				threshold := 0.9 * average
				if row.Valor > threshold && !warn {
					expected := ""
					if value, ok := averages[Key{season: GetSeason(row.Dia), weekDay: int(row.Dia.Weekday()), hour: int(row.Hora-1)}]; ok {
						expected = fmt.Sprintf("%f", value.valor / float32(value.count))
					}

					log.Printf("[%d][%d-%d-%d %d] - streak=%d value=%f threshold=%f computed=%f expected=%s",
						row.ID, row.Dia.Year(), row.Dia.Month(), row.Dia.Day(), row.Hora, zerosCount, row.Valor, threshold,
						average, expected,
					)
				}
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
	}

	result[idx].ID = currentId
	result[idx].Missing = missing
	result[idx].MaxStreakMissing = maxStreakMissing
	result[idx].SingleStreakMissing = singleStreakMissing
	result[idx].Zeros = zeros
	result[idx].MaxStreakZeros = maxStreakZeros
	result[idx].SingleStreakZeros = singleStreakZeros
	result[idx].Repeated = repeated
	result[idx].RepeatedSameValue = repeatedSameValue

	log.Printf("processBatch time: %v", time.Since(now))

	return result
}

func getIdsToProcess(db *sql.DB, ids *[]int64) {
	// query := fmt.Sprintf(`
	// 	select fid.id as id
	// 	from filtered_ids fid left join analisis a on a.id=fid.id
	// 	where a.id is null
	// 	limit %d
	// `, customersBatch)
	query := fmt.Sprintf(`
		select id from analisis where max_streak_zeros>=100 and max_streak_zeros<=100
		limit %d
	`, customersBatch)
	

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

func readData(db *sql.DB, ids []int64, data *[]Row) {
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
	`, parquetResidencialesPath, strings.Join(strs, ", "))

	rows, err := db.Query(query)
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

	for rows.Next() {
		if err := rows.Scan(&id, &valor, &dia, &hora); err != nil {
			log.Fatalf("processId failed: %v", err)
		}
		*data = append(*data, Row{ID: id, Valor: valor, Dia: dia, Hora: hora})
	}

	log.Printf("readData time: %v", time.Since(now))
}
