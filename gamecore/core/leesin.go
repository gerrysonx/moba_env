package core

import (
	"fmt"
	"time"
)

type LeeSin struct {
	Hero
}

func (hero *LeeSin) Tick(gap_time float64) {
	now := time.Now()
	fmt.Printf("Hero is ticking, %v gap_time is:%v\n", now, gap_time)
}
