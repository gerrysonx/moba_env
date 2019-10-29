package core

import (
	"fmt"
	"time"
)

type Ancient struct {
	BaseInfo
}

func (hero * Ancient) Tick(gap_time float64) {
	now := time.Now()
	fmt.Printf("Ancient is ticking, %v gap_time is:%v\n", now, gap_time)
}

func (ancient * Ancient) Init(a ...interface{}) BaseFunc {
	now := time.Now()
	fmt.Printf("Ancient is ticking, %v\n", now)
	return ancient
}
