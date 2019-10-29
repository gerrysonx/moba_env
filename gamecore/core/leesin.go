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

var leesin_template LeeSin

func (hero *LeeSin) Init(a ...interface{}) BaseFunc {

	if leesin_template.health == 0 {
		// Not initialized yet, initialize first, load config from json file
		leesin_template.InitFromJson("./cfg/leesin.json")
	} else {
		// Already initialized, we can copy
	}

	*hero = leesin_template
	return hero
}

func (hero *LeeSin) UseSkill(skill_idx uint8) {

	switch skill_idx {
	case 0:
		// Hit from distance

	case 1:
		// Arcame Shift
		// Alter position

	case 2:
		// Trueshot Barrage

	}

}
