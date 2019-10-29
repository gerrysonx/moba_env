package core

import (
	"fmt"
	"time"
)

type Vayne struct {
	Hero
}

func (hero *Vayne) Tick(gap_time float64) {
	now := time.Now()
	fmt.Printf("Hero is ticking, %v gap_time is:%v\n", now, gap_time)
}

var vayne_template Vayne

func (hero *Vayne) Init(a ...interface{}) BaseFunc {

	if vayne_template.health == 0 {
		// Not initialized yet, initialize first, load config from json file
		vayne_template.InitFromJson("./cfg/vayne.json")
	} else {
		// Already initialized, we can copy
	}

	*hero = vayne_template
	return hero
}

func (hero *Vayne) UseSkill(skill_idx uint8) {

	switch skill_idx {
	case 0:
		// Tumble

	case 1:
		// Silver Bolts

	case 2:
		// Condemn

	case 3:
		// Final Hour

	}

}
