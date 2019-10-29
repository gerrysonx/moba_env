package core

type Monster struct {
	BaseInfo
}

func (monster *Monster) Tick(gap_time float64) {
	// now := time.Now()
	// fmt.Printf("Monster is ticking, %v gap_time is:%v\n", now, gap_time)
}

var monster_template Monster

func (monster *Monster) Init(a ...interface{}) BaseFunc {

	if monster_template.health == 0 {
		// Not initialized yet, initialize first, load config from json file
		monster_template.InitFromJson("./cfg/monster.json")
	} else {
		// Already initialized, we can copy
	}

	tmp_pos := monster.position
	tmp_camp := monster.camp
	*monster = monster_template
	monster.position = tmp_pos
	monster.camp = tmp_camp
	return monster
}
