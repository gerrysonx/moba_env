package core

import (
	"time"

	"github.com/ungerik/go3d/vec3"
)

type Tower struct {
	BaseInfo
}

func (tower *Tower) Tick(gap_time float64) {
	//return
	game := &GameInst
	now := time.Now()
	//	fmt.Printf("Tower is ticking, %v gap_time is:%v\n", now, gap_time)

	// Footman's logic is very simple
	// If there is any enemy within view-sight
	// Start attack
	// Else move towards target direction
	pos := tower.Position()
	isEnemyNearby, enemy := CheckEnemyNearby(tower.Camp(), tower.AttackRange(), &pos)
	if isEnemyNearby {
		// Check if time to make hurt
		now_seconds := float64(now.UnixNano()) / 1e9
		if (tower.LastAttackTime() + tower.AttackFreq()) < float64(now_seconds) {
			// Make damage
			dir_a := enemy.Position()
			dir_b := tower.Position()
			dir := vec3.Sub(&dir_a, &dir_b)
			bullet := new(Bullet).Init(tower.Camp(), tower.Position(), dir, tower.Damage())
			game.AddUnits = append(game.AddUnits, bullet)

			tower.SetLastAttackTime(now_seconds)

		}
	}
}

var tower_template Tower

func (tower *Tower) Init(a ...interface{}) BaseFunc {

	if tower_template.health == 0 {
		// Not initialized yet, initialize first, load config from json file
		tower_template.InitFromJson("./cfg/tower.json")
	} else {
		// Already initialized, we can copy
	}
	tmp_pos := tower.position
	tmp_camp := tower.camp
	*tower = tower_template
	tower.position = tmp_pos
	tower.camp = tmp_camp
	return tower
}
