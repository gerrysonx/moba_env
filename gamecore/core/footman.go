package core

import (
	"math/rand"
	"time"

	"github.com/ungerik/go3d/vec3"
)

type Footman struct {
	BaseInfo
	TargetLaneMilestone int32
	Milestones          []vec3.T
}

var footman_template Footman

func (footman *Footman) Init(a ...interface{}) BaseFunc {
	wanted_camp := a[0].(int32)

	if footman_template.health == 0 {
		// Not initialized yet, initialize first, load config from json file
		footman_template.InitFromJson("./cfg/footman.json")
	} else {
		// Already initialized, we can copy
	}

	*footman = footman_template
	InitWithCamp(footman, wanted_camp)
	footman.Milestones = a[1].([]vec3.T)
	footman.TargetLaneMilestone = 0
	footman.position = footman.Milestones[0]
	footman.position[0] += rand.Float32() * 50
	footman.position[1] += rand.Float32() * 50
	return footman
}

func (footman *Footman) Tick(gap_time float64) {
	game := &GameInst

	now := time.Now()
	// fmt.Printf("Footman is ticking, %v gap_time is:%v\n", now, gap_time)
	// Footman's logic is very simple
	// If there is any enemy within view-sight
	// Start attack
	// Else move towards target direction
	pos := footman.Position()
	isEnemyNearby, enemy := CheckEnemyNearby(footman.Camp(), footman.AttackRange(), &pos)
	if isEnemyNearby {
		// Check if time to make hurt
		now_seconds := float64(now.UnixNano()) / 1e9
		if (footman.LastAttackTime() + footman.AttackFreq()) < float64(now_seconds) {
			// Make damage
			dir_a := enemy.Position()
			dir_b := footman.Position()
			dir := vec3.Sub(&dir_a, &dir_b)
			bullet := new(Bullet).Init(footman.Camp(), footman.Position(), dir, footman.Damage())
			game.AddUnits = append(game.AddUnits, bullet)

			footman.SetLastAttackTime(now_seconds)
		}
	} else {
		// March towards target direction
		dir := footman.Direction()
		dir = dir.Scaled(float32(gap_time))
		dir = dir.Scaled(float32(footman.speed))
		newPos := vec3.Add(&pos, &dir)
		footman.SetPosition(newPos)

		// Check milestone distance
		dist := vec3.Distance(&newPos, &footman.Milestones[footman.TargetLaneMilestone])
		if dist < 20 {
			if int(footman.TargetLaneMilestone) == (len(footman.Milestones) - 1) {
				// Already the last milestone
			} else {
				footman.TargetLaneMilestone += 1
			}
		}
		// Calculate new direction
		footman.direction = footman.Milestones[footman.TargetLaneMilestone]
		footman.direction.Sub(&newPos)
		footman.direction.Normalize()
	}
}
