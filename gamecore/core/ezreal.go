package core

import (
	"github.com/ungerik/go3d/vec3"
)

type Ezreal struct {
	Hero
}

func (hero *Ezreal) Tick(gap_time float64) {
	//	fmt.Printf("Ezreal is ticking, %v gap_time is:%v\n", now, gap_time)
	game := &GameInst
	// fmt.Printf("Hero is ticking, %v gap_time is:%v\n", now, gap_time)
	// Hero's logic is very simple
	// If there is any enemy within view-sight
	// Start attack
	// Else move towards target direction
	pos := hero.Position()
	// Check milestone distance
	targetPos := hero.TargetPos()
	dist := vec3.Distance(&pos, &targetPos)
	pos_ready := false
	if dist < 20 {
		// Already the last milestone
		pos_ready = true
	}

	isEnemyNearby, enemy := CheckEnemyNearby(hero.Camp(), hero.AttackRange(), &pos)
	if isEnemyNearby && pos_ready {
		// Check if time to make hurt
		now_seconds := game.LogicTime
		if (hero.LastAttackTime() + hero.AttackFreq()) < float64(now_seconds) {
			// Make damage
			dir_a := enemy.Position()
			dir_b := hero.Position()
			dir := vec3.Sub(&dir_a, &dir_b)
			bullet := new(Bullet).Init(hero.Camp(), hero.Position(), dir, hero.Damage())
			game.AddUnits = append(game.AddUnits, bullet)

			hero.SetLastAttackTime(now_seconds)
		}
	} else {

		if pos_ready {
			// Do nothing

		} else {
			// March towards target direction
			dir := hero.Direction()
			dir = dir.Scaled(float32(gap_time))
			dir = dir.Scaled(float32(hero.speed))
			newPos := vec3.Add(&pos, &dir)
			hero.SetPosition(newPos)

			// Calculate new direction
			hero.direction = targetPos
			hero.direction.Sub(&newPos)
			hero.direction.Normalize()
		}
	}
}

var enzeal_template Ezreal

func (hero *Ezreal) Init(a ...interface{}) BaseFunc {
	wanted_camp := a[0].(int32)
	if enzeal_template.health == 0 {
		// Not initialized yet, initialize first, load config from json file
		enzeal_template.InitFromJson("./cfg/ezreal.json")
	} else {
		// Already initialized, we can copy
	}

	*hero = enzeal_template
	pos_x := a[1].(float32)
	pos_y := a[2].(float32)
	InitHeroWithCamp(hero, wanted_camp, pos_x, pos_y)

	return hero

}

func (hero *Ezreal) UseSkill(skill_idx uint8, a ...interface{}) {

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
