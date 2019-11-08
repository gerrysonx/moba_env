package core

import (
	"fmt"
	"time"

	"github.com/ungerik/go3d/vec3"
)

type Vi struct {
	Hero
}

func (hero *Vi) Chase(pos_enemy vec3.T, gap_time float64) {
	pos := hero.Position()

	// Check milestone distance
	targetPos := pos_enemy
	dist := vec3.Distance(&pos, &targetPos)
	if dist < hero.AttackRange() {
		// Already the last milestone
		hero.direction[0] = 0
		hero.direction[1] = 0
	} else {
		// March towards enemy pos
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

func (hero *Vi) Escape(pos_enemy vec3.T, gap_time float64) {
	pos := hero.Position()

	// Escape from enemy pos
	// Check milestone distance
	targetPos := pos_enemy
	dist := vec3.Distance(&pos, &targetPos)
	if dist > 100 {
		// Already the last milestone
		hero.direction[0] = 0
		hero.direction[1] = 0
	} else {
		dir := hero.Direction()
		dir = dir.Scaled(float32(gap_time))
		dir = dir.Scaled(float32(hero.speed))
		newPos := vec3.Add(&pos, &dir)
		hero.SetPosition(newPos)

		// Calculate new direction
		hero.direction = targetPos
		hero.direction.Sub(&newPos)
		// Revert the target direction, so we let the  hero escape
		hero.direction[0] = -hero.direction[0]
		hero.direction[1] = -hero.direction[1]
		hero.direction.Normalize()
	}
}

func (hero *Vi) Tick(gap_time float64) {
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

var Vi_template Vi

func (hero *Vi) Init(a ...interface{}) BaseFunc {
	wanted_camp := a[0].(int32)

	if Vi_template.health == 0 {
		// Not initialized yet, initialize first, load config from json file
		Vi_template.InitFromJson("./cfg/vi.json")
	} else {
		// Already initialized, we can copy
	}

	*hero = Vi_template

	pos_x := a[1].(float32)
	pos_y := a[2].(float32)
	InitHeroWithCamp(hero, wanted_camp, pos_x, pos_y)
	return hero
}

func (hero *Vi) UseSkill(skill_idx uint8, a ...interface{}) {

	switch skill_idx {
	case 0:
		// Rocket Grab
		fmt.Println("Skill Rocket Grab is used.")
		// Clear skilltarget pos
		hero.SetSkillTargetPos(0, 0)

		go func(hero *Vi) {
			//	old_callback_fn := GameInst.window.SetMouseButtonCallback(GameInst.func_call_back)
			for {
				time.Sleep(time.Duration(0.5 * float64(time.Second)))
				// Wait for left button click to select position
				skill_target_pos := hero.SkillTargetPos()
				if skill_target_pos[0] != 0 || skill_target_pos[1] != 0 {
					// Use skill
					var target_v vec3.T
					target_v[0] = skill_target_pos[0]
					target_v[1] = skill_target_pos[1]
					AoEDamage(target_v, 30.0, hero.Camp(), 200.0)
					break
				}
			}
			//	GameInst.window.SetMouseButtonCallback(old_callback_fn)

		}(hero)
	case 1:
		// Overdrive
		arr := []BaseFunc{hero}
		AddSpeedBuff(arr, 0)
		fmt.Println("Skill Overdrive is used.")
	case 2:
		// Power Fist
		arr := []BaseFunc{hero}
		AddSpeedBuff(arr, 1)
		fmt.Println("Skill Power Fist is used.")
	case 3:
		// Static Field
		fmt.Println("Skill Static Field is used.")

	}

}
