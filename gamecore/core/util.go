package core

import (
	"os"

	"github.com/ungerik/go3d/vec3"
)

func LogBytes(file_handle *os.File, log []byte) {
	file_handle.Write(log)
	file_handle.WriteString("\n")
	file_handle.Sync()

}

func remove(s []Buff, i int) []Buff {
	s[i] = s[len(s)-1]
	return s[:len(s)-1]
}

func CanAttackEnemy(unit BaseFunc, enemy_pos *vec3.T) bool {
	unit_pos := unit.Position()
	dist := vec3.Distance(enemy_pos, &unit_pos)
	if dist < unit.AttackRange() {
		return true
	}

	return false

}

func CheckEnemyNearby(camp int32, radius float32, position *vec3.T) (bool, BaseFunc) {

	game := GameInst
	dist := float32(0)
	for _, v := range game.BattleUnits {
		if _, ok := v.(*Bullet); ok {
			continue
		}

		unit_pos := v.Position()
		dist = vec3.Distance(position, &unit_pos)
		if (dist < radius) && (v.Camp() != camp) && (v.Health() > 0) {
			return true, v
		}
	}

	// fmt.Println("->CheckEnemyNearby")
	return false, nil
}

func InitWithCamp(battle_unit BaseFunc, camp int32) {
	const_num := float32(0.707106781)
	if camp == 0 {
		battle_unit.SetPosition(vec3.T{1, 999})
		battle_unit.SetCamp(0)
		battle_unit.SetDirection(vec3.T{const_num, -const_num})
	} else {
		battle_unit.SetPosition(vec3.T{999, 1})
		battle_unit.SetCamp(1)
		battle_unit.SetDirection(vec3.T{-const_num, const_num})
	}
}

func InitHeroWithCamp(hero_unit HeroFunc, camp int32, pos_x float32, pos_y float32) {
	battle_unit := hero_unit.(BaseFunc)
	InitWithCamp(battle_unit, camp)
	battle_unit.SetPosition(vec3.T{pos_x, pos_y})
	hero_unit.SetTargetPos(pos_x, pos_y)
	battle_unit.SetDirection(vec3.T{0, 0})
}
