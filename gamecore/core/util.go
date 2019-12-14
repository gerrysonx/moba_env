package core

import (
	"os"

	"github.com/ungerik/go3d/vec3"
)

var LogHandle *os.File

func LogStr(log string) {
	if LogHandle == nil {
		return
	}
	LogHandle.WriteString(log)
	LogHandle.WriteString("\n")
	LogHandle.Sync()
}

func LogBytes(log []byte) {
	if LogHandle == nil {
		return
	}
	LogHandle.Write(log)
	LogHandle.WriteString("\n")
	LogHandle.Sync()
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

func CheckUnitOnDir(position *vec3.T, dir *vec3.T) (bool, BaseFunc) {
	game := &GameInst

	for _, v := range game.BattleUnits {
		if _, ok := v.(*Bullet); ok {
			continue
		}

		unit_pos := v.Position()
		unit_dir := unit_pos.Sub(position)
		angle := vec3.Angle(unit_dir, dir)
		if angle < float32(0.3) {
			return true, v
		}
	}

	return false, nil
}

func GetEnemyClearDir(my_camp int32, position *vec3.T) (bool, vec3.T) {
	game := &GameInst
	total_dir := vec3.T{0, 0, 0}

	for _, v := range game.BattleUnits {
		if _, ok := v.(*Bullet); ok {
			continue
		}

		if v.Camp() != my_camp {
			unit_pos := v.Position()
			unit_dir := unit_pos.Sub(position)
			unit_dir.Normalize()
			total_dir = *total_dir.Add(unit_dir)
		}
	}

	total_dir.Normalize()
	total_dir[0] = -total_dir[0]
	total_dir[1] = -total_dir[1]

	return false, total_dir
}

func CheckEnemyOnDirAngle(my_camp int32, position *vec3.T, dir *vec3.T, angle_threshold float32) (bool, BaseFunc) {
	game := &GameInst

	for _, v := range game.BattleUnits {
		if _, ok := v.(*Bullet); ok {
			continue
		}

		unit_pos := v.Position()
		unit_dir := unit_pos.Sub(position)
		angle := vec3.Angle(unit_dir, dir)
		if angle < float32(angle_threshold) && v.Camp() != my_camp {
			return true, v
		}
	}

	return false, nil
}

func CheckEnemyOnDir(my_camp int32, position *vec3.T, dir *vec3.T) (bool, BaseFunc) {
	game := &GameInst

	for _, v := range game.BattleUnits {
		if _, ok := v.(*Bullet); ok {
			continue
		}

		unit_pos := v.Position()
		unit_dir := unit_pos.Sub(position)
		angle := vec3.Angle(unit_dir, dir)
		if angle < float32(0.3) && v.Camp() != my_camp {
			return true, v
		}
	}

	return false, nil
}

func CheckEnemyOnDirWithinDist(my_camp int32, position *vec3.T, dir *vec3.T, dist_thres float32) (bool, BaseFunc) {
	game := &GameInst

	for _, v := range game.BattleUnits {
		if _, ok := v.(*Bullet); ok {
			continue
		}
		unit_pos := v.Position()
		dist := vec3.Distance(position, &unit_pos)
		unit_dir := unit_pos.Sub(position)
		angle := vec3.Angle(unit_dir, dir)
		if angle < float32(0.3) && v.Camp() != my_camp && dist < dist_thres {
			return true, v
		}
	}

	return false, nil
}

func CheckEnemyNearby(camp int32, radius float32, position *vec3.T) (bool, BaseFunc) {
	game := &GameInst
	dist := float32(0)
	min_dist := radius
	var min_dist_enemy BaseFunc
	min_dist_enemy = nil
	for _, v := range game.BattleUnits {
		if _, ok := v.(*Bullet); ok {
			continue
		}

		unit_pos := v.Position()
		dist = vec3.Distance(position, &unit_pos)
		if (dist < min_dist) && (v.Camp() != camp) && (v.Health() > 0) {
			min_dist = dist
			min_dist_enemy = v
		}
	}

	if min_dist_enemy != nil {
		return true, min_dist_enemy
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
	battle_unit.ClearAllBuff()
	battle_unit.ClearLastSkillUseTime()
}

func ConvertNum2Dir(action_code int) (dir vec3.T) {
	offset_x := float32(0)
	offset_y := float32(0)
	const_val := float32(20)

	switch action_code {
	case 0: // do nothing
		offset_x = float32(const_val)
		offset_y = float32(const_val)
	case 1:
		offset_x = float32(-const_val)
		offset_y = float32(-const_val)
	case 2:
		offset_x = float32(0)
		offset_y = float32(-const_val)
	case 3:
		offset_x = float32(const_val)
		offset_y = float32(-const_val)
	case 4:
		offset_x = float32(-const_val)
		offset_y = float32(0)
	case 5:
		offset_x = float32(const_val)
		offset_y = float32(0)
	case 6:
		offset_x = float32(-const_val)
		offset_y = float32(const_val)
	case 7:
		offset_x = float32(0)
		offset_y = float32(const_val)
	}

	dir[0] = offset_x
	dir[1] = offset_y
	return dir
}
