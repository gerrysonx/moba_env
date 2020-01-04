package core

import (
	"encoding/json"
	"fmt"
	"log"
	"math/rand"
	"os"
	"path/filepath"
	"time"
)

type GameTrainState struct {
	SelfHeroPosX   float32
	SelfHeroPosY   float32
	SelfHeroHealth float32
	OppoHeroPosX   float32
	OppoHeroPosY   float32
	OppoHeroHealth float32

	SlowBuffState      float32
	SlowBuffRemainTime float32

	SelfWin int32
}

type GameMultiPlayerTrainState struct {
	SelfHero0PosX       float32
	SelfHero0PosY       float32
	SelfHero0Health     float32
	SelfHero0HealthFull float32

	SelfHero1PosX       float32
	SelfHero1PosY       float32
	SelfHero1Health     float32
	SelfHero1HealthFull float32

	OppoHeroPosX       float32
	OppoHeroPosY       float32
	OppoHeroHealth     float32
	OppoHeroHealthFull float32

	SlowBuffState      float32
	SlowBuffRemainTime float32

	SelfWin int32
}

type Game struct {
	CurrentTime float64
	LogicTime   float64
	BattleUnits []BaseFunc
	AddUnits    []BaseFunc
	BattleField *BattleField

	SelfHeroes      []HeroFunc
	OppoHeroes      []HeroFunc
	ManualCtrlEnemy bool
	MultiPlayer     bool

	train_state              GameTrainState
	multi_player_train_state GameMultiPlayerTrainState
	skill_targets            []SkillTarget
	skill_targets_add        []SkillTarget
}

type TestConfig struct {
	Restricted_x   float32
	Restricted_y   float32
	Restricted_w   float32
	Restricted_h   float32
	OppoHeroes     []int32
	SelfHeroes     []int32
	SpawnAreaWidth float32
}

func (game *Game) LoadTestCase(test_cfg_name string) {
	root_dir, err := filepath.Abs(filepath.Dir(os.Args[0]))
	if err != nil {
		log.Fatal(err)
	}

	full_cfg_name := fmt.Sprintf("%s/%s", root_dir, test_cfg_name)

	file_handle, err := os.Open(full_cfg_name)
	if err != nil {
		return
	}

	buffer := make([]byte, 10000)
	read_count, err := file_handle.Read(buffer)
	if err != nil {
		return
	}
	buffer = buffer[:read_count]
	var testconfig TestConfig
	now := time.Now()
	rand.Seed(now.UnixNano())

	if err = json.Unmarshal(buffer, &testconfig); err == nil {
		game.BattleField = &BattleField{Restricted_x: testconfig.Restricted_x,
			Restricted_y: testconfig.Restricted_y,
			Restricted_w: testconfig.Restricted_w,
			Restricted_h: testconfig.Restricted_h}

		born_area_side_width := int(testconfig.SpawnAreaWidth)

		// Self heroes count
		self_heroes_count := len(testconfig.SelfHeroes)
		oppo_heroes_count := len(testconfig.OppoHeroes)

		for idx := 0; idx < self_heroes_count; idx += 1 {
			start_pos := (1000 - born_area_side_width) / 2
			rand_num_1 := rand.Intn(born_area_side_width)
			rand_num_2 := rand.Intn(born_area_side_width)
			self_hero := HeroMgrInst.Spawn(testconfig.SelfHeroes[idx], int32(0), float32(start_pos+rand_num_1), float32(start_pos+rand_num_2))
			game.BattleUnits = append(game.BattleUnits, self_hero)
			game.SelfHeroes = append(game.SelfHeroes, self_hero.(HeroFunc))
		}

		// Oppo heroes count
		for idx := 0; idx < oppo_heroes_count; idx += 1 {
			start_pos := (1000 - born_area_side_width) / 2
			rand_num_1 := rand.Intn(born_area_side_width)
			rand_num_2 := rand.Intn(born_area_side_width)
			oppo_hero := HeroMgrInst.Spawn(testconfig.OppoHeroes[idx], int32(1), float32(start_pos+rand_num_1), float32(start_pos+rand_num_2))
			game.BattleUnits = append(game.BattleUnits, oppo_hero)
			game.OppoHeroes = append(game.OppoHeroes, oppo_hero.(HeroFunc))
		}

	} else {
		fmt.Println("Error is:", err)
	}
}

func (game *Game) Init() {
	now := time.Now()
	game.LogicTime = float64(now.UnixNano()) / 1e9

	// map_units := game.BattleField.LoadMap("./map/3_corridors.png")

	game.BattleUnits = []BaseFunc{}
	game.AddUnits = []BaseFunc{}
	game.skill_targets = []SkillTarget{}
	game.skill_targets_add = []SkillTarget{}
	game.SelfHeroes = []HeroFunc{}
	game.OppoHeroes = []HeroFunc{}
	game.multi_player_train_state.SelfHero0Health = 0
	game.multi_player_train_state.SelfHero1Health = 0
	game.multi_player_train_state.OppoHeroHealth = 0

	game.LoadTestCase("./cfg/maps/0.json")

	LogStr("Game Inited.")
}

func (game *Game) AddTarget(target SkillTarget) {
	game.skill_targets_add = append(game.skill_targets_add, target)
}

func (game *Game) HandleInput() {

}

func (game *Game) HandleCallback() {

}

func (game *Game) GetGameState(reverse bool) []float32 {
	var self_unit BaseFunc
	var oppo_unit BaseFunc

	if reverse {
		oppo_unit = game.DefaultHero.(BaseFunc)
		self_unit = game.OppoHero.(BaseFunc)
	} else {
		self_unit = game.DefaultHero.(BaseFunc)
		oppo_unit = game.OppoHero.(BaseFunc)
	}

	game_state := make([]float32, 8)
	game_state[0] = self_unit.Position()[0]/1000.0 - 0.5
	game_state[1] = self_unit.Position()[1]/1000.0 - 0.5
	game_state[2] = self_unit.Health()/self_unit.MaxHealth() - 0.5

	game_state[3] = oppo_unit.Position()[0]/1000.0 - 0.5
	game_state[4] = oppo_unit.Position()[1]/1000.0 - 0.5
	game_state[5] = oppo_unit.Health()/oppo_unit.MaxHealth() - 0.5

	slow_buff_state := 0.0
	slow_buff_remain_time_ratio := 0.0

	slow_buff := oppo_unit.GetBuff(BuffSpeedSlow)
	if slow_buff != nil {
		slow_buff_state = 1.0
		slow_buff_remain_time_ratio = (slow_buff.base.Life + slow_buff.addTime - game.LogicTime) / slow_buff.base.Life
	}
	game_state[6] = float32(slow_buff_state)
	game_state[7] = float32(slow_buff_remain_time_ratio)
	return game_state
}

func (game *Game) DumpMultiPlayerGameState() []byte {
	//fmt.Printf("->DumpGameState, len(game.BattleUnits) is:%d, logictime:%f\n", len(game.BattleUnits), game.LogicTime)
	self_hero_0_unit := game.DefaultHeroes[0].(BaseFunc)
	self_hero_1_unit := game.DefaultHeroes[1].(BaseFunc)
	oppo_unit := game.OppoHero.(BaseFunc)
	// LogStr(fmt.Sprintf("DumpMultiPlayerGameState, unit 1 health:%v, unit 2 health:%v, oppo health:%v", self_hero_0_unit.Health(), self_hero_1_unit.Health(), oppo_unit.Health()))

	if (self_hero_0_unit.Health() > 0 || self_hero_1_unit.Health() > 0) && oppo_unit.Health() > 0 {
		game.multi_player_train_state.SelfWin = 0
		game.multi_player_train_state.SelfHero0PosX = self_hero_0_unit.Position()[0]
		game.multi_player_train_state.SelfHero0PosY = self_hero_0_unit.Position()[1]
		game.multi_player_train_state.SelfHero0Health = self_hero_0_unit.Health()
		game.multi_player_train_state.SelfHero0HealthFull = self_hero_0_unit.MaxHealth()

		game.multi_player_train_state.SelfHero1PosX = self_hero_1_unit.Position()[0]
		game.multi_player_train_state.SelfHero1PosY = self_hero_1_unit.Position()[1]
		game.multi_player_train_state.SelfHero1Health = self_hero_1_unit.Health()
		game.multi_player_train_state.SelfHero1HealthFull = self_hero_1_unit.MaxHealth()

		game.multi_player_train_state.OppoHeroPosX = oppo_unit.Position()[0]
		game.multi_player_train_state.OppoHeroPosY = oppo_unit.Position()[1]
		game.multi_player_train_state.OppoHeroHealth = oppo_unit.Health()
		game.multi_player_train_state.OppoHeroHealthFull = oppo_unit.MaxHealth()

		slow_buff_state := 0.0
		slow_buff := oppo_unit.GetBuff(BuffSpeedSlow)
		slow_buff_remain_time_ratio := 0.0
		if slow_buff != nil {
			slow_buff_state = 1.0
			if slow_buff.base.Life == 0 {
				// slow_buff_remain_time_ratio = 0.777
				// panic("slow_buff.base.Life shouldn't be 0")
				LogStr(fmt.Sprintf("slow_buff.base.Life == 0 when dumping game state, buff_id:%v, val1:%v", slow_buff.base.Id, slow_buff.base.Val1))
			} else {
				slow_buff_remain_time_ratio = (slow_buff.base.Life + slow_buff.addTime - game.LogicTime) / slow_buff.base.Life
			}
		}
		game.multi_player_train_state.SlowBuffState = float32(slow_buff_state)
		game.multi_player_train_state.SlowBuffRemainTime = float32(slow_buff_remain_time_ratio)

	} else {
		if oppo_unit.Health() <= 0 {
			game.multi_player_train_state.SelfWin = 1
		} else {
			game.multi_player_train_state.SelfWin = -1
		}
	}

	e, err := json.Marshal(game.multi_player_train_state)
	if err != nil {
		return []byte(fmt.Sprintf("Marshal train_state failed.%v", game.multi_player_train_state))

	}

	return e
}

func (game *Game) Tick(gap_time float64) {
	//fmt.Printf("->gameTick, len(game.BattleUnits) is:%d, logictime:%f\n", len(game.BattleUnits), game.LogicTime)
	game.LogicTime += gap_time
	now := time.Now()
	game.CurrentTime = float64(now.UnixNano()) / 1e9
	// fmt.Printf("Game is ticking, %v-%v, gap time is:%v\n", now.Second(), game.CurrentTime, gap_time)
	var temp_arr []BaseFunc
	for _, v := range game.BattleUnits {
		if v.Health() > 0 {
			v.Tick(gap_time)
			temp_arr = append(temp_arr, v)
		} else {
			// Remove v from array, or just leave it there
		}
	}

	for _, v := range game.AddUnits {
		temp_arr = append(temp_arr, v)
	}
	game.AddUnits = []BaseFunc{}
	game.BattleUnits = temp_arr

	// Handle skill targets callbacks
	var temp_arr2 []SkillTarget
	for _, v := range game.skill_targets {
		if v.trigger_time < game.LogicTime {
			v.callback(&v)
		} else {
			temp_arr2 = append(temp_arr2, v)
		}
	}

	temp_arr2 = append(temp_arr2, game.skill_targets_add...)
	game.skill_targets_add = []SkillTarget{}
	game.skill_targets = temp_arr2

}

func (game *Game) HandleMultiAction(action_code_0 int, action_code_1 int, action_code_2 int) {
	battle_unit := game.DefaultHero.(BaseFunc)
	cur_pos := battle_unit.Position()

	offset_x := float32(0)
	offset_y := float32(0)

	switch action_code_0 {
	case 0: // do nothing
		// Remain the same position
		game.DefaultHero.SetTargetPos(cur_pos[0], cur_pos[1])
	case 1:
		// move
		dir := ConvertNum2Dir(action_code_1)
		offset_x = dir[0]
		offset_y = dir[1]
		target_pos_x := float32(cur_pos[0] + offset_x)
		target_pos_y := float32(cur_pos[1] + offset_y)
		// Check self position
		// game.DefaultHero.SetTargetPos(target_pos_x, target_pos_y)
		is_target_within := game.BattleField.Within(target_pos_x, target_pos_y)
		if is_target_within {
			game.DefaultHero.SetTargetPos(target_pos_x, target_pos_y)
		}

	case 2:
		// normal attack
		// Remain the same position
		game.DefaultHero.SetTargetPos(cur_pos[0], cur_pos[1])
	case 3:
		// skill 1
		dir := ConvertNum2Dir(action_code_2)
		offset_x = dir[0]
		offset_y = dir[1]
		game.DefaultHero.UseSkill(0, offset_x, offset_y)
		// Set skill target
	case 4:
		// skill 2
		dir := ConvertNum2Dir(action_code_2)
		offset_x = dir[0]
		offset_y = dir[1]
		game.DefaultHero.UseSkill(1, offset_x, offset_y)
	case 5:
		// skill 3
		dir := ConvertNum2Dir(action_code_2)
		offset_x = dir[0]
		offset_y = dir[1]
		game.DefaultHero.UseSkill(2, offset_x, offset_y)
	case 6:
		// extra skill
		dir := ConvertNum2Dir(action_code_2)
		offset_x = dir[0]
		offset_y = dir[1]
		game.DefaultHero.UseSkill(3, offset_x, offset_y)

	case 9:
		game.Init()
	}

}

func (game *Game) HandleMultiPlayerAction(player_idx int, action_code_0 int, action_code_1 int, action_code_2 int) {
	if action_code_0 == 9 {
		game.Init()
		return
	}

	battle_unit := game.DefaultHeroes[player_idx].(BaseFunc)
	cur_pos := battle_unit.Position()
	if battle_unit.Health() <= 0 {
		return
	}

	offset_x := float32(0)
	offset_y := float32(0)

	switch action_code_0 {
	case 0: // do nothing
		// Remain the same position
		battle_unit.(HeroFunc).SetTargetPos(cur_pos[0], cur_pos[1])
	case 1:
		// move
		dir := ConvertNum2Dir(action_code_1)
		offset_x = dir[0]
		offset_y = dir[1]
		target_pos_x := float32(cur_pos[0] + offset_x)
		target_pos_y := float32(cur_pos[1] + offset_y)
		// Check self position
		// game.DefaultHero.SetTargetPos(target_pos_x, target_pos_y)
		is_target_within := game.BattleField.Within(target_pos_x, target_pos_y)
		if is_target_within {
			battle_unit.(HeroFunc).SetTargetPos(target_pos_x, target_pos_y)
		}

	case 2:
		// normal attack
		// Remain the same position
		battle_unit.(HeroFunc).SetTargetPos(cur_pos[0], cur_pos[1])
	case 3:
		// skill 1
		dir := ConvertNum2Dir(action_code_2)
		offset_x = dir[0]
		offset_y = dir[1]
		battle_unit.(HeroFunc).UseSkill(0, offset_x, offset_y)
		// Set skill target
	case 4:
		// skill 2
		dir := ConvertNum2Dir(action_code_2)
		offset_x = dir[0]
		offset_y = dir[1]
		battle_unit.(HeroFunc).UseSkill(1, offset_x, offset_y)
	case 5:
		// skill 3
		dir := ConvertNum2Dir(action_code_2)
		offset_x = dir[0]
		offset_y = dir[1]
		battle_unit.(HeroFunc).UseSkill(2, offset_x, offset_y)
	case 6:
		// extra skill
		dir := ConvertNum2Dir(action_code_2)
		offset_x = dir[0]
		offset_y = dir[1]
		battle_unit.(HeroFunc).UseSkill(3, offset_x, offset_y)

	}
}

func (game *Game) HandleAction(action_code int) {
	battle_unit := game.DefaultHero.(BaseFunc)
	cur_pos := battle_unit.Position()

	offset_x := float32(0)
	offset_y := float32(0)
	const_val := float32(100)

	switch action_code {
	case 0: // do nothing
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
	case 8:
		offset_x = float32(const_val)
		offset_y = float32(const_val)
	case 9:
		game.Init()
	}

	game.DefaultHero.SetTargetPos(float32(cur_pos[0]+offset_x), float32(cur_pos[1]+offset_y))
}

// Global game object
var GameInst Game
