package core

import (
	"encoding/json"
	"fmt"
	"log"
	"math/rand"
	"os"
	"path/filepath"
	"time"

	"github.com/ungerik/go3d/vec3"
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

type GameVarPlayerTrainState struct {
	SelfHeroCount      int32
	SelfHeroPosX       []float32
	SelfHeroPosY       []float32
	SelfHeroHealth     []float32
	SelfHeroHealthFull []float32

	OppoHeroCount      int32
	OppoHeroPosX       []float32
	OppoHeroPosY       []float32
	OppoHeroHealth     []float32
	OppoHeroHealthFull []float32

	SlowBuffState      float32
	SlowBuffRemainTime float32

	SelfWin int32
}

type Game struct {
	SceneId     int
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
	var_player_train_state   GameVarPlayerTrainState
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
	SelfTowers     []float32
	OppoTowers     []float32
	SelfCreeps     []float32
	OppoCreeps     []float32
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
		const battle_field_width = 1000
		const battle_field_height = 1000

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

		const tower_attributes_count = 3
		self_tower_count := len(testconfig.SelfTowers) / tower_attributes_count
		oppo_tower_count := len(testconfig.OppoTowers) / tower_attributes_count
		for idx := 0; idx < self_tower_count; idx += 1 {
			tower_x := testconfig.SelfTowers[idx*tower_attributes_count+0] * battle_field_width
			tower_y := testconfig.SelfTowers[idx*tower_attributes_count+1] * battle_field_height
			tower_id := int32(testconfig.SelfTowers[idx*tower_attributes_count+2])

			self_hero := HeroMgrInst.Spawn(tower_id, int32(0), float32(tower_x), float32(tower_y))
			game.BattleUnits = append(game.BattleUnits, self_hero)
		}

		for idx := 0; idx < oppo_tower_count; idx += 1 {
			tower_x := testconfig.OppoTowers[idx*tower_attributes_count+0] * battle_field_width
			tower_y := testconfig.OppoTowers[idx*tower_attributes_count+1] * battle_field_height
			tower_id := int32(testconfig.OppoTowers[idx*tower_attributes_count+2])

			self_hero := HeroMgrInst.Spawn(tower_id, int32(1), float32(tower_x), float32(tower_y))
			game.BattleUnits = append(game.BattleUnits, self_hero)
		}

		const creep_attributes_count = 7
		self_creep_count := len(testconfig.SelfCreeps) / creep_attributes_count
		for idx := 0; idx < self_creep_count; idx += 1 {
			creep_spawn_x := testconfig.SelfCreeps[idx*creep_attributes_count+0] * battle_field_width
			creep_spawn_y := testconfig.SelfCreeps[idx*creep_attributes_count+1] * battle_field_height
			creep_target_x := testconfig.SelfCreeps[idx*creep_attributes_count+2] * battle_field_width
			creep_target_y := testconfig.SelfCreeps[idx*creep_attributes_count+3] * battle_field_height
			creep_spawn_freq := float64(testconfig.SelfCreeps[idx*creep_attributes_count+4])
			creep_id := int32(testconfig.SelfCreeps[idx*creep_attributes_count+5])
			creep_count := int32(testconfig.SelfCreeps[idx*creep_attributes_count+6])

			creep_spawn_mgr := new(CreepMgr)
			creep_spawn_mgr.SetAttackFreq(creep_spawn_freq)
			creep_spawn_mgr.SetPosition(vec3.T{creep_spawn_x, creep_spawn_y})
			creep_spawn_mgr.SetDirection(vec3.T{creep_target_x, creep_target_y})
			creep_spawn_mgr.SetCamp(int32(0))
			creep_spawn_mgr.SetHealth(1)
			creep_spawn_mgr.SetId(0)
			creep_spawn_mgr.CreepId = creep_id
			creep_spawn_mgr.CreepCount = creep_count
			game.BattleUnits = append(game.BattleUnits, creep_spawn_mgr)
		}

		oppo_creep_count := len(testconfig.OppoCreeps) / creep_attributes_count
		for idx := 0; idx < oppo_creep_count; idx += 1 {
			creep_spawn_x := testconfig.OppoCreeps[idx*creep_attributes_count+0] * battle_field_width
			creep_spawn_y := testconfig.OppoCreeps[idx*creep_attributes_count+1] * battle_field_height
			creep_target_x := testconfig.OppoCreeps[idx*creep_attributes_count+2] * battle_field_width
			creep_target_y := testconfig.OppoCreeps[idx*creep_attributes_count+3] * battle_field_height
			creep_spawn_freq := float64(testconfig.OppoCreeps[idx*creep_attributes_count+4])
			creep_id := int32(testconfig.OppoCreeps[idx*creep_attributes_count+5])
			creep_count := int32(testconfig.OppoCreeps[idx*creep_attributes_count+6])

			creep_spawn_mgr := new(CreepMgr)
			creep_spawn_mgr.SetAttackFreq(creep_spawn_freq)
			creep_spawn_mgr.SetPosition(vec3.T{creep_spawn_x, creep_spawn_y})
			creep_spawn_mgr.SetDirection(vec3.T{creep_target_x, creep_target_y})
			creep_spawn_mgr.SetCamp(int32(1))
			creep_spawn_mgr.SetHealth(1)
			creep_spawn_mgr.SetId(0)
			creep_spawn_mgr.CreepId = creep_id
			creep_spawn_mgr.CreepCount = creep_count
			game.BattleUnits = append(game.BattleUnits, creep_spawn_mgr)
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

	game.LoadTestCase(fmt.Sprintf("./cfg/maps/%d.json", game.SceneId))

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
		oppo_unit = game.SelfHeroes[0].(BaseFunc)
		self_unit = game.OppoHeroes[0].(BaseFunc)
	} else {
		self_unit = game.SelfHeroes[0].(BaseFunc)
		oppo_unit = game.OppoHeroes[0].(BaseFunc)
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

func (game *Game) ClearGameStateData() {
	game.var_player_train_state.SelfHeroCount = 0
	game.var_player_train_state.SelfHeroPosX = game.var_player_train_state.SelfHeroPosX[:0]
	game.var_player_train_state.SelfHeroPosY = game.var_player_train_state.SelfHeroPosY[:0]
	game.var_player_train_state.SelfHeroHealth = game.var_player_train_state.SelfHeroHealth[:0]
	game.var_player_train_state.SelfHeroHealthFull = game.var_player_train_state.SelfHeroHealthFull[:0]

	game.var_player_train_state.OppoHeroCount = 0
	game.var_player_train_state.OppoHeroPosX = game.var_player_train_state.OppoHeroPosX[:0]
	game.var_player_train_state.OppoHeroPosY = game.var_player_train_state.OppoHeroPosY[:0]
	game.var_player_train_state.OppoHeroHealth = game.var_player_train_state.OppoHeroHealth[:0]
	game.var_player_train_state.OppoHeroHealthFull = game.var_player_train_state.OppoHeroHealthFull[:0]
}

func (game *Game) DumpVarPlayerGameState() []byte {
	game.ClearGameStateData()
	all_oppo_heroes_dead := true
	all_self_heroes_dead := true

	game.var_player_train_state.SelfWin = 0
	game.var_player_train_state.SelfHeroCount = int32(len(game.SelfHeroes))
	for i := 0; i < int(game.var_player_train_state.SelfHeroCount); i += 1 {
		self_hero_unit := game.SelfHeroes[i].(BaseFunc)
		game.var_player_train_state.SelfHeroPosX = append(game.var_player_train_state.SelfHeroPosX, self_hero_unit.Position()[0])
		game.var_player_train_state.SelfHeroPosY = append(game.var_player_train_state.SelfHeroPosY, self_hero_unit.Position()[1])
		game.var_player_train_state.SelfHeroHealth = append(game.var_player_train_state.SelfHeroHealth, self_hero_unit.Health())
		game.var_player_train_state.SelfHeroHealthFull = append(game.var_player_train_state.SelfHeroHealthFull, self_hero_unit.MaxHealth())
		if self_hero_unit.Health() > float32(0.0) {
			all_self_heroes_dead = false
		}
	}

	game.var_player_train_state.OppoHeroCount = int32(len(game.OppoHeroes))
	for i := 0; i < int(game.var_player_train_state.OppoHeroCount); i += 1 {
		oppo_hero_unit := game.OppoHeroes[i].(BaseFunc)
		game.var_player_train_state.OppoHeroPosX = append(game.var_player_train_state.OppoHeroPosX, oppo_hero_unit.Position()[0])
		game.var_player_train_state.OppoHeroPosY = append(game.var_player_train_state.OppoHeroPosY, oppo_hero_unit.Position()[1])
		game.var_player_train_state.OppoHeroHealth = append(game.var_player_train_state.OppoHeroHealth, oppo_hero_unit.Health())
		game.var_player_train_state.OppoHeroHealthFull = append(game.var_player_train_state.OppoHeroHealthFull, oppo_hero_unit.MaxHealth())
		if oppo_hero_unit.Health() > float32(0.0) {
			all_oppo_heroes_dead = false
		}
	}

	if all_oppo_heroes_dead || all_self_heroes_dead {
		if all_oppo_heroes_dead {
			game.var_player_train_state.SelfWin = 1
		} else {
			game.var_player_train_state.SelfWin = -1
		}
	}

	e, err := json.Marshal(game.var_player_train_state)
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

func (game *Game) HandleMultiPlayerAction(player_idx int, action_code_0 int, action_code_1 int, action_code_2 int) {
	if action_code_0 == 9 {
		game.Init()
		return
	}

	battle_unit := game.SelfHeroes[player_idx].(BaseFunc)
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

// Global game object
var GameInst Game
