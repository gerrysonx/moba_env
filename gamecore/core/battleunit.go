package core

import (
	"encoding/json"
	"fmt"
	"log"
	"os"
	"path/filepath"

	"github.com/ungerik/go3d/vec3"
)

const (
	BuffSpeedSlow  = 0
	BuffHealth     = 1
	BuffLockSkill  = 2
	BuffAttackFreq = 3
)

type Buff struct {
	base    BuffConfig
	addTime float64
	oldVal  float32
}

// Buff config
type BuffConfig struct {
	Id   int32
	Type int32
	Life float64
	Val1 float32
	Val2 float32
	Val3 float32
}

var buff_config_map map[int32]BuffConfig

func InitBuffConfig(config_file_name string) {
	if 0 == len(buff_config_map) {
		buff_config_map = make(map[int32]BuffConfig)
	}

	file_handle, err := os.Open(config_file_name)
	if err != nil {
		return
	}

	buffer := make([]byte, 10000)
	read_count, err := file_handle.Read(buffer)
	if err != nil {
		return
	}
	buffer = buffer[:read_count]
	var jsoninfo []BuffConfig

	if err = json.Unmarshal(buffer, &jsoninfo); err == nil {
		for _, v := range jsoninfo {
			buff_config_map[v.Id] = v
		}
	} else {
		fmt.Println("Error is:", err)
	}

}

func NewBuff(buff_id int32, a ...interface{}) *Buff {
	new_buff := new(Buff)
	if 0 == len(buff_config_map) {
		root_dir, err := filepath.Abs(filepath.Dir(os.Args[0]))
		if err != nil {
			log.Fatal(err)
		}

		buff_config_map = make(map[int32]BuffConfig)
		InitBuffConfig(fmt.Sprintf("%s/cfg/skills.json", root_dir))
	}
	new_buff.base = buff_config_map[buff_id]
	buff_type := buff_config_map[buff_id].Type
	switch buff_type {
	case BuffSpeedSlow:
		// Old speed
		new_buff.oldVal = a[0].(float32)

	case BuffHealth:

	case BuffLockSkill:

	case BuffAttackFreq:
		// Old freq
		new_buff.oldVal = a[0].(float32)

	}

	return new_buff
}

type BaseInfo struct {
	type_id             int32
	velocity            vec3.T
	position            vec3.T
	direction           vec3.T
	view_range          float32 // In meters
	attack_range        float32 // In meters
	attack_freq         float64 // Time gap between two attacks, in seconds
	last_attack_time    float64
	last_skill_use_time [4]float64
	camp                int32 // 0 for camp-1, 1 for camp-2, 2 for neutral
	health              float32
	max_health          float32
	damage              float32
	speed               float32 // unit is meter per second
	buffs               map[int32]*Buff
}

type BaseFunc interface {
	Velocity() vec3.T

	Position() vec3.T
	SetPosition(newPos vec3.T)

	Direction() vec3.T
	SetDirection(direction vec3.T)

	ViewRange() float32   // In meters
	AttackRange() float32 // In meters
	AttackFreq() float64  // Time gap between two attacks, in seconds
	SetAttackFreq(new_freq float64) bool

	LastAttackTime() float64
	SetLastAttackTime(float64)

	LastSkillUseTime(skill_idx uint8) float64
	SetLastSkillUseTime(skill_idx uint8, last_use_time float64)
	ClearLastSkillUseTime()

	Camp() int32 // 0 for camp-1, 1 for camp-2, 2 for neutral
	SetCamp(int32)

	Health() float32
	SetHealth(float32)

	MaxHealth() float32
	SetMaxHealth(float32)

	Damage() float32
	SetDamage(damage float32)
	DealDamage(damage float32) bool
	Speed() float32
	SetSpeed(speed float32)
	DealSpeed(percentage float32) bool

	AddBuff(idx int32, buff *Buff)
	DelBuff(idx int32)
	GetBuff(idx int32) *Buff
	ClearAllBuff()

	Tick(gap_time float64)
	Init(a ...interface{}) BaseFunc
	GetId() int32
}

func (baseinfo *BaseInfo) Velocity() vec3.T {
	return baseinfo.velocity
}

func (baseinfo *BaseInfo) SetPosition(newPos vec3.T) {
	baseinfo.position = newPos
}

func (baseinfo *BaseInfo) Position() vec3.T {
	return baseinfo.position
}

func (baseinfo *BaseInfo) Direction() vec3.T {
	return baseinfo.direction
}

func (baseinfo *BaseInfo) SetDirection(direction vec3.T) {
	baseinfo.direction = direction
}

func (baseinfo *BaseInfo) AttackRange() float32 {
	return baseinfo.attack_range
}

func (baseinfo *BaseInfo) ViewRange() float32 {
	return baseinfo.view_range
}

func (baseinfo *BaseInfo) AttackFreq() float64 {
	return baseinfo.attack_freq
}

func (baseinfo *BaseInfo) SetAttackFreq(new_freq float64) bool {
	baseinfo.attack_freq = new_freq
	return true
}

func (baseinfo *BaseInfo) LastAttackTime() float64 {
	return baseinfo.last_attack_time
}

func (baseinfo *BaseInfo) SetLastAttackTime(last_attack_time float64) {
	baseinfo.last_attack_time = last_attack_time
}

func (baseinfo *BaseInfo) ClearLastSkillUseTime() {
	baseinfo.last_skill_use_time[0] = 0.0
	baseinfo.last_skill_use_time[1] = 0.0
	baseinfo.last_skill_use_time[2] = 0.0
	baseinfo.last_skill_use_time[3] = 0.0
}

func (baseinfo *BaseInfo) LastSkillUseTime(skill_idx uint8) float64 {
	return baseinfo.last_skill_use_time[skill_idx]
}

func (baseinfo *BaseInfo) SetLastSkillUseTime(skill_idx uint8, last_attack_time float64) {
	baseinfo.last_skill_use_time[skill_idx] = last_attack_time
}

func (baseinfo *BaseInfo) Camp() int32 {
	return baseinfo.camp
}

func (baseinfo *BaseInfo) SetCamp(camp int32) {
	baseinfo.camp = camp
}

func (baseinfo *BaseInfo) Health() float32 {
	return baseinfo.health
}

func (baseinfo *BaseInfo) SetHealth(newHealth float32) {
	baseinfo.health = newHealth
}

func (baseinfo *BaseInfo) MaxHealth() float32 {
	return baseinfo.max_health
}

func (baseinfo *BaseInfo) SetMaxHealth(max_health float32) {
	baseinfo.max_health = max_health
}

func (baseinfo *BaseInfo) DealDamage(damage float32) bool {
	baseinfo.health -= damage
	return true
}

func (baseinfo *BaseInfo) SetDamage(damage float32) {
	baseinfo.damage = damage
}

func (baseinfo *BaseInfo) Damage() float32 {
	return baseinfo.damage
}

func (baseinfo *BaseInfo) Speed() float32 {
	return baseinfo.speed
}

func (baseinfo *BaseInfo) SetSpeed(speed float32) {
	baseinfo.speed = speed
}

func (baseinfo *BaseInfo) DealSpeed(percentage float32) bool {
	baseinfo.speed *= percentage
	return true
}

func (baseinfo *BaseInfo) AddBuff(idx int32, buff *Buff) {
	game := &GameInst
	baseinfo.buffs[idx] = buff
	buff.addTime = game.LogicTime
}

func (baseinfo *BaseInfo) DelBuff(idx int32) {
	baseinfo.buffs[idx] = nil
}

func (baseinfo *BaseInfo) GetBuff(idx int32) *Buff {
	value, ok := baseinfo.buffs[idx]
	if ok {
		return value
	} else {
		return nil
	}
}

func (baseinfo *BaseInfo) GetId() int32 {
	return baseinfo.type_id

}

func (baseinfo *BaseInfo) ClearAllBuff() {
	baseinfo.buffs = map[int32]*Buff{}
}

type JsonInfo struct {
	AttackRange float32
	AttackFreq  float64
	Health      float32
	Damage      float32
	Stub1       float32
	Speed       float32
	ViewRange   float32
	Id          int32
}

func (baseinfo *BaseInfo) InitFromJson(cfg_name string) bool {
	// Need to initialize buffs
	dir, err := filepath.Abs(filepath.Dir(os.Args[0]))
	if err != nil {
		log.Fatal(err)
	}
	full_path := fmt.Sprintf("%s/%s", dir, cfg_name)

	baseinfo.buffs = make(map[int32]*Buff)

	file_handle, err := os.Open(full_path)
	if err != nil {
		return false
	}

	buffer := make([]byte, 200)
	read_count, err := file_handle.Read(buffer)
	if err != nil {
		return false
	}
	buffer = buffer[:read_count]
	var jsoninfo JsonInfo

	if err = json.Unmarshal(buffer, &jsoninfo); err == nil {
		baseinfo.attack_range = jsoninfo.AttackRange
		baseinfo.attack_freq = jsoninfo.AttackFreq
		baseinfo.health = jsoninfo.Health
		baseinfo.max_health = jsoninfo.Health
		baseinfo.damage = jsoninfo.Damage
		baseinfo.speed = jsoninfo.Speed
		baseinfo.view_range = jsoninfo.ViewRange
		baseinfo.type_id = jsoninfo.Id

	} else {
		file_handle.Close()
		return false
	}

	file_handle.Close()

	return true
}
