package core

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"strconv"
	"strings"
)

type Skill struct {
	Name string
	Desc string
	Type int32
	Life float64
	Val1 float32
	Val2 float32
	Val3 float32
}

type SkillMgr struct {
	skills map[int32]Skill
}

func (skillmgr *SkillMgr) LoadCfg(id int32, config_file_name string) {

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
	var skill Skill

	if err = json.Unmarshal(buffer, &skill); err == nil {
		skillmgr.skills[id] = skill
	} else {
		fmt.Println("Error is:", err)
	}

}

func (skillmgr *SkillMgr) LoadCfgFolder(config_file_folder string) {
	// Load all skill configs under folder
	skillmgr.skills = make(map[int32]Skill)
	files, err := ioutil.ReadDir(config_file_folder)
	if err != nil {
		log.Fatal(err)
	}

	for _, f := range files {
		cfg_file_name := f.Name()
		segs := strings.Split(cfg_file_name, ".")
		id, _ := strconv.Atoi(segs[0])
		id32 := int32(id)
		fmt.Println(cfg_file_name)
		cfg_full_file_name := fmt.Sprintf("%s/%s", config_file_folder, cfg_file_name)
		skillmgr.LoadCfg(id32, cfg_full_file_name)
	}
}

var SkillMgrInst SkillMgr
