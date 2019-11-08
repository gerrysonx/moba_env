package nn

import (
	"fmt"
	"log"
	"os"
	"path/filepath"
)

type Model struct {
	inference_pb    string
	bouquet_state   [][]float32
	formatted_state [][]float32
	model           NeuralNet
}

func (hero *Model) PrepareInput() {
	hero.bouquet_state = make([][]float32, 4)
	for i := 0; i < 4; i++ {
		hero.bouquet_state[i] = make([]float32, 6)
	}

	hero.formatted_state = make([][]float32, 6)
	for i := 0; i < 6; i++ {
		hero.formatted_state[i] = make([]float32, 4)
	}

	root_dir, err := filepath.Abs(filepath.Dir(os.Args[0]))
	full_path := fmt.Sprintf("%s/../../model_%s", root_dir, "338944")
	if err != nil {
		log.Fatal(err)
	}

	hero.model = NeuralNet{}
	hero.model.Load(full_path)
}

func (hero *Model) UpdateInput(one_state []float32) {
	hero.bouquet_state = append(hero.bouquet_state, one_state)
	hero.bouquet_state = hero.bouquet_state[1:5]
}

func (hero *Model) GetInput() [][][]float32 {
	// from bouquet states to formatted states
	for i := 0; i < 6; i++ {
		for j := 0; j < 4; j++ {
			hero.formatted_state[i][j] = hero.bouquet_state[j][i]
		}
	}

	formatted_input := [][][]float32{hero.formatted_state}
	return formatted_input
}

func (hero *Model) SampleAction(game_state []float32) uint8 {
	// Inference from nn
	hero.UpdateInput(game_state)
	input_val := hero.GetInput()
	predict := hero.model.Ref(input_val)
	max_value := float32(-10000.0)
	max_idx := -1
	for idx, val := range predict[0] {
		if val > max_value {
			max_idx = idx
			max_value = val
		}
	}

	return uint8(max_idx)

}
