package nn

import (
	"fmt"

	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/tensorflow/tensorflow/tensorflow/go/op"
)

type NeuralNet struct {
	Type             string
	inf_pb_path      string
	saved_model      *tf.SavedModel
	input_op         tf.Output
	output_policy_op tf.Output
	output_value_op  tf.Output
}

func (nn *NeuralNet) Test() {
	s := op.NewScope()
	c := op.Const(s, "Hello0 from tensorflow version "+tf.Version())
	graph, err := s.Finalize()
	if err != nil {
		panic(err)
	}

	// Execute the graph in a session
	sess, err := tf.NewSession(graph, nil)
	if err != nil {
		panic(err)
	}

	output, err := sess.Run(nil, []tf.Output{c}, nil)
	if err != nil {
		panic(err)
	}

	fmt.Println(output[0].Value())
}

func (nn *NeuralNet) Load(pb_path string) {
	var err error
	nn.saved_model, err = tf.LoadSavedModel(pb_path, []string{"serve"}, nil)
	if err != nil {
		panic(err)
	}

	nn.input_op = nn.saved_model.Graph.Operation("input/s").Output(0)
	nn.output_policy_op = nn.saved_model.Graph.Operation("policy_net_new/soft_logits").Output(0)
	nn.output_value_op = nn.saved_model.Graph.Operation("policy_net_new/value_output").Output(0)
}

func (nn *NeuralNet) Release() {
	nn.saved_model.Session.Close()
}

func (nn *NeuralNet) Ref(input_data [][][]float32) [][]float32 {
	input_tensor, err := tf.NewTensor(input_data)
	if err != nil {
		panic(err)
	}

	result, run_err := nn.saved_model.Session.Run(
		map[tf.Output]*tf.Tensor{nn.input_op: input_tensor},
		[]tf.Output{nn.output_policy_op},
		nil)

	if run_err != nil {
		panic(run_err)
	}

	// fmt.Printf("Most likely number in input is %v \n", result[0].Value())

	return result[0].Value().([][]float32)
}
