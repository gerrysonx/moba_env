import os, sys
import tensorflow as tf

arr = [47, 19, 33, 28, 13, 8, 4, 22, 14]
lol = arr[::3]
a = 10

def get_output(input_pl):
    with tf.variable_scope('scope_5001', reuse = tf.AUTO_REUSE):
        var1 = tf.get_variable(name = 'var1_001', shape = (5, 1))
        output_var = tf.matmul(input_pl, var1)
    return output_var
    pass

if __name__ == '__main__':
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    root_folder = os.path.split(os.path.abspath(__file__))[0]
    
    input1 = tf.placeholder(shape = (2, 5), dtype = tf.float32)

    with tf.variable_scope('scope_4001', reuse = tf.AUTO_REUSE):
        output_arr = []
        for idx in range(4):
            output_var = get_output(input1)
            output_arr.append(output_var)
        output_var = tf.stack(output_arr, axis = 1)    
        pass

    train_writer = tf.summary.FileWriter('{}/../_test_summary'.format(root_folder), graph=tf.get_default_graph())    
    pass
