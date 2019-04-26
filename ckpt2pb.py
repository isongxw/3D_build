from tensorflow.python.tools import inspect_checkpoint as chkp
import tensorflow as tf
from glob import glob
import pydicom
import numpy as np
from datetime import datetime

def load_scan(fileList):
    slices = [pydicom.dcmread(s) for s in fileList]
    slices.sort(key=lambda x: int(x.InstanceNumber), reverse=False)  # 将slices按照.dcm文件的InstanceNumber属性进行排序
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
    for s in slices:
        s.SliceThickness = slice_thickness

    return slices

def create_freeze_graph():
    saver = tf.train.import_meta_graph("models/Hand.cpkt-5000.meta", clear_devices=True)
    graph = tf.get_default_graph()  # 获得默认的图
    output_nodes = ["chanarc/sigm"] 

    with tf.Session(graph=tf.get_default_graph()) as sess:
        sess.run(tf.global_variables_initializer())
        input_graph_def = sess.graph.as_graph_def()
        saver.restore(sess, "models/Hand.cpkt-5000")
        output_graph_def = tf.graph_util.convert_variables_to_constants(sess,
                                                                        input_graph_def,
                                                                        output_nodes)
        with open("frozen_model.pb", "wb") as f:
            f.write(output_graph_def.SerializeToString())
    print('finish')

def freeze_graph_test(pb_path, dcm_input, output_dir, batch_size):
    '''
    :param pb_path:pb文件的路径
    :param dcm_input:测试图片
    :return:
    '''

    input = np.stack([s.pixel_array for s in dcm_input])
    input = np.array(input)
    input.astype(np.float32)
    height = input.shape[1]
    width = input.shape[2]
    input = np.expand_dims(input,3)
    output = np.zeros_like(input)
    pic_count = 0
    i = 0

    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()
        with open(pb_path, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            tf.import_graph_def(output_graph_def, name="")
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            # 定义输入的张量名称,对应网络结构的输入张量
            input_image_tensor = sess.graph.get_tensor_by_name("Placeholder:0")
            input_keep_prob_tensor = sess.graph.get_tensor_by_name("Placeholder_3:0")
            # 定义输出的张量名称
            output_tensor_name = sess.graph.get_tensor_by_name("chanarc/sigm:0")
            while i < input.shape[0]:
                print(datetime.now(),' batch', i/5, 'start')
                images = input[i:i+batch_size, :, :, :]
                out=sess.run(output_tensor_name, feed_dict={input_image_tensor: images, input_keep_prob_tensor: 1.0})
                for j in range(batch_size):
                    temp = out[j,:]
                    current_pic = i + j
                    for k in range(height):
                        for l in range(width):
                            if temp[k][l][0] > 0.5:
                                output[current_pic][k][l] = 255
                            else:
                                output[current_pic][k][l] = 0
                i += batch_size
        # 将结果写回dcm文件
        print('write start')
        if output_dir is not None:
            for i in range(input.shape[0]):
                dcm_input[i].PixelData = output[i]
                dcm_input[i].save_as(output_dir + '/' + str(pic_count) + ".dcm")
                pic_count += 1
        print('write finish')
            # print("out:{}".format(out))
            # score = tf.nn.softmax(out, name='pre')
            # class_id = tf.argmax(score, 1)
            # print("pre class_id:{}".format(sess.run(class_id)))

path = "F:/g_p/data/export/hand_t_data"
fileList = glob(path + '/*.dcm')
dcm_input = load_scan(fileList)
freeze_graph_test("frozen_model.pb", dcm_input, 'hand_output', 5)