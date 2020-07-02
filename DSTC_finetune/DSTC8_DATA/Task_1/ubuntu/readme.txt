


将数据集的json 文件放进来
运行顺序
*注意将超参数改为bert-large
data_preprocess.py　　生成txt 数据

HUB_hea_dic.py　　　　　生成REMBED 词表


data_HUB/REBED_preprocess_advisor.py　生成训练用的tfrecord 文件(447行的　num_epoch　需要与生成的训练文件数量相对应 == data_preprocess.py 中的num_train_epochs　参数）




