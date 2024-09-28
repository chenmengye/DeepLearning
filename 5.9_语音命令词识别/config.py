# ################################################################
#                             HyperParameters
# ################################################################

class Hyperparameters:
    # ################################################################
    #                             Data
    # ################################################################
    device = 'cpu' # cuda
    data_root = './data'

    cls_mapper_path = './data/cls_mapper.json'

    metadata_train_path = './data/train_kws.txt'
    metadata_eval_path = './data/eval_kws.txt'
    metadata_test_path = './data/test_kws.txt'

    classes_num = 8 # 8 classes= 7 speech command + bng
    mel_size = 40
    seed = 1234

    # ################################################################
    #                             Model
    # ################################################################
    data_point_channel = mel_size
    rnn_hidden_dim = 256
    rnn_layer_num = 2
    is_bidirection = True
    fc_drop = 0.3

    # ################################################################
    #                             Exp
    # ################################################################
    batch_size = 8
    init_lr = 5e-4
    epochs = 100
    verbose_step = 250
    save_step = 500


HP = Hyperparameters()