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
    train_data_root = './data/shp_marcel_train/Marcel-Train'
    test_data_root = './data/shp_marcel_test/Marcel-Test'

    metadata_train_path = './data/train_hand_gesture.txt'
    metadata_eval_path = './data/eval_hand_gesture.txt'
    metadata_test_path = './data/test_hand_gesture.txt'

    classes_num = 6
    seed = 1234

    # ################################################################
    #                             Model
    # ################################################################
    data_channels = 3
    conv_kernel_size = 3
    fc_drop_prob = 0.3

    # ################################################################
    #                             Exp
    # ################################################################
    batch_size = 1
    init_lr = 5e-4
    epochs = 100
    verbose_step = 250
    save_step = 500


HP = Hyperparameters()