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
    train_data_root = './data/train/'
    dev_data_root = './data/dev/'
    test_data_root = './data/test/'

    metadata_train_path = './data/meta_train.txt'
    metadata_dev_path = './data/meta_dev.txt'
    metadata_test_path = './data/meta_test.txt'

    classes_num = 11
    seed = 1234

    # ################################################################
    #                             Model
    # ################################################################
    if_conv_frozen = True

    # ################################################################
    #                             Exp
    # ################################################################
    batch_size = 32
    init_lr = 5e-4
    epochs = 100
    verbose_step = 30
    save_step = 30


HP = Hyperparameters()