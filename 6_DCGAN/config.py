# ################################################################
#                             HyperParameters
# ################################################################

class Hyperparameters:
    # ################################################################
    #                             Data
    # ################################################################
    device = 'cpu' # cpu
    data_root = 'D:/Mooc/celeba'

    image_size = 64
    seed = 1234

    # ################################################################
    #                             Model
    # ################################################################
    z_dim = 100 # laten z dimension
    data_channels = 3 # RGB face

    # ################################################################
    #                             Exp
    # ################################################################
    batch_size = 64
    n_workers = 2       # data loader works
    beta = 0.5          # adam optimizer 0.5
    init_lr = 0.0002
    epochs = 1000
    verbose_step = 250  # evaluation: store image during training
    save_step = 1000    # save model step


HP = Hyperparameters()