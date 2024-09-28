# ################################################################
#                             HyperParameters
# ################################################################
# semi-supervised learning:
#     1. model structure
#     2. hype setting are important!
class Hyperparameters:
    # ################################################################
    #                             Data
    # ################################################################
    device = 'cpu' # cuda for training, cpu/cuda for inference
    classes_num = 10 # cifar10
    n_labeled = 250 # total labeled data number
    seed = 1234

    # ################################################################
    #                             Model
    # ################################################################
    T = 0.5 # sharpen temperature
    K = 2 # agument K
    alpha = 0.75 # beta sample hype
    lambda_u = 75. # consistency loss weight
    # ################################################################
    #                             Exp
    # ################################################################
    batch_size = 8
    init_lr = 0.002
    epochs = 1000
    verbose_step = 300
    save_step = 300

HP = Hyperparameters()

