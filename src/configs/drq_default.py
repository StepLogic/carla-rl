import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    config.actor_lr = 3e-4
    config.critic_lr = 3e-4
    config.temp_lr = 1e-5
    config.hidden_dims = (256, 256)
    config.cnn_features = (32, 64)
    config.cnn_filters = (1, 3)
    config.cnn_strides = (2, 2)
    # config.hidden_dims = (256, 256)
    # config.cnn_features = (32, 64, 128, 256)
    # config.cnn_filters = (3, 3, 3, 3)
    # config.cnn_strides = (2, 2, 2, 2)
    config.cnn_padding = "VALID"
    config.latent_dim = 128
    config.encoder = "d4pg"
    config.discount = 0.99
    config.tau = 0.005
    config.num_qs=10
    config.init_temperature = 0.2
    config.target_entropy = -0.5
    config.backup_entropy = True
    config.critic_reduction = "mean"
    return config
