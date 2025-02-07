import ml_collections


def get_config():
    config = ml_collections.ConfigDict()
    config.actor_lr = 3e-4
    config.critic_lr = 3e-4
    config.hidden_dims = (256, 256)
    config.cnn_features = (32, 64, 128, 256)
    config.cnn_filters = (3, 3, 3, 3)
    config.cnn_strides = (2, 2, 2, 2)
    config.cnn_padding = "VALID"
    config.latent_dim = 50
    config.encoder = "d4pg"
    config.discount = 0.98
    config.critic_reduction = "mean"
    config.clip_ratio = 0.2  # Add clip ratio
    config.gae_lambda = 0.95  # Add GAE lambda
    return config
