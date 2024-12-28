from ml_collections import ConfigDict

from vision_rl.stb3.jax_experiments import JAXExperiments

def get_config():
    config = ConfigDict()
    
    # Framework settings
    config.framework = "torch"
    config.num_workers = 1
    config.num_gpus_per_worker = 1
    config.num_cpus_per_worker = 3
    
    # Training parameters
    config.rollout_fragment_length = 16
    config.timesteps_per_iteration = 2000
    config.train_batch_size = 16
    config.learning_starts = 5000
    config.buffer_size = 15000
    config.lr = 0.0003
    
    # Exploration configuration
    config.exploration_config = ConfigDict()
    config.exploration_config.type = "EpsilonGreedy"
    config.exploration_config.initial_epsilon = 1.0
    config.exploration_config.final_epsilon = 0.1
    config.exploration_config.epsilon_timesteps = 50000
    
    # Environment configuration
    config.env_config = ConfigDict()
    
    # Carla settings
    config.env_config.carla = ConfigDict()
    config.env_config.carla.host = "localhost"
    config.env_config.carla.timeout = 20.0
    config.env_config.carla.timestep = 0.1
    config.env_config.carla.retries_on_error = 25
    config.env_config.carla.resolution_x = 600
    config.env_config.carla.resolution_y = 600
    config.env_config.carla.quality_level = "Low"
    config.env_config.carla.enable_map_assets = True
    config.env_config.carla.enable_rendering = True
    config.env_config.carla.show_display = True
    config.env_config.carla.town = "Town02"
    
    # Experiment settings
    config.env_config.experiment = ConfigDict()
    config.env_config.experiment.type = JAXExperiments
    
    # Hero configuration
    config.env_config.experiment.hero = ConfigDict()
    config.env_config.experiment.hero.blueprint = "vehicle.mercedes.coupe_2020"
    
    # Sensors configuration
    config.env_config.experiment.hero.sensors = ConfigDict()
    
    # Collision sensor
    config.env_config.experiment.hero.sensors.collision = ConfigDict()
    config.env_config.experiment.hero.sensors.collision.type = "sensor.other.collision"
    
    # RGB camera
    config.env_config.experiment.hero.sensors.rgb = ConfigDict()
    config.env_config.experiment.hero.sensors.rgb.type = "sensor.camera.rgb"
    config.env_config.experiment.hero.sensors.rgb.image_size_x = 180
    config.env_config.experiment.hero.sensors.rgb.image_size_y = 180
    config.env_config.experiment.hero.sensors.rgb.transform = "1.9, 0.0, 1.7, 0.0, -15.0, 0.0"
    
    # IMU sensor
    config.env_config.experiment.hero.sensors.imu = ConfigDict()
    config.env_config.experiment.hero.sensors.imu.type = "sensor.other.imu"
    
    # Lane invasion sensor
    config.env_config.experiment.hero.sensors.lane_invasion = ConfigDict()
    config.env_config.experiment.hero.sensors.lane_invasion.type = "sensor.other.lane_invasion"
    
    # Background activity
    config.env_config.experiment.background_activity = ConfigDict()
    config.env_config.experiment.background_activity.n_vehicles = 0
    config.env_config.experiment.background_activity.n_walkers = 0
    config.env_config.experiment.background_activity.tm_hybrid_mode = True
    
    # Weather and other settings
    config.env_config.experiment.weather = "CloudySunset"
    config.env_config.experiment.others = ConfigDict()
    config.env_config.experiment.others.framestack = 1
    config.env_config.experiment.others.max_time_idle = 600
    config.env_config.experiment.others.max_dist = 200
    config.env_config.experiment.others.target_speed = 5.0
    
    return config