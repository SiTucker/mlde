import ml_collections
import torch


def get_config():
    config = ml_collections.ConfigDict()

    config.training = training = ml_collections.ConfigDict()
    training.n_epochs = 100
    training.batch_size = 16 
    training.snapshot_freq = 25
    training.log_freq = 50
    training.eval_freq = 1000

    config.eval = evaluate = ml_collections.ConfigDict()
    evaluate.batch_size = 16

    config.data = data = ml_collections.ConfigDict()
    data.dataset_name = (
        "lowres"
    )
    data.input_transform_key = "stan"
    data.target_transform_key = "sqrturrecen"
    data.input_transform_dataset = None
    data.time_inputs = False
    data.image_size = 128

    config.model = model = ml_collections.ConfigDict()
    model.name = "u-net"
    model.loss = "MSELoss"

    config.optim = optim = ml_collections.ConfigDict()
    optim.optimizer = "Adam"
    optim.lr = 2e-4

    config.seed = 42
    config.device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )

    return config
