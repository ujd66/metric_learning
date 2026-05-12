import yaml


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


_DEFAULT_CONFIG = {
    "seed": 42,
    "device": "cuda",
    "num_points": 2048,
    "input_channels": 3,
    "num_known_classes": 19,
    "use_negative_class": True,
    "negative_label": 19,
    "num_classes": 20,
    "embedding_dim": 256,
    "label_mapping": {
        "negative_names": ["negative", "qita", "other", "others", "其他"],
        "force_negative_label": 19,
    },
    "data": {
        "root": "./dataset",
        "train_split": "train",
        "val_split": "val",
        "test_split": "test",
    },
    "augmentation": {
        "use_jitter": True,
        "jitter_sigma": 0.01,
        "jitter_clip": 0.05,
        "use_random_z_rotation": True,
        "use_random_dropout": True,
        "dropout_ratio": 0.1,
    },
    "train": {
        "epochs": 100,
        "batch_size": 32,
        "lr": 0.001,
        "weight_decay": 0.0001,
        "optimizer": "adamw",
        "use_class_weight": True,
        "class_weight_max": 10.0,
    },
}


def get_default_config():
    import copy
    return copy.deepcopy(_DEFAULT_CONFIG)
