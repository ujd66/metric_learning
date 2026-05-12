import yaml


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_default_config():
    import copy
    return copy.deepcopy(_DEFAULT_CONFIG)


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
    "experiment": {
        "name": "pointnet_ce_supcon",
        "output_subdir": "ce_supcon",
    },
    "loss": {
        "ce_weight": 1.0,
        "metric_type": "supcon",
        "metric_weight": 0.05,
        "temperature": 0.07,
        "include_negative_in_metric_loss": False,
        "warmup_epochs_for_metric_loss": 10,
    },
    "metric_learning": {
        "enabled": False,
        "normalize_embedding": True,
        "exclude_negative": True,
        "hard_pair_focus": True,
    },
    "sampler": {
        "use_pk_sampler": False,
        "classes_per_batch": 8,
        "samples_per_class": 4,
        "include_negative": False,
        "drop_singleton_classes": True,
    },
    "analysis": {
        "focus_class_pairs": [],
    },
    "prototype": {
        "enabled": True,
        "build_split": "train",
        "normalize_embeddings": True,
        "use_known_only": True,
    },
    "ood": {
        "enabled": True,
        "similarity_threshold": 0.65,
        "threshold_min": 0.1,
        "threshold_max": 0.99,
        "threshold_step": 0.01,
        "target_known_accept_rate": 0.95,
        "threshold_selection": "max_negative_rejection_under_known_accept_constraint",
    },
}
