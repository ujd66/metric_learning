"""Samplers for metric learning training.

PKSampler ensures each mini-batch contains samples from P classes
with K samples per class, which is essential for contrastive losses.
"""

import warnings

import numpy as np
import torch
from torch.utils.data import Sampler


class PKSampler(Sampler):
    """Sample mini-batches with P classes and K samples per class.

    Each batch has exactly P * K samples. For classes with fewer than K
    samples, sampling is done with replacement.

    Each epoch iterates enough to cover all training samples at least once.

    Args:
        labels: List/array of integer labels for each sample.
        classes_per_batch: Number of distinct classes per batch (P).
        samples_per_class: Number of samples per class per batch (K).
        include_negative: If False, exclude negative_label from sampling.
        negative_label: Label id for the negative class.
        drop_singleton_classes: If True, skip classes with < 2 samples.
        shuffle_classes: Whether to shuffle class order each epoch.
    """

    def __init__(
        self,
        labels,
        classes_per_batch=8,
        samples_per_class=4,
        include_negative=False,
        negative_label=19,
        drop_singleton_classes=True,
        shuffle_classes=True,
    ):
        self.labels = np.asarray(labels)
        self.P = classes_per_batch
        self.K = samples_per_class
        self.include_negative = include_negative
        self.negative_label = negative_label
        self.drop_singleton_classes = drop_singleton_classes
        self.shuffle_classes = shuffle_classes

        # Build per-class index map
        self.class_indices = {}
        for idx, lbl in enumerate(self.labels):
            lbl = int(lbl)
            if lbl not in self.class_indices:
                self.class_indices[lbl] = []
            self.class_indices[lbl].append(idx)

        # Filter classes
        available_classes = []
        for lbl, indices in sorted(self.class_indices.items()):
            # Exclude negative if configured
            if not self.include_negative and lbl == self.negative_label:
                continue
            # Exclude singleton if configured
            if self.drop_singleton_classes and len(indices) < 2:
                warnings.warn(
                    f"PKSampler: dropping class {lbl} (only {len(indices)} sample(s))"
                )
                continue
            available_classes.append(lbl)

        self.available_classes = available_classes

        if len(self.available_classes) < self.P:
            warnings.warn(
                f"PKSampler: only {len(self.available_classes)} classes available, "
                f"reducing P from {self.P} to {len(self.available_classes)}"
            )
            self.P = len(self.available_classes)

        # Compute number of batches per epoch
        # Goal: each epoch should see all training data roughly once
        total_samples = sum(
            len(self.class_indices[lbl]) for lbl in self.available_classes
        )
        batch_size = self.P * self.K
        self.num_batches = max(1, total_samples // batch_size)

    def __iter__(self):
        class_order = list(self.available_classes)
        if self.shuffle_classes:
            np.random.shuffle(class_order)

        # For each class, create a pool of samples by repeating the class
        # indices enough times to last the full epoch.
        # Each batch picks P classes and K samples per class.
        # With C classes, each class appears in approximately
        # num_batches * P / C batches, needing K * num_batches * P / C samples.

        all_indices = []
        for lbl in class_order:
            indices = np.array(self.class_indices[lbl])
            # How many times this class will be sampled this epoch
            C = len(class_order)
            times_sampled = max(1, self.num_batches * self.P // C + 1)
            needed = times_sampled * self.K

            if len(indices) >= needed:
                # Sample without replacement, shuffle
                pool = np.random.choice(indices, size=needed, replace=False)
            else:
                # Repeat and shuffle
                repeats = (needed // len(indices)) + 1
                pool = np.tile(indices, repeats)
                np.random.shuffle(pool)
                pool = pool[:needed]

            all_indices.extend(pool.tolist())

        # Now group into batches: each batch takes P*K consecutive items
        # but since classes are sequential, we need to interleave
        # Reorganize: group into chunks of P classes, K samples each
        batch_size = self.P * self.K
        organized = []

        # Per-class pools
        class_pools = {}
        offset = 0
        for lbl in class_order:
            times_sampled = max(1, self.num_batches * self.P // len(class_order) + 1)
            needed = times_sampled * self.K
            class_pools[lbl] = all_indices[offset:offset + needed]
            offset += needed

        # Build batches
        class_cursors = {lbl: 0 for lbl in class_order}
        for _ in range(self.num_batches):
            # Pick P classes (round-robin through shuffled order)
            batch_classes = []
            for j in range(self.P):
                cls_idx = j % len(class_order)
                batch_classes.append(class_order[cls_idx])

            batch_indices = []
            for lbl in batch_classes:
                pool = class_pools[lbl]
                cursor = class_cursors[lbl]
                if cursor + self.K <= len(pool):
                    batch_indices.extend(pool[cursor:cursor + self.K])
                    class_cursors[lbl] = cursor + self.K
                else:
                    # Wrap around
                    remaining = pool[cursor:]
                    needed_extra = self.K - len(remaining)
                    batch_indices.extend(remaining)
                    batch_indices.extend(pool[:needed_extra])
                    class_cursors[lbl] = needed_extra

            if self.shuffle_classes:
                np.random.shuffle(batch_indices)
            organized.extend(batch_indices)

        yield from organized

    def __len__(self):
        return self.num_batches * self.P * self.K
