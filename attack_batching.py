"""Validation for fixed adversarial-optimization batch sizes."""

MIN_ATTACK_BATCH_SIZE = 1


def initialize_attack_batch_state(args):
    batch_size = int(getattr(args, "attack_batch_size", MIN_ATTACK_BATCH_SIZE))
    if batch_size < MIN_ATTACK_BATCH_SIZE:
        raise ValueError("--attack-batch-size must be at least 1")
    args.attack_batch_size = batch_size
    return batch_size


def current_attack_batch_size(args):
    return initialize_attack_batch_state(args)
