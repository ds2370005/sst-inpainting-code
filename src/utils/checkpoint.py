"""Explicit boundary between the old simplified network and the paper revision."""
MODEL_FORMAT = 'hirahara_tables_v2'


def require_model_format(checkpoint):
    if checkpoint.get('model_format') != MODEL_FORMAT:
        raise ValueError('Incompatible legacy checkpoint: revised paper architecture requires retraining (hirahara_tables_v2).')


def load_generator_state(model, checkpoint):
    require_model_format(checkpoint)
    state = checkpoint['generator_state_dict']
    if hasattr(model, 'anomaly_scale'):
        import torch
        if 'anomaly_scale' not in state or not torch.allclose(
            state['anomaly_scale'].cpu(), model.anomaly_scale.detach().cpu(), rtol=1e-6, atol=0):
            raise ValueError('Checkpoint anomaly normalization differs from current configuration')
    model.load_state_dict(state, strict=True)
