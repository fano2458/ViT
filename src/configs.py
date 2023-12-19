import ml_collections


def get_ti16_config():
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.hidden_size = 192
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 768
    config.transformer.num_heads = 3
    config.transformer.num_layers = 12
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.classifier = 'token'
    config.representation_size = None
    return config  

def get_ti32_config():
    config = get_ti16_config()
    config.patches.size = (32,32)
    return config
