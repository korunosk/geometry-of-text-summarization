CONFIG_MODELS = {
    'TransformSinkhornRegModel': {
        'D_in': 768,
        'D_out': 768,
        'p': 2,
        'blur': .05,
        'scaling': .9,
        'learning_rate': 1e-2,
        'batch_size': 100
    },
    'TransformSinkhornPRModel': {
        'D_in': 768,
        'D_out': 768,
        'p': 2,
        'blur': .05,
        'scaling': .9,
        'scaling_factor': 1,
        'learning_rate': 1e-2,
        'batch_size': 100
    },
    'NeuralNetSinkhornPRModel': {
        'D_in': 768,
        'D_out': 768,
        'p': 2,
        'blur': .05,
        'scaling': .9,
        'scaling_factor': 1,
        'learning_rate': 1e-2,
        'batch_size': 100
    },
    'NeuralNetScoringPRModel': {
        'emb_dim': 768,
        'D_in': 1536,
        'H': 1536,
        'scaling_factor': 1,
        'learning_rate': 1e-2,
        'batch_size': 100
    },
    'NeuralNetScoringPREmbModel': {
        'emb_dim': 768,
        'D_in': 1536,
        'H': 1536,
        'scaling_factor': 1,
        'learning_rate': 1e-2,
        'batch_size': 100
    }
}
