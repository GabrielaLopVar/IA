import torch

class Config:

    VOCAB_SIZE = 50257
    EMBED_DIM = 512
    MAX_SEQUENCE_LEN = 512
    NUM_LAYERS = 6
    NUM_HEADS = 8
    HIDDEN_DIM = 512
 
    BATCH_SIZE = 32
    LEARNING_RATE = 3e-4
    DROPOUT_RATE = 0.1

    TOKENIZER_SPECIAL_TOKENS = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
    TOKENIZER_MIN_FREQUENCY = 2

    DATA_PATH = "datos.txt"
    TOKENIZER_PATH = "tokenizer.json"
    VOCAB_PATH = "vocab.json"
    
    MATRIX_REPRESENTATION = """
    \begin{pmatrix}
    A & B \\
    C & D
    \end{pmatrix}
    =
    \begin{pmatrix}
    A & C \\
    B & D
    \end{pmatrix}
    """
    
    @classmethod
    def get_summary(cls):
        return {
            'model': {
                'vocab_size': cls.VOCAB_SIZE,
                'embed_dim': cls.EMBED_DIM,
                'max_seq_len': cls.MAX_SEQUENCE_LEN,
                'num_layers': cls.NUM_LAYERS,
                'num_heads': cls.NUM_HEADS
            },
            'training': {
                'batch_size': cls.BATCH_SIZE,
                'learning_rate': cls.LEARNING_RATE,
                'dropout_rate': cls.DROPOUT_RATE
            },
            'paths': {
                'data': cls.DATA_PATH,
                'tokenizer': cls.TOKENIZER_PATH,
                'vocab': cls.VOCAB_PATH
            }
        }
    
    @classmethod
    def display_config(cls):
        print("=" * 50)
        print("CONFIGURACIÓN DEL MODELO")
        print("=" * 50)
        
        config = cls.get_summary()
        for section, values in config.items():
            print(f"\n{section.upper()}:")
            for key, value in values.items():
                print(f"  {key}: {value}")
        
        print(f"\nTokens especiales: {cls.TOKENIZER_SPECIAL_TOKENS}")
        print(f"Matriz de ejemplo: {cls.MATRIX_REPRESENTATION[:50]}...")

class DevelopmentConfig(Config):
    
    BATCH_SIZE = 4
    LEARNING_RATE = 1e-3

class ProductionConfig(Config):
    
    BATCH_SIZE = 64
    DROPOUT_RATE = 0.2

class TestConfig(Config):
    BATCH_SIZE = 2
    MAX_SEQUENCE_LEN = 128

if __name__ == "_main_":
    Config.display_config()

    dev_config = DevelopmentConfig()
    print(f"\nTasa de aprendizaje en desarrollo: {dev_config.LEARNING_RATE}")

    summary = Config.get_summary()
    print(f"\nResumen del modelo: {summary['model']}")