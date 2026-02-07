import os

class Config:
    # RUTAS
    DATA_PATH = "datos.txt"
    MODEL_PATH = "cachito_modelo.pth"
    VOCAB_PATH = "vocabulario.json"

    # MODELO
    EMBED_DIM = 512  
    NUM_HEADS = 8    
    NUM_LAYERS = 6   
    MAX_SEQUENCE_LEN = 128 
    DROPOUT_RATE = 0.1

    # ENTRENAMIENTO
    BATCH_SIZE = 32  
    LEARNING_RATE = 1e-3 
    EPOCHS = 5
    STEPS_PER_EPOCH = 100

    # VARIABLE DINÁMICA
    VOCAB_SIZE = 0 

    @classmethod
    def dict(cls):
        return {k: v for k, v in cls.__dict__.items() if not k.startswith('__') and not callable(v)}