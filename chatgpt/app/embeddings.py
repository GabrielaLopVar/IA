import torch
import torch.nn as nn
import json
import os

# ==========================================
# 1. CLASE TOKENIZADOR (Paso 2 y 3)
# ==========================================
class SimpleTokenizer:
    def __init__(self):
        self.vocab = {}  # Token -> ID
        self.ids_to_tokens = {} # ID -> Token
        self.next_id = 0

    def fit(self, text):
        """Lee el texto y crea el vocabulario"""
        # Limpieza básica: minúsculas y split por espacios
        tokens = text.lower().split()
        unique_tokens = sorted(list(set(tokens)))
        
        for token in unique_tokens:
            if token not in self.vocab:
                self.vocab[token] = self.next_id
                self.ids_to_tokens[self.next_id] = token
                self.next_id += 1
        
        self.save_vocab()

    def encode(self, text):
        """Convierte una frase en una lista de IDs"""
        tokens = text.lower().split()
        # Si el token no existe, lo ignoramos por ahora (o podrías usar un token <UNK>)
        return [self.vocab[t] for t in tokens if t in self.vocab]

    def save_vocab(self, filename="vocabulario.json"):
        """Guarda el vocabulario en un JSON (Paso 2)"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=4)
        print(f"✅ Vocabulario guardado en {filename}")

# ==========================================
# 2. CLASE EMBEDDING (Paso 4)
# ==========================================
class SequentialEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim, max_seq_len):
        super(SequentialEmbedding, self).__init__()
        # Definimos la capa de embedding de PyTorch
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim)
        self.max_seq_len = max_seq_len

    def forward(self, x):
        # x es el tensor de IDs
        return self.embedding(x)

# ==========================================
# 3. EJECUCIÓN DEL MAIN (Paso 5 y 6)
# ==========================================
def main():
    # --- Paso 1: Cargar archivo de texto ---
    file_path = "datos.txt"
    if not os.path.exists(file_path):
        with open(file_path, "w") as f:
            f.write("el gato corre por el jardin y el perro salta la valla")
    
    with open(file_path, "r", encoding="utf-8") as f:
        texto_entrenamiento = f.read()

    # --- Paso 2: Instanciar y entrenar Tokenizador ---
    tokenizer = SimpleTokenizer()
    tokenizer.fit(texto_entrenamiento)

    # --- Paso 3: Probar con una frase de ejemplo ---
    frase_ejemplo = "el gato salta"
    tokens_ids = tokenizer.encode(frase_ejemplo)
    print(f"\nFrase: '{frase_ejemplo}'")
    print(f"Tokens IDs: {tokens_ids}")

    # --- Paso 4 & 5: Configurar Embedding y convertir a Tensor ---
    VOCAB_SIZE = len(tokenizer.vocab)
    EMBED_DIM = 512    # Dimensión de cada vector
    MAX_LEN = 512      # Longitud máxima permitida
    
    # Instanciamos nuestra clase
    model_embedding = SequentialEmbedding(VOCAB_SIZE, EMBED_DIM, MAX_LEN)

    # Convertimos la lista de IDs a un Tensor de PyTorch
    # El shape debe ser [1, longitud_secuencia] (Batch size = 1)
    input_tensor = torch.tensor(tokens_ids).unsqueeze(0) 
    print(f"Shape del tensor de entrada: {input_tensor.shape}")

    # --- Paso 6: Obtener la matriz vectorial ---
    with torch.no_grad(): # No necesitamos calcular gradientes todavía
        matriz_vectorial = model_embedding(input_tensor)

    print("\n--- RESULTADO FINAL ---")
    print(f"Shape de la matriz de embedding: {matriz_vectorial.shape}")
    # El shape será [1, 3, 512] -> [Batch, Tokens, Dimensiones]
    print(f"Primeros 5 valores del primer token:\n{matriz_vectorial[0][0][:5]}")
    print("-----------------------")

if __name__ == "__main__":
    main()