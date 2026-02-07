import json
import os

class SimpleTokenizer:
    def __init__(self):
        self.vocab = {"<PAD>": 0, "<UNK>": 1}
        self.inverse_vocab = {0: "<PAD>", 1: "<UNK>"}

    def fit(self, text):
        # Limpieza básica y creación de vocabulario por palabras
        words = text.replace("\n", " \n ").split(" ")
        unique_words = sorted(list(set([w for w in words if w])))
        
        for i, word in enumerate(unique_words):
            if word not in self.vocab:
                idx = len(self.vocab)
                self.vocab[word] = idx
                self.inverse_vocab[idx] = word

    def encode(self, text):
        words = text.replace("\n", " \n ").split(" ")
        return [self.vocab.get(w, self.vocab["<UNK>"]) for w in words if w]

    def decode(self, tokens):
        return " ".join([self.inverse_vocab.get(t, "<UNK>") for t in tokens])

    def save(self, ruta):
        with open(ruta, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=4)

    def load(self, ruta):
        if not os.path.exists(ruta):
            print(f"[!] Archivo no encontrado: {ruta}")
            return
            
        with open(ruta, "r", encoding="utf-8") as f:
            self.vocab = json.load(f)
            
        self.inverse_vocab = {}
        for word, idx in self.vocab.items():
            try:
                self.inverse_vocab[int(idx)] = word
            except ValueError:
                # Si por alguna razón el ID no es número, intentamos lo inverso
                self.inverse_vocab[idx] = word
        
        print(f"[ OK ] Vocabulario cargado: {len(self.vocab)} palabras.")