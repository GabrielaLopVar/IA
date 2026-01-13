from tokenizers import Tokenizer, models, trainers

tokenizer = Tokenizer(models.BPE())

with open("El arte de ser nosotros.txt", "r", encoding="utf-8") as f:
    contenido_archivo = f.read()

textos = [
    contenido_archivo,
    "BET cmolc (10^6/8)",
    "C(105) 10KHz/MHz 1Mx: DCT",
    "Self size: 1Mx: DCT",
    "Self mark: DCT"
]

trainer = trainers.BpeTrainer(
    vocab_size=1000,
    special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
)

tokenizer.train_from_iterator(textos, trainer)

texto_prueba = "BET cmolc (10^6/8)"
encoding = tokenizer.encode(texto_prueba)

print("Tokens:", encoding.tokens)
print("IDs:", encoding.ids)
print("Decodificado:", tokenizer.decode(encoding.ids))