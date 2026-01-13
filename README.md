# ***BPE Tokenizer para Textos Híbridos: Literarios y Técnicos***

## ***Código:***

```python
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
```

## _¿Cómo funciona?_

_Este código implementa un tokenizador BPE, dónde:_

- ###  _Importación e inicialización_

_Primero se importan los componentes esenciales de la biblioteca tokenizers: Tokenizer para crear el tokenizador, models para elegir el algoritmo de tokenización (BPE en este caso), y trainers para configurar el entrenamiento. Luego se instancia un tokenizador específicamente con el modelo BPE (Byte Pair Encoding), que es un algoritmo que aprende subpalabras fusionando progresivamente los caracteres más frecuentes del texto._

```python
from tokenizers import Tokenizer, models, trainers
tokenizer = Tokenizer(models.BPE())  
```

- ### _Preparación de datos_

_El código carga primero un archivo de texto llamado "El arte de ser nosotros.txt" en formato UTF-8. Este contenido se almacena en una variable y luego se combina con otros textos de ejemplo en una lista llamada textos. Estos textos adicionales parecen contener términos técnicos, fórmulas o códigos específicos. Al incluir ambos tipos de contenido, se asegura que el tokenizador aprenda tanto lenguaje natural como vocabulario técnico._

```python
with open("El arte de ser nosotros.txt", "r", encoding="utf-8") as f:
    contenido_archivo = f.read()

textos = [
    contenido_archivo,
    "BET cmolc (10^6/8)",
    "C(105) 10KHz/MHz 1Mx: DCT",
    "Self size: 1Mx: DCT",
    "Self mark: DCT"
```

- ### _Configuración del entrenamiento_

_Se crea un objeto BpeTrainer con parámetros clave: vocab_size=1000 lo que limita el vocabulario a mil tokens, obligando al algoritmo a aprender las subpalabras más útiles y frecuentes. Los special_tokens son símbolos reservados para funciones específicas: [UNK] representa palabras desconocidas, [CLS] y [SEP] para tareas de clasificación y separación, [PAD] para igualar longitudes, y [MASK] para entrenamiento con enmascaramiento._

```python
trainer = trainers.BpeTrainer(
    vocab_size=1000,  # Tamaño máximo del vocabulario
    special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]  # Tokens especiales
)
```

- ### _Entrenamiento del tokenizador_
 
_Finalmente, se entrena el tokenizador llamando a train_from_iterator con la lista de textos y el trainer configurado. El algoritmo BPE analiza estadísticamente todo el corpus, identifica los pares de caracteres o subpalabras más frecuentes, y los fusiona iterativamente. Repite este proceso hasta alcanzar el tamaño de vocabulario especificado, creando un conjunto óptimo de tokens que balancea eficiencia y cobertura léxica para los textos proporcionados._

```python
 tokenizer.train_from_iterator(textos, trainer)
```
