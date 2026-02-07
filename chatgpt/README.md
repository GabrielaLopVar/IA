# Tokenizador Simple y Modelo de Embeddings con PyTorch


## Descripción
Este proyecto implementa un sistema completo de procesamiento de texto que incluye tokenización y generación de embeddings vectoriales usando PyTorch. El código demuestra los conceptos fundamentales de Procesamiento de Lenguaje Natural (NLP) desde los fundamentos.


## Estructura del Proyecto

Componentes Principales

1. SimpleTokenizer (SimpleTokenizer)

    Clase personalizada que maneja la tokenización de texto:

    - Construye un vocabulario a partir de texto de entrenamiento
    - Asigna IDs únicos a cada token (palabra)
    - Guarda el vocabulario en formato JSON
    - Convierte texto en secuencias de IDs

2. SequentialEmbedding (SequentialEmbedding)

    Modelo de red neuronal basado en PyTorch que:

    - Implementa una capa de embedding para convertir tokens IDs en vectores densos
    - Hereda de `nn.Module` de PyTorch
    - Configurable en dimensiones de embedding y tamaño de vocabulario

3. Función Principal (main)
    
    Orquesta todo el proceso:
    1. Crea/lee datos de entrenamiento
    2. Entrena el tokenizador
    3. Codifica texto de ejemplo
    4. Genera embeddings vectoriales
    5. Muestra resultados


## Características

- Tokenización básica: Divide texto en tokens por espacios y convierte a minúsculas
- Gestión de vocabulario: Asignación automática de IDs y persistencia en JSON
- Embeddings aprendibles: Capa de embedding de PyTorch con inicialización aleatoria
- Procesamiento por lotes: Soporte para tensores batch (dimensiones extra)
- Manejo de GPU/CPU: Compatible con aceleración por GPU de PyTorch


## Flujo de Datos

    text
    Texto crudo → Tokenización → IDs de tokens → Embeddings → Vectores densos


## Configuración Técnica

Parámetros del Modelo

- Dimensión de embedding: 512 (vector de 512 valores por token)
- Longitud máxima de secuencia: 512 tokens
- Tamaño de vocabulario: Determinado automáticamente por los datos

Dependencias

        python
        torch >= 1.9.0
        json (built-in)
        os (built-in)


## Ejemplo de Uso

Entrada:

    python
    texto_entrenamiento = "el gato corre por el jardin y el perro salta la valla"
    frase_ejemplo = "el gato salta"

Proceso:

Tokenización: `"el gato salta"` → `[0, 1, 2]`

Embedding: `[0, 1, 2]` → `tensor[3, 512]`

Salida:

    text
    Shape de la matriz de embedding: torch.Size([1, 3, 512])


## Archivos Generados

`vocabulario.json`\
Contiene el mapeo token↔ID en formato JSON legible:

    json
    {
        "el": 0,
        "gato": 1,
        "salta": 2,
        ...
    }
`datos.txt`\
Archivo de texto de entrenamiento (creado automáticamente si no existe)


## Modo de Uso
1. Ejecución básica:

        bash
        python embedding_model.py

2. Personalización:

   - Modificar file_path para usar tus propios datos
   - Ajustar EMBED_DIM y MAX_LEN según necesidades
   - Cambiar el texto de ejemplo en frase_ejemplo


## Aplicaciones Potenciales
- Aprendizaje de representaciones de palabras
- Preprocesamiento para modelos de NLP más complejos
- Prototipado rápido de sistemas de embeddings
- Educación en conceptos de tokenización y word embeddings