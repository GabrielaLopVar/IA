# *** PANADERIA.GPT: El Cerebro de CachitoGPT***

Este proyecto representa la creación de una Inteligencia Artificial artesanal, diseñada para aprender y conversar sobre el mundo de la panadería utilizando una arquitectura de vanguardia.

## ***Código Principal del Chat (`chat.py`):***

```python
# Cargamos el cerebro (modelo) y el traductor (tokenizador)
tokenizer.load("exports/vocabulario.json")
model.load_state_dict(torch.load("exports/cachito_model.pth", weights_only=True))

# El bucle donde ocurre la magia de la generación
while True:
    usuario = input("\nUSUARIO > ")
    contexto = torch.tensor([tokenizer.encode(usuario)])
    
    # La IA predice la siguiente palabra una por una hasta completar la idea
    for i in range(80):
        logits, _ = model(contexto)
        proximo_token = torch.argmax(logits[:, -1, :])
        palabra = tokenizer.decode([proximo_token])
        
        if palabra is None or palabra == "<PAD>": break
        print(palabra, end=" ", flush=True)

¿Cómo funciona? (Explicación Humana)
Construir esta IA fue como armar un rompecabezas tecnológico. Aquí te explicamos los pilares de lo que hicimos:

 Le dimos un "Cerebro" (Arquitectura Transformer)
No usamos una simple base de datos de "pregunta y respuesta". Implementamos una estructura llamada Transformer, la misma tecnología que utiliza ChatGPT. Este cerebro no busca respuestas escritas, sino que calcula probabilidades. Cuando le preguntas por un "pan", el modelo activa sus conexiones neuronales para entender si te refieres a una receta, un ingrediente o un proceso de horneado.

 Le enseñamos un idioma (Tokenización de Palabras)
Las computadoras no entienden letras, solo números. Creamos un Motor de Texto que actúa como un traductor. Este motor tomó tu archivo datos.txt y creó un diccionario único donde cada palabra de tu panadería (como "masa", "leña" o "fermentado") tiene un número de identidad. Así, cuando tú escribes, la IA "lee" números y nos traduce la respuesta de vuelta a palabras humanas.

 La pusimos a estudiar (Entrenamiento y Pérdida)
En el archivo train.py, sometimos al modelo a un examen intenso. La IA leyó tus recetas una y otra vez, intentando adivinar cuál era la siguiente palabra. Al principio fallaba mucho (tenía un "Loss" o error alto de 5.4), pero tras varias vueltas, el error bajó a menos de 1.0, lo que significa que empezó a formar frases coherentes sobre tus productos.

La hicimos "Resistente" (Blindaje de Código)
Uno de los mayores retos fue cuando la IA se "confundía" y lanzaba errores técnicos (TypeError). Corregimos el código para que, si la IA encuentra una palabra extraña o se queda sin ideas, no se cierre el programa. Implementamos validaciones que le dicen: "Si no sabes qué decir, detente con elegancia", permitiendo que el chat sea fluido y profesional.

Guía de Uso Rápido
Alimentación: Guarda toda la información de tu panadería en data/datos.txt.

Aprendizaje: Ejecuta python train.py para que la IA estudie tus textos.

Conversación: Ejecuta python chat.py para entrar al sistema PANADERIA.GPT y empezar a chatear.

“CachitoGPT no solo repite palabras, entiende el arte de hornear a través de datos.”
