# 🐾 CachitoGPT: Inteligencia Artificial con Sello de Equipo

¡Bienvenidos a **CachitoGPT**! Este proyecto no es solo código; es el resultado de nuestra curiosidad, de muchas horas de estudio y, sobre todo, de la unión de cuatro compañeros decididos a entender cómo funciona el "cerebro" de una IA.

##  Equipo de Desarrollo
* **Elián García**
* **Gabriela López**
* **Arianna Escalona**
* **Victor Zerpa**

## Nuestra Identidad
* **UNEFA - Núcleo Carabobo (Extensión Bejuma)**
* **Materia:** Microprocesadores | **Facilitador:** Gabriel Baute

---

## Lo que Hicimos

Más allá de los algoritmos y los tensores, **CachitoGPT** nos dejó una enseñanza que no estaba en ningún manual: **nadie llega lejos solo**. Este proyecto fue nuestra mayor lección.

Aprendimos que la Inteligencia Artificial es compleja, pero cuando hay cuatro mentes apoyándose, los problemas se volvieron pequeños. Hubo momentos de frustración cuando el código no corría, pero ahí descubrimos que **la fuerza del equipo** está en que, cuando uno se cansaba, el otro tenía la solución o una palabra de aliento.

Aprendimos a escucharnos, a confiar en el trabajo del compañero y a entender que cada pieza de código era como un eslabón de una cadena; si uno fallaba, todos estábamos ahí para repararlo. Nos vamos con la satisfacción de saber que, como equipo de la UNEFA, somos capaces de crear algo increíble cuando trabajamos con respeto, paciencia y unión.

---

##  ¿Qué es CachitoGPT?
Es un modelo de lenguaje basado en la arquitectura **Transformer**. Lo construimos desde los cimientos: diseñamos la atención, la red neuronal y el sistema de entrenamiento para que pudiera aprender de nuestro propio archivo de datos (`datos.txt`).

##  El Proceso (Paso a Paso)
1. **Arquitectura:** Creamos los módulos de atención (`attention.py`) y bloques de Transformer.
2. **Datos:** Desarrollamos un `tokenizador.py` para que la IA pudiera "leer" nuestro idioma.
3. **Entrenamiento:** Corrimos el proceso en `train.py` hasta que las respuestas tuvieron sentido.
4. **Compilación:** Generamos los archivos finales: `modelo_compilado.pt` y `configuracion.json`.

##  El Proceso de Afinación (Fine-Tuning)
No bastó con programar; tuvimos que "educar" a CachitoGPT para que fuera coherente. Así lo pulimos:
* **Correccion del Dataset:** Limpiamos y organizamos el archivo `datos.txt` para que las secuencias de aprendizaje fueran claras y sin ruidos.
* **Ajuste de Temperatura:** Calibramos la "creatividad" del modelo en el chat para evitar que repitiera palabras o inventara términos sin sentido.
* **Optimización del Loss:** Ajustamos la tasa de aprendizaje (*Learning Rate*) en `config.py` tras monitorear la curva de error durante varias horas de entrenamiento.

##  Prueba de Funcionamiento
Para validar que Cachito realmente "razona", le hicimos la prueba de fuego:

* **Pregunta:** "¿Cómo hacer un pan?"
* **Respuesta:** "Para hacer un pan necesitas harina, agua, levadura y sal. Debes amasar bien, dejar reposar la masa para que crezca y luego hornear hasta que esté dorado."

**Resultado:** ✅ Éxito. El modelo demuestra capacidad para organizar ideas y explicar procesos de forma coherente.

## Contenido de la Entrega
* **`chatgpt/`**: El corazón del modelo.
* **`modelo_compilado.pt`**: El conocimiento adquirido.
* **`configuracion.json`**: El mapa técnico del proyecto.
* **`chat.py`**: El puente para hablar con CachitoGPT.

---
*Hecho con orgullo Bejuma, Febrero de 2026.*
