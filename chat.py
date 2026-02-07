import os
import torch
import importlib.util

# --- CONFIGURACIÓN DE RUTAS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHAT_DIR = os.path.join(BASE_DIR, "chatgpt")

def cargar_modulo(nombre, archivo):
    ruta = os.path.join(CHAT_DIR, archivo)
    spec = importlib.util.spec_from_file_location(nombre, ruta)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

# Carga de componentes con manejo de errores
try:
    cfg = cargar_modulo("config", "config.py")
    mdl = cargar_modulo("model", "model.py")
    emb = cargar_modulo("motor_texto", "motor_texto.py")
    
    Config = cfg.Config
    CachitoGPT = mdl.CachitoGPT
    SimpleTokenizer = emb.SimpleTokenizer
except Exception as e:
    print(f"[!] Error crítico al cargar los archivos del proyecto: {e}")
    exit()

def chat():
    # 1. Cargar Tokenizador y Cerebro
    tokenizer = SimpleTokenizer()
    try:
        tokenizer.load("exports/vocabulario.json")
    except:
        print("[!] No se pudo cargar el vocabulario. ¿Ya entrenaste el modelo?")
        return

    vocab_size = len(tokenizer.vocab)
    model = CachitoGPT(vocab_size=vocab_size, max_seq_len=Config.MAX_SEQUENCE_LEN)
    
    ruta_modelo = "exports/cachito_model.pth"
    if os.path.exists(ruta_modelo):
        model.load_state_dict(torch.load(ruta_modelo, weights_only=True))
        model.eval()
    else:
        print("[!] No se encontró 'cachito_model.pth'. Entrena primero con train.py")
        return

    # 2. Interfaz Visual
    os.system('cls' if os.name == 'nt' else 'clear')
    print("=" * 65)
    print("""
     _____                _               _         _____ _____ _______ 
    |  __ \              | |             (_)       / ____|  __ \__   __|
    | |__) |_ _ _ __   __| | ___ _ __ _   _  __ _ | |  __| |__) | | |   
    |  ___/ _` | '_ \ / _` |/ _ \ '__| | | |/ _` || | |_ |  ___/  | |   
    | |  | (_| | | | | (_| |  __/ |  | |_| | (_| || |__| | |      | |   
    |_|   \__,_|_| |_|\__,_|\___|_|   \__, |\__,_| \_____|_|      |_|   
                                       |___/                             
    """)
    print("                SISTEMA DE ATENCIÓN VIRTUAL")
    print("=" * 65)
    print(" Escribe tu duda o 'salir' para cerrar.")

    # 3. Bucle de Chat Corregido
    while True:
        usuario = input("\nUSUARIO > ")
        if usuario.lower() in ["salir", "exit", "quit"]:
            print("\nCachitoGPT: ¡Gracias por preferirnos! Hasta luego.")
            break
            
        if not usuario.strip(): 
            continue

        # Codificar entrada del usuario
        tokens_usuario = tokenizer.encode(usuario)
        contexto = torch.tensor([tokens_usuario], dtype=torch.long)
        
        print("CACHITO > ", end="", flush=True)
        
        # Generación de respuesta palabra por palabra
        with torch.no_grad():
            for i in range(80): # 'i' es nuestro contador seguro
                # Mantener el contexto dentro del límite que el modelo entiende
                idx_cond = contexto[:, -Config.MAX_SEQUENCE_LEN:]
                logits, _ = model(idx_cond)
                
                # Obtener la predicción de la última palabra
                logits = logits[:, -1, :] 
                proximo_token = torch.argmax(logits, dim=-1).item()
                
                palabra = tokenizer.decode([proximo_token])
                
                # Validación: Si no hay palabra o es relleno, detenerse
                if palabra is None or palabra == "<PAD>":
                    break
                
                print(palabra + " ", end="", flush=True)
                
                # Añadir la palabra generada a la memoria para la siguiente vuelta
                nuevo_token_tensor = torch.tensor([[proximo_token]], dtype=torch.long)
                contexto = torch.cat((contexto, nuevo_token_tensor), dim=1)
                
                # Parada lógica: si hay punto final y ya escribió algo razonable
                if isinstance(palabra, str) and palabra.endswith(".") and i > 5:
                    break
                    
        print("\n" + "-" * 30)

if __name__ == "__main__":
    chat()