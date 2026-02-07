import os
import torch
import importlib.util
import sys  # <--- IMPORTANTE: Para el cierre inmediato

# --- CONFIGURACIÓN DE RUTAS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHAT_DIR = os.path.join(BASE_DIR, "chatgpt")

def cargar_modulo(nombre, archivo):
    ruta = os.path.join(CHAT_DIR, archivo)
    spec = importlib.util.spec_from_file_location(nombre, ruta)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

# Carga de componentes
try:
    cfg = cargar_modulo("config", "config.py")
    mdl = cargar_modulo("model", "model.py")
    emb = cargar_modulo("motor_texto", "motor_texto.py")
    
    Config = cfg.Config
    CachitoGPT = mdl.CachitoGPT
    SimpleTokenizer = emb.SimpleTokenizer
except Exception as e:
    print(f"[!] Error: {e}")
    sys.exit()

def chat():
    # 1. Preparación de IA
    tokenizer = SimpleTokenizer()
    tokenizer.load("exports/vocabulario.json")
    vocab_size = len(tokenizer.vocab)
    
    model = CachitoGPT(vocab_size=vocab_size, max_seq_len=Config.MAX_SEQUENCE_LEN)
    ruta_modelo = "exports/cachito_model.pth"
    
    if os.path.exists(ruta_modelo):
        model.load_state_dict(torch.load(ruta_modelo, weights_only=True))
        model.eval()
    else:
        print("[!] Modelo no encontrado.")
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

    # 3. BUCLE DE CHAT (CORREGIDO)
    while True:
        usuario = input("\nUSUARIO > ")
        
        # Validación de Salida Instantánea
        if usuario.lower() in ["salir", "exit", "quit"]:
            print("\nCachitoGPT: ¡Hasta la próxima!")
            sys.exit() # <--- Aquí cerramos todo el proceso

        if not usuario.strip():
            continue

        # Generación de la IA
        contexto = torch.tensor([tokenizer.encode(usuario)], dtype=torch.long)
        print("CACHITO > ", end="", flush=True)
        
        with torch.no_grad():
            for i in range(80):
                idx_cond = contexto[:, -Config.MAX_SEQUENCE_LEN:]
                logits, _ = model(idx_cond)
                logits = logits[:, -1, :] 
                
                proximo_token = torch.argmax(logits, dim=-1).item()
                palabra = tokenizer.decode([proximo_token])
                
                if palabra is None or palabra == "<PAD>":
                    break
                
                print(palabra + " ", end="", flush=True)
                
                nuevo_token = torch.tensor([[proximo_token]], dtype=torch.long)
                contexto = torch.cat((contexto, nuevo_token), dim=1)
                
                # Parada lógica por punto final
                if isinstance(palabra, str) and palabra.endswith(".") and i > 10:
                    break
        print("\n" + "-" * 30)

if __name__ == "__main__":
    chat()