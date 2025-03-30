import torch
import torch.nn as nn
import numpy as np
import os

# Detectar si hay GPU disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Mensaje sobre el uso de la GPU o CPU
if device.type == "cuda":
    print("Usando GPU (CUDA) para el entrenamiento.")
elif device.type == "cpu":
    print("Usando CPU para el entrenamiento.")

# Datos de ejemplo (bilingüe)
saludos = [
    # Inglés
    "hi", "hello", "good morning", "how are you", "what's your name",
    # Español
    "hola", "buenos dias", "buenas tardes", "buenas noches", "como estas",
    "cual es tu nombre", "que tal", "saludos"
]

respuestas = [
    # Respuestas en inglés
    "Hello!", "Hi there!", "Good morning!", "I'm doing great, thanks!", "My name is AI Bot!",
    # Respuestas en español
    "¡Hola!", "¡Buenos días!", "¡Buenas tardes!", "¡Buenas noches!", "¡Estoy muy bien, gracias!",
    "Me llamo AI Bot", "¡Muy bien!", "¡Hola! ¿Cómo estás?"
]

# Convertir los datos a tensores
def texto_a_tensor(texto, max_length=30):  
    tensor = torch.zeros(max_length, dtype=torch.long)
    for i, char in enumerate(texto):
        if i < max_length:
            tensor[i] = ord(char)
    return tensor

# Convertir tensor a texto
def tensor_a_texto(tensor):
    return ''.join([chr(int(val)) for val in tensor if val > 0])

# Encontrar la longitud máxima
max_length = max(max(len(s) for s in saludos), max(len(s) for s in respuestas))

# Crear tensores de entrada y salida
entradas = [texto_a_tensor(saludo, max_length).view(-1, 1) for saludo in saludos]
salidas = [texto_a_tensor(respuesta, max_length).view(-1, 1) for respuesta in respuestas]

# Definir el modelo
class SaludoAI(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SaludoAI, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(256, hidden_size)  
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers=2)  
        self.out = nn.Linear(hidden_size, output_size)
    
    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output, hidden = self.gru(embedded, hidden)
        output = self.out(output[0])
        return output, hidden

    def init_hidden(self):
        return torch.zeros(2, 1, self.hidden_size).to(device)  # Mover a la GPU

# Parámetros del modelo
hidden_size = 256  
output_size = 256  

# Ruta para guardar el modelo
modelo_path = './training_data/modelo_saludo.pth'

# Cargar el modelo si ya existe
if os.path.exists(modelo_path):
    modelo = torch.load(modelo_path)
    print("Modelo cargado desde el archivo.")
else:
    # Crear el modelo y moverlo a la GPU
    modelo = SaludoAI(1, hidden_size, output_size).to(device)

# Definir la función de pérdida y el optimizador
criterio = nn.CrossEntropyLoss()
optimizador = torch.optim.Adam(modelo.parameters(), lr=0.001)

# Entrenar el modelo solo si hay nuevos datos
if len(entradas) > 0 and len(salidas) > 0:
    n_epocas = 2000
    print("Entrenando el modelo...")
    for epoca in range(n_epocas):
        loss_total = 0
        for entrada, salida in zip(entradas, salidas):
            hidden = modelo.init_hidden()
            modelo.zero_grad()
            loss = 0  # Reset loss for the batch
            
            # Aumentar el tamaño del lote
            entrada = entrada.repeat(2, 1)  # Ejemplo de duplicar el tamaño del lote
            salida = salida.repeat(2, 1)
            
            # Mover tensores a la GPU
            entrada = entrada.to(device)
            salida = salida.to(device)
            
            for i in range(len(entrada)):
                try:
                    output, hidden = modelo(entrada[i].view(1, -1).to(device), hidden)
                    loss += criterio(output.view(1, -1), salida[i].view(1))
                except KeyboardInterrupt:
                    print('\nEntrenamiento interrumpido. Guardando progreso...')
                    break
            loss.backward()  # Backward pass after accumulating loss
            optimizador.step()  # Update optimizer
            loss_total += loss.item()  
        if (epoca + 1) % 200 == 0:  
            print(f'Época {epoca+1}/{n_epocas}, Loss: {loss_total/len(entradas):.4f}')

    # Guardar el modelo después del entrenamiento
    torch.save(modelo, modelo_path)
    print("Modelo guardado.")

# Función para predecir la respuesta
def predecir(saludo):
    with torch.no_grad():
        entrada = texto_a_tensor(saludo.lower(), max_length).to(device)  # Mover a la GPU
        hidden = modelo.init_hidden()
        salida_texto = ""
        
        # Generar respuesta
        for i in range(max_length):
            output, hidden = modelo(entrada[i].view(1).to(device), hidden)
            valor_predicho = output.argmax().item()
            if valor_predicho == 0:  
                break
            salida_texto += chr(valor_predicho)
        
        return salida_texto

def mostrar_ayuda():
    print("\n=== Comandos disponibles ===")
    print("/help - Muestra esta ayuda")
    print("/exit - Salir del chat")
    print("========================\n")

def chat():
    print("\n=== Bienvenido al Chat con AI ===")
    print("Escribe '/help' para ver los comandos disponibles")
    print("Puedes escribir en español o inglés")
    print("============================\n")
    
    while True:
        usuario_input = input("Tú: ").strip()
        
        if usuario_input.lower() == "/exit":
            print("\n¡Hasta luego! Gracias por chatear.")
            break
        elif usuario_input.lower() == "/help":
            mostrar_ayuda()
        elif usuario_input:
            respuesta = predecir(usuario_input)
            print(f"Bot: {respuesta}\n")
        else:
            print("Bot: Por favor, escribe algo o usa '/help' para ver los comandos disponibles.\n")

if __name__ == "__main__":
    try:
        chat()
    except KeyboardInterrupt:
        print('\n¡Hasta luego! Gracias por chatear.')