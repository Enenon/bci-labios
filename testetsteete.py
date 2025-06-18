import socket
import zmq
import keyboard
import time

def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception as e:
        print(f"Erro ao obter IP local: {e}")
        return "127.0.0.1"

def send_ip_UDP(ip):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    broadcast_port = 12346

    print(f"Enviando IP detectado: {ip}")
    try:
        sock.sendto(ip.encode(), ('<broadcast>', broadcast_port))
        print(f"Broadcast enviado: {ip}")
    finally:
        sock.close()

# ===============================================================
# Inicialização
local_ip = get_local_ip()
print(f"IP Local detectado: {local_ip}")
send_ip_UDP(local_ip)

# Configuração do socket ZMQ
context = zmq.Context()
socket_pub = context.socket(zmq.PUB)
socket_pub.bind("tcp://*:5555")

print("Use ← para LEFT, → para RIGHT. ESC para sair.")

# Flags de estado para saber quando soltar
left_pressed = False
right_pressed = False

try:
    while True:
        # LEFT HAND
        if keyboard.is_pressed('left'):
            if not left_pressed:
                socket_pub.send_string("LEFT_HAND_CLOSE")
                print("← Pressionado: LEFT_HAND_CLOSE")
                left_pressed = True
        else:
            if left_pressed:
                socket_pub.send_string("LEFT_HAND_OPEN")
                print("← Solto: LEFT_HAND_OPEN")
                left_pressed = False

        # RIGHT HAND
        if keyboard.is_pressed('right'):
            if not right_pressed:
                socket_pub.send_string("RIGHT_HAND_CLOSE")
                print("→ Pressionado: RIGHT_HAND_CLOSE")
                right_pressed = True
        else:
            if right_pressed:
                socket_pub.send_string("RIGHT_HAND_OPEN")
                print("→ Solto: RIGHT_HAND_OPEN")
                right_pressed = False

        # Encerrar
        if keyboard.is_pressed('esc'):
            print("ESC pressionado. Encerrando.")
            break

        time.sleep(0.05)

except KeyboardInterrupt:
    print("Interrompido.")
finally:
    socket_pub.close()
    context.term()
    print("Finalizado.")
