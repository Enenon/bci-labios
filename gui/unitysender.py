from dependencias import *

class UnitySender:
    def __init__(self, port=PORTA_UNITY, udp_port=PORTA_UDP_UNITY):
        self.port = port; self.udp_port = udp_port; self.context = zmq.Context(); self.socket = self.context.socket(zmq.PUB)
        try: self.socket.setsockopt(zmq.CONFLATE, 1)
        except Exception: pass 
        try: self.socket.bind(f"tcp://*:{port}")
        except Exception as e: print(f"Erro ao ligar a porta ZMQ: {e}") 

        self.local_ip = self.get_local_ip()
        self.send_ip_udp_broadcast()
        self.queue = []; self.lock = threading.Lock(); self.running = True
        self.thread = threading.Thread(target=self.sender_loop, daemon=True); self.thread.start()

    def get_local_ip(self):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.connect(("8.8.8.8", 80)); ip = s.getsockname()[0]; s.close(); return ip
        except Exception: return "127.0.0.1"

    def send_ip_udp_broadcast(self):
        def _broadcast():
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP); s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            while self.running:
                try: s.sendto(self.local_ip.encode(), ('<broadcast>', self.udp_port)); sleep(1.0) 
                except Exception: pass
            s.close()
        threading.Thread(target=_broadcast, daemon=True).start()

    def send(self, msg):
        with self.lock: self.queue.append(msg)

    def sender_loop(self):
        last_ping = time()
        while self.running:
            with self.lock:
                if self.queue:
                    msg_to_send = str(self.queue.pop(0))
                    try: self.socket.send_string(msg_to_send)
                    except Exception: pass
            if time() - last_ping > 1.0:
                try: self.socket.send_string("CONNECTED")
                except Exception: pass
                last_ping = time()
            sleep(0.01) 

    def stop(self):
        self.running = False
        try: self.socket.close(); self.context.term()
        except Exception: pass
