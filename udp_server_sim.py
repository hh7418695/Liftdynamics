#!/usr/bin/env python3
import socket
import json
import sys

HOST = '127.0.0.1'
PORT = 12312

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
try:
    sock.bind((HOST, PORT))
except Exception as e:
    print(f"Failed to bind {HOST}:{PORT}: {e}")
    sys.exit(1)

print(f"UDP simulator listening on {HOST}:{PORT}")
print("Press Ctrl+C to stop")

try:
    while True:
        data, addr = sock.recvfrom(2048)
        try:
            s = data.decode('utf-8', errors='ignore')
        except:
            s = str(data)
        print(f"Received from {addr}: {s}")
        force = [0.0, 0.0, 0.0]
        try:
            payload = json.loads(s)
            pos = payload.get('position') or payload.get('pos') or [0.0, 0.0, 0.0]
            # If position is provided, return a simple spring-like force toward origin
            if isinstance(pos, (list, tuple)) and len(pos) >= 3:
                k = 0.02
                force = [-k * float(pos[0]), -k * float(pos[1]), -k * float(pos[2])]
        except Exception as e:
            print(f"Failed to parse JSON: {e}")
        reply = json.dumps({"force": force})
        sock.sendto(reply.encode('utf-8'), addr)
        print(f"Replied to {addr}: {reply}\n")
except KeyboardInterrupt:
    print("UDP simulator stopped by user")
finally:
    sock.close()
