import socket
#import threading
from multiprocessing import Process

def handle_client(client_socket, client_address):
    print(f"Accepted connection from {client_address}")
    while True:
        data = client_socket.recv(1024)
        if not data:
            break
        print(f"Received from {client_address}: {data.decode()}")
        client_socket.send(b"Message received")
    client_socket.close()

def start_server():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(('localhost', 8080))
    server_socket.listen(5)

    print("Server listening on localhost:8080")

    while True:
        client_socket, client_address = server_socket.accept()
        #thread = threading.Thread(target=handle_client, args=(client_socket, client_address))
        #thread.start()
        p = Process(target=handle_client, args=(client_socket, client_address))
        p.start()

if __name__ == "__main__":
    start_server()