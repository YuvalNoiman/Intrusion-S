import socket
#import threading
from multiprocessing import Process
from db_controller import connect_to_db

def handle_client(client_socket, client_address):
    print(f"Accepted connection from {client_address}")
    data = client_socket.recv(1024)
    message = data.decode()
    print(f"Received from {client_address}: {message}")
    #client_socket.send(b"Message received")
    #database.set_record_by_list(message.split(","))
	
    client_socket.close()
    return f"{message}"

def start_server():
    database = connect_to_db() 
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    host = socket.gethostname()
    server_socket.bind((host, 8080))
    server_socket.listen(5)

    print("Server listening on " + host + ":8080")
    number = 0
    #messages = []
    while True:
        client_socket, client_address = server_socket.accept()
        #thread = threading.Thread(target=handle_client, args=(client_socket, client_address, database))
        #thread.start()
        #p = Process(target=handle_client, args=(client_socket, client_address))
        message = handle_client(client_socket, client_address)
        database.set_record_by_list(message.split(","))
        '''
        messages.append(p)
        p.start()
        number += 1
        if number % 2 == 0:
                for message in messages:
                        database.set_record_by_list(message.get().split(","))
                        messages.clear() 
        '''


if __name__ == "__main__":
    start_server()