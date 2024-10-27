from flask import Flask, jsonify
import pyodbc
import os

app = Flask(__name__)
port = int(os.environ.get("PORT", 3000))

# MS SQL Server configuration
config = {
    'server': 'your_server_name_or_ip',
    'database': 'your_database_name',
    'username': 'your_username',
    'password': 'your_password',
    'driver': '{ODBC Driver 17 for SQL Server}',
}

# Connect to MS SQL Server
def connect_to_db():
    try:
        conn = pyodbc.connect(
            f"DRIVER={config['driver']};"
            f"SERVER={config['server']};"
            f"DATABASE={config['database']};"
            f"UID={config['username']};"
            f"PWD={config['password']}"
        )
        print('Connected to MS SQL Server')
        return conn
    except Exception as e:
        print('Error connecting to MS SQL Server:', e)
        return None

# API endpoint to fetch status
@app.route('/api/status', methods=['GET'])
def get_status():
    conn = connect_to_db()
    if conn:
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT TOP 1 * FROM status ORDER BY timestamp DESC")
            row = cursor.fetchone()
            if row:
                return jsonify(dict(zip([column[0] for column in cursor.description], row))), 200
            else:
                return jsonify({"error": "No data found"}), 404
        except Exception as e:
            print('Error executing query:', e)
            return jsonify({"error": "Internal server error"}), 500
        finally:
            cursor.close()
            conn.close()
    else:
        return jsonify({"error": "Database connection failed"}), 500

# Serve static files from the 'public' directory
@app.route('/<path:path>')
def static_files(path):
    return app.send_static_file(path)

# Start the server
if __name__ == '__main__':
    app.run(port=port)