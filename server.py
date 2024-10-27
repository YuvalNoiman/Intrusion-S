from flask import Flask, jsonify
import pyodbc
import os

app = Flask(__name__)

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
        return conn
    except Exception as e:
        print('Error connecting to MS SQL Server:', e)
        return None

# API endpoint to fetch historical data
@app.route('/api/history', methods=['GET'])
def get_history():
    conn = connect_to_db()
    if conn:
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT * FROM status ORDER BY timestamp DESC")
            rows = cursor.fetchall()
            history = [dict(zip([column[0] for column in cursor.description], row)) for row in rows]
            return jsonify(history), 200
        except Exception as e:
            print('Error executing query:', e)
            return jsonify({"error": "Internal server error"}), 500
        finally:
            cursor.close()
            conn.close()
    else:
        return jsonify({"error": "Database connection failed"}), 500

if __name__ == '__main__':
    app.run(port=3000)