from flask import Flask, render_template, jsonify
import pyodbc
from db_controller import connect_to_db

DATABASE = connect_to_db() 
app = Flask(__name__)

# Fetch data from the database
def fetch_data(query):
    '''
    conn = return_conn
    if not conn:
        print("Database connection failed")
        return None

    try:
        cursor = conn.cursor()
        cursor.execute(query)
        results = cursor.fetchall()
        cursor.close()
        return results
    except Exception as e:
        print(f"Error executing query: {e}")
        return None
    finally:
        conn.close()
    '''

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/history')
def history():
    return render_template('history.html')

# Route for the settings page
@app.route('/settings')
def settings():
    return render_template('settings.html')

# Endpoint to get the most recent client information
@app.route('/api/client', methods=['GET'])
def get_most_recent_client():
    query = "SELECT TOP 1 Client, IsAttack, TimeStamp FROM Client_Attack_Status_Table ORDER BY TimeStamp DESC"
    try:
        result = DATABASE.get_query(query)
        if result:
            client_data = {
                'client': result[0][0],  # Client name
                'isAttack': result[0][1],  # Attack status (boolean)
                'timestamp': result[0][2],  # Timestamp of the record
            }
            return jsonify(client_data)
        else:
            return jsonify({'client': 'No client information available'}), 404
    except Exception as e:
        print(f"Error fetching client information: {e}")
        return jsonify({"error": "Internal server error"}), 500

# API endpoint to fetch historical data
@app.route('/api/history', methods=['GET'])
def get_history():
    query = "SELECT Client, IsAttack, TimeStamp FROM Client_Attack_Status_Table ORDER BY TimeStamp DESC"
    try:
        rows = DATABASE.get_query(query)
        if rows:
            # Convert fetched rows into dictionaries
            history = [
                {
                    'client': row[0],
                    'isAttack': row[1],
                    'timestamp': row[2]
                } for row in rows
            ]
            return jsonify(history), 200
        else:
            return jsonify({"error": "No data found"}), 404
    except Exception as e:
        print('Error fetching historical data:', e)
        return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    app.run(port=3000)
