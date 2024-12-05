import pyodbc as odbc  # Ensure pyodbc is installed

# Configuration
DRIVER_NAME = 'ODBC Driver 17 for SQL Server'  # Updated driver name
SERVER_NAME = 'Tree-1'  # Replace with your server name
SERVER PASSWORD: 'your_password',
DEFAULT_DATABASE = 'Client_Attack_Status'
DEFAULT_TABLE = 'Client_Attack_Status_Table'

# Connection string
connection_string = f"""
    DRIVER={{{DRIVER_NAME}}};
    SERVER={SERVER_NAME};
    DATABASE={DEFAULT_DATABASE};
    Trusted_Connection=yes;
"""
conn = odbc.connect(connection_string, autocommit=True)


class DataBaseEngine:
    def __init__(self, connection):
        print("Star Engine Initialized:")
        self.__Connection = connection
        self.cursor = connection.cursor()
        self.cursor.execute(f"USE {DEFAULT_DATABASE}")
        print("Connected to Database:", DEFAULT_DATABASE)

    def set_database(self):
        print("START: SET_DATABASE()")
        set_database = input("Enter the database name to switch to: ")
        SQL_Query = f"USE {set_database}"
        self.cursor.execute(SQL_Query)
        print(f"Switched to database: {set_database}")

    def create_database(self):
        print('START: CREATE_DATABASE')
        create_database = input("Input the name of the database you wish to create: ")
        SQL_Query = f"CREATE DATABASE {create_database}"
        self.cursor.execute(SQL_Query)
        print(f"Database {create_database} created successfully.")

    def create_table(self):
        print('START: CREATE_TABLE')
        SQL_Query = f"USE {DEFAULT_DATABASE}"
        self.cursor.execute(SQL_Query)
        SQL_Query = f"""
        CREATE TABLE {DEFAULT_TABLE} (
            IsAttack BIT NOT NULL,
            Client VARCHAR(100) NOT NULL,
            TimeStamp DATETIME NOT NULL
        );
        """
        self.cursor.execute(SQL_Query)
        print(f"Table {DEFAULT_TABLE} created successfully in {DEFAULT_DATABASE}.")

    def set_record_by_list(self, incoming_list):
        print("START: SET_RECORD_BY_LIST")
        SQL_Query = f"USE {DEFAULT_DATABASE}"
        self.cursor.execute(SQL_Query)
        IsAttack, Client, TimeStamp = incoming_list
        SQL_Query = f"""
        INSERT INTO {DEFAULT_TABLE} (IsAttack, Client, TimeStamp)
        VALUES (?, ?, ?)
        """
        self.cursor.execute(SQL_Query, (IsAttack, Client, TimeStamp))
        print(f"Record inserted: {incoming_list}")

    def set_just_a_record_test(self):
        print("START: SET_JUST_A_RECORD_TEST")
        SQL_Query = f"USE {DEFAULT_DATABASE}"
        self.cursor.execute(SQL_Query)
        test_list = [1, "TestClient", "2024-12-04 12:00:00"]
        IsAttack, Client, TimeStamp = test_list
        SQL_Query = f"""
        INSERT INTO {DEFAULT_TABLE} (IsAttack, Client, TimeStamp)
        VALUES (?, ?, ?)
        """
        self.cursor.execute(SQL_Query, (IsAttack, Client, TimeStamp))
        print(f"Test record inserted: {test_list}")

    def get_record_by_list(self):
        print('START: GET_RECORD')
        SQL_Query = f"USE {DEFAULT_DATABASE}"
        self.cursor.execute(SQL_Query)
        SQL_Select = f"SELECT * FROM {DEFAULT_TABLE}"
        self.cursor.execute(SQL_Select)
        records = self.cursor.fetchall()
        returning_list = [list(record) for record in records]
        print(f"Retrieved records: {returning_list}")
        return returning_list


# Initialize the database engine
database = DataBaseEngine(conn)

# Interactive menu for operations
while True:
    print(
        """
        Options:
        1. set_database()
        2. create_database()
        3. create_table()
        4. set_record_by_list()
        5. get_record_by_list()
        6. set_just_a_record_test()
        0. exit()
        """
    )
    current_input = input("Enter your input/number: ")
    match current_input:
        case '1':
            database.set_database()
        case '2':
            database.create_database()
        case '3':
            database.create_table()
        case '4':
            incoming_junk_list = [1, "ExampleClient", "2024-12-04 14:30:00"]
            database.set_record_by_list(incoming_junk_list)
        case '5':
            returning_table = database.get_record_by_list()
            print("Records in table:", returning_table)
        case '6':
            database.set_just_a_record_test()
        case '0':
            print("Exiting program.")
            break
        case _:
            print("Invalid input. Please try again.")
