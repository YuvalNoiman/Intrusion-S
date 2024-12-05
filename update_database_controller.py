import pyodbc as odbc # pip install pypyodbc

DRIVER_NAME = 'ODBC Driver 17 for SQL Server'
SERVER_NAME = 'YU-HP' # Need to find your server name by using sqlcmd through the terminal 
DATABASE_NAME = 'tempdb' 
DEFAULT_DATABASE = 'Client_Attack_Status'
DEFAULT_TABLE = 'Client_Attack_Status_Table'

connection_string = f"""
    DRIVER={{{DRIVER_NAME}}};
    SERVER={SERVER_NAME};
    DATABASE={DATABASE_NAME};
    Trusted_Connection=yes;
"""
conn = odbc.connect(connection_string, autocommit=True)

def return_conn():
    conn = odbc.connect(connection_string, autocommit=True)
    return conn

def connect_to_db():
    conn = odbc.connect(connection_string, autocommit=True)
    database = DataBaseEngine(conn)
    return database

class DataBaseEngine():
    def __init__(self, connection):
        print("Star Engine:")
        self.__Connection = connection
        self.cursor = connection.cursor()
        #self.cursor.execute("USE Client_Attack_Status")
        print(self.__Connection)
    def set_database(self):
        print("START: SET_DATABASE()")
        set_database = input()
        SQL_Query = f"""
        USE {set_database}
        """
        self.cursor.execute(SQL_Query)
    def create_database(self):
        print('START: CREATE_DATABASE')
        create_database = input("Input the name of the Database you wish to create: ")
        SQL_Query = f"""
        CREATE DATABASE {create_database}
        """
        self.cursor.execute(SQL_Query)
    def create_table(self):
        print('START: CREATE_TABLE')
        SQL_Query = f"""
        USE {DEFAULT_DATABASE}
        """
        self.cursor.execute(SQL_Query)
        SQL_Query = """
        CREATE TABLE Client_Attack_Status_Table (
            IsAttack BIT,
            Client VARCHAR(100),
            TimeStamp VARCHAR(100),
        );
        """
        self.cursor.execute(SQL_Query)
    def set_record_by_list(self, Incoming_List):
        print("START: SET_RECORD_BY_LIST")
        SQL_Query = f"""
        USE {DEFAULT_DATABASE}
        """
        self.cursor.execute(SQL_Query)
        InputList = Incoming_List
        IsAttack = InputList[0]
        Client = InputList[1]
        TimeStamp = InputList[2]

        SQL_Query = f"""
        INSERT INTO {DEFAULT_TABLE} (IsAttack, Client, TimeStamp)
        VALUES (?, ?, ?)
        """        
        self.cursor.execute(SQL_Query, (IsAttack, Client, TimeStamp))     

    def set_just_a_record_test(self):
        print("START: SET_RECORD_BY_LIST")
        SQL_Query = f"""
        USE {DEFAULT_DATABASE}
        """
        self.cursor.execute(SQL_Query)

        thislist = [1, "stringg", "stringg"]
        IsAttack = thislist[0]
        Client = thislist[1]
        TimeStamp = thislist[2]

        SQL_Query = f"""
        INSERT INTO {DEFAULT_TABLE} (IsAttack, Client, TimeStamp)
        VALUES (?, ?, ?)
        """        
        self.cursor.execute(SQL_Query, (IsAttack, Client, TimeStamp))    

    def get_record_by_list(self):
        print('START: GET_RECORD')
        SQL_Query = f"""        
        USE {DEFAULT_DATABASE}
        """
        self.cursor.execute(SQL_Query)
        SQL_Select = f"SELECT * FROM {DEFAULT_TABLE}"
        self.cursor.execute(SQL_Select)

        records = self.cursor.fetchall()
        returning_list = []

        for record in records:
            returning_list.append(record)

        return returning_list

    def get_query(self, query):  
        SQL_Query = f"""        
        USE {DEFAULT_DATABASE}
        """
        self.cursor.execute(SQL_Query)
        self.cursor.execute(query)
        records = self.cursor.fetchall()
        return records

def main():
        database = DataBaseEngine(conn)

        while(True):
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
                    incoming_junk_list = thislist = [1, "stringg", "stringg"]
                    database.set_record_by_list(incoming_junk_list)
                case '5':
                    returning_table = database.get_record_by_list()
                    print(returning_table)
                case '6':
                    database.set_just_a_record_test()
                case '0':
                    break

if __name__ == "__main__":
        main()