import pypyodbc as odbc # pip install pypyodbc

''' Note on following: 
        DRIVER_NAME can be name whatever you want 
        SERVER_NAME: Server name can be found by the following
            1. Run your command line: In command line do the following
                a. sqlcmd (this will run sql manager command line, this was installed with sql server)
                b. 1> SELECT @@SERVERNAME AS 'MSSQLSERVER'; 
                c. 2> GO
                d. Output: DESKTOP-5QD580I (SERVER_NAME)
                e. 1> CREATE DATABASE testDB; (this will create a database, testDB can be whatever)
                f. 2> GO
                g. Output: (Nothing you'll return back to 1> )
                h. 1> exit()
            '''

DRIVER_NAME = 'SQL SERVER'
SERVER_NAME = 'DESKTOP-5QD580I' # Need to find your server name by using sqlcmd 
DATABASE_NAME = 'testDB'

connection_string = f"""
    DRIVER={{{DRIVER_NAME}}};
    SERVER={SERVER_NAME};
    DATABASE={DATABASE_NAME};
    Trust_Connection=yes;

"""
#    uid=<username>;
#    pwd=<password>;

conn = odbc.connect(connection_string)
print(conn)

''' Error Meanings:
    Server Name is incoorect: pypyodbc.DatabaseError: ('08001', '[08001] [Microsoft][ODBC SQL Server Driver]
        [DBNETLIB]SQL Server does not exist or access denied.')
    Database does not exit: pypyodbc.OperationalError: ('HYT00', '[HYT00] [Microsoft][ODBC SQL Server Driver]
        Login timeout expired') '''
