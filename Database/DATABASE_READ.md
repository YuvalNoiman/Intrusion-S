Connect to SQL Server using python:
1. 

Installation instructions:
1. Go to ' https://www.microsoft.com/en-us/sql-server/sql-server-downloads?msockid=16228e619a34674e1e2a9aeb9bda6606' and download SQL Server 2022 Developer from webpage.
2. Run the SQL2022-SSEI-Dev.exe file and click Custom.
3. Set your media location and install.
4. After installation SQL Server INstallation Center will open or open by "SETUP.EXT" in Developer_ENU folder created
5. In the SQL Server INstallation Center click "Installation" on the side.
6. Click on "New SQL Server standalone installation or add features to an existing installation"
7. Next Specify a free editon: Developer then Next.
8. Accept the license agreement. Click "I accept the license term". Then Next.
9. For "microsoft update" section. It is up to you if you wish to click on "use microsfot update ...". Next
10. For "Installation Rules" ignore Windows Firewall warming. Click Next. (Standard warning)
11. If using planning to use "Azure Extension for SQL Server" proceed with checked box fill out. Either way Next
12. Imporant: Under Feature Selection click on "Database Enginee Services" 
13. Still under "Feature Selection" edit instance root directory and shared feature directory if required.
14. Under Instance configurations, click on "Default instance" if sql server version not already installed. Next
15. Under "Server COnfiguration" section. Just click next unless you know what you're doing.
16. Under "Database Engine COnfiguration" leave "Windows authnentication mode" alone. 
        Imporant: Click on "Add Current User" to add yourself as administrator.
        Note: Here you can edit space and memory usage here. Under "TempDB" and "Memory".
17. After click "Install" to install with configured settings.
18. Installation complete and "close"

Reference:
Installation:
https://www.youtube.com/watch?v=oKsYmoCHTtQ&t=790s&ab_channel=SQLServer101

https://www.youtube.com/watch?v=g69lFxZdcVQ&t=148s&ab_channel=JieJenn

Note on "basic" install option:
A basic installation of SQL Server includes the following:
Database Engine Services: The core of SQL Server that stores, processes, and secures data
Supporting components: The necessary components to support the Database Engine Services
Default server configurations: The default settings for the server 
A basic installation is a good choice for users who want to quickly install SQL Server for small-scale deployments, testing, or development. 
Other components that can be installed with SQL Server include: 
SQL Server Replication: An optional component 
Client components: Includes backward compatibility components, connectivity components, management tools, software development kit, and SQL Server Books Online components 
SQL Server Management Studio (SSMS): An integrated environment for managing SQL Server and databases 
SQL Server can be installed using the SQL Server Installation Wizard

SQL Server info:
SSIS, SSAS, SSRS
https://www.youtube.com/watch?v=I_Ae3suaL-U&ab_channel=.NETInterviewPreparationvideos