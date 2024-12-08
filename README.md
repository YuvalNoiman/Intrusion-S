By Tung Le, Aaron Tang, Yuval Noiman, Nathan Bupte, Omar Ramirez

# Intrusion-S

To get working on client:

First download files from github.

Second install python and python-tk using sudo apt-get install python-tk and sudo apt-get install python

Third install pip using sudo apt-get install pip and install python modules using pip install -r requirements.txt

Fourth sudo apt-get install libpcap-dev to get the extractor to work.

Fifth you will need to download the extractor from here https://github.com/AI-IDS/kdd99_feature_extractor and build it. You might need to install cmake using sudo apt-get install cmake. Once the files are built using their instructions move the folder with the .exe to built_kdd99extractor replacing the current contents.

Sixth run the program using "python -m client".

To get working on server:

First download files from github.

Second install python using sudo apt-get install python

Third install pip using sudo apt-get install pip and install python modules using pip install -r requirements.txt

Fourth you will need to set install required database, database drivers, and database managers. Oonce that is done you will have run db.py and create a database and table using db_controller.py.

Fifth you would run the server using "python -m server" and website using "python -m app".

On both:

All code should work on internet though you would need to use portforwarding and all code should work on the cloud but would require different set up.
