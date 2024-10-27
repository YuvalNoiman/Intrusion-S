# Python Server Setup Guide

This guide provides step-by-step instructions on how to set up a Python server that connects to a Microsoft SQL Server database. 

## Table of Contents
- [Prerequisites](#prerequisites)
- [Installation Steps](#installation-steps)
- [Running the Server](#running-the-server)
- [Testing the Server](#testing-the-server)
- [Troubleshooting](#troubleshooting)

## Prerequisites

Before you begin, ensure you have the following installed on your machine:

1. **Python**: Download and install Python from the [official Python website](https://www.python.org/downloads/). Ensure you check the box to add Python to your PATH during installation.
   
2. **Microsoft SQL Server**: If you don't have SQL Server installed, download it from the [official Microsoft SQL Server website](https://www.microsoft.com/en-us/sql-server/sql-server-downloads). You may also want to install SQL Server Management Studio (SSMS) for easier database management.

3. **ODBC Driver**: Install the appropriate ODBC driver for SQL Server. You can download it from the [Microsoft ODBC Driver for SQL Server page](https://docs.microsoft.com/en-us/sql/connect/odbc/download-odbc-driver-for-sql-server).

## Installation Steps

1. **Set up Enviorment** 
  ```python -m venv venv

2. **Activate enviorment**
  ```venv\Scripts\activate

3. **Install requirements.txt**
  ```pip install -r requirements.txt

3. **Start Server**
  ```python server.py