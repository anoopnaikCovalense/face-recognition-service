import mysql.connector
from mysql.connector import errorcode

class Database:

    def __init__(self):
        # init mysql connection
        self.connection = mysql.connector.connect(
            host="localhost",
            user="root",
            passwd="12345678",
            database="crm24012019"
        )

    def query(self, q, arg=()):
        cursor = self.connection.cursor(buffered=True)
        cursor.execute(q, arg)
        results = cursor.fetchall()
        cursor.close()
        return results

    def insert(self, q, arg=()):
        cursor = self.connection.cursor(buffered=True)
        cursor.execute(q, arg)
        self.connection.commit()
        result = cursor.lastrowid
        cursor.close()
        return result

    def select(self, q, arg=()):
        cursor = self.connection.cursor(buffered=True)
        cursor.execute(q, arg)
        return cursor.fetchall()

    def delete(self, q, arg=()):
        cursor = self.connection.cursor(buffered=True)
        result = cursor.execute(q, arg)
        self.connection.commit()
        return result