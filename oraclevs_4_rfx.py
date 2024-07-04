"""
Extensions to Oracle VS
"""

import os
from oracledb import Connection

from langchain_community.vectorstores.oraclevs import OracleVS

from utils import get_console_logger, debug_bool

logger = get_console_logger()

VERBOSE = debug_bool(os.environ.get("DEBUG", "False"))


class OracleVS4RFX(OracleVS):
    """
    This class extends OracleVS and has been defined to add utility methods
    """

    @classmethod
    def list_collections(cls, connection: Connection):
        """
        return a list of all collections (tables) with a type vector
        in the schema in use
        """

        query = """
                SELECT DISTINCT table_name
                FROM user_tab_columns
                WHERE data_type = 'VECTOR'
                ORDER by table_name ASC
                """

        with connection.cursor() as cursor:
            cursor.execute(query)

            rows = cursor.fetchall()

            list_collections = []
            for row in rows:
                list_collections.append(row[0])

        return list_collections

    @classmethod
    def list_books_in_collection(cls, connection: Connection, collection_name: str):
        """ "
        get the list of books name in the collection
        taken from metadata
        expect metadata contains source
        """
        query = f"""
                SELECT DISTINCT json_value(METADATA, '$.source') AS books
                FROM {collection_name}
                ORDER by books ASC
                """
        with connection.cursor() as cursor:
            cursor.execute(query)

            rows = cursor.fetchall()

            list_books = []
            for row in rows:
                list_books.append(row[0])

        return list_books

    @classmethod
    def delete_documents(
        cls, connection: Connection, collection_name: str, doc_names: list
    ):
        """
        doc_names: list of names of docs to drop
        """
        for doc_name in doc_names:
            sql = f"""
                  DELETE FROM {collection_name}
                  WHERE json_value(METADATA, '$.source') = :doc
                  """

            if VERBOSE:
                logger.info("Drop %s", doc_name)
                logger.info(sql)

            cur = connection.cursor()

            cur.execute(sql, [doc_name])

            cur.close()

        connection.commit()
