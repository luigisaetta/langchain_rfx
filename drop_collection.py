"""
drop a collection

This utility can be used to drop a collection. 
It drops a table in the current schema. Use with caution.

The table name must be in correct capitalization.
"""

import argparse
import oracledb

# we're using directly drop_table_purge
from langchain_community.vectorstores.oraclevs import drop_table_purge, _table_exists

from utils import get_console_logger

from config_private import DB_USER, DB_PWD, DB_HOST_IP, DB_SERVICE

#
# Main
#
logger = get_console_logger()

# handling input
parser = argparse.ArgumentParser(description="Utility to drop collections.")

parser.add_argument("collection_name", type=str, help="Collection name.")

args = parser.parse_args()

# a collection is implemented as a table in the DB
table_name = args.collection_name

dsn = f"{DB_HOST_IP}:1521/{DB_SERVICE}"

connection = oracledb.connect(user=DB_USER, password=DB_PWD, dsn=dsn)

logger.info("")

if _table_exists(connection, table_name):
    drop_table_purge(connection, table_name)

    logger.info("Table %s dropped!", table_name)
    logger.info("")
else:
    logger.info("Table %s does not exist!", table_name)
    logger.info("")
