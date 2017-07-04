import pymongo
import pymongo.errors


def get_mongodb_client(host, port):
    """
    Establish connection to MongoDB.
    """
    try:
        connection = pymongo.MongoClient(host, port)
    except pymongo.errors.ConnectionFailure:
        raise Exception(
            "Need a MongoDB server running on {}, port {}".format(host, port))
    return connection


def get_db_entries(host, port, db_name, collection_name, projection=None):
    """
    Establish connection to MongoDB.
    """
    with get_mongodb_client(host, port) as connection:
        db = connection[db_name]
        collection = db[collection_name]
        entries = [x for x in collection.find(projection=projection)]
        return entries
