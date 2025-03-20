import pymongo

# Connection string
connection_string = "mongodb://54.169.96.208:27017/nawabii"

# Create a connection
client = pymongo.MongoClient(connection_string)

# Access the database
db = client.nawabii  # or client["nawabii"]

# Test the connection
try:
    # The ismaster command is cheap and does not require auth
    client.admin.command('ismaster')
    print("MongoDB connection successful")
except pymongo.errors.ConnectionFailure:
    print("MongoDB connection failed")
# List all collections in that database
print("\nCollections in 'nawabii' database:")
for collection_name in db.list_collection_names():
    print(f"- {collection_name}")