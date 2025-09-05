import motor.motor_asyncio
import os
from dotenv import load_dotenv

load_dotenv()

MONGO_URL = os.getenv("mongodbURL")
client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URL)
db = client["nyaya_samiti"]   # database name
users_collection = db["users"]