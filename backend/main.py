from fastapi import FastAPI,HTTPException
from database import client, users_collection
from models import User

app=FastAPI()

@app.on_event("startup")
async def startup_db_check():
    try:
        await client.admin.command("ping")
        print("Database connection successful")
    except Exception as e:
        print("Database connection failed")
        print(e)

@app.get("/")
async def root():
    return {"message": "Welcome to landing page"}

@app.post("/users/")
async def create_user(user: User):
    # Logic to create a new user
    user_dict=user.dict()
    result=await users_collection.insert_one(user_dict)

    return {"id":str(result.inserted_id),"message": "User created successfully", "user": user}
