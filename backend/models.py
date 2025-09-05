from pydantic import BaseModel,EmailStr,Field
from typing import List

class FamilyMember(BaseModel):
    name:str=Field(...,min_length=1)
    email:EmailStr=Field(...,description="Valid email address is required")
    mobno:str=Field(...,min_length=10)
    age:int=Field(...,gt=0)
    relation:str=Field(...,min_length=1)
    address:str=Field(...,min_length=10)
    video_urls:str=Field(...,description="URL of the video")

class User(BaseModel):
    name: str=Field(...,min_length=1)
    email: EmailStr=Field(...,description="Valid email address is required")
    mobno: str=Field(...,min_length=10)
    age: int=Field(...,gt=0)
    address:str=Field(...,min_length=10)
    no_of_family_members:int=Field(...,gt=0)
    family_member:List[FamilyMember]=Field(...,description="List of family members")
    





