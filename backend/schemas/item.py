from typing import Optional
from pydantic import BaseModel
from datetime import date, datetime


# shared properties
class ItemBase(BaseModel):
    item_label: Optional[str] = "vodka"
    item_probability: Optional[str] = "0.89"
    item_price: Optional[str] = None
    item_image: Optional[str] = None
    #date_query: Optional[date] = datetime.now().date()


# this will be used to validate data while creating a Job
class ItemCreate(ItemBase):
    item_label: str
    item_probability: Optional[str]
    item_price: Optional[str]
    item_image: Optional[str]
    #date_query: Optional[date]


# this will be used to format the response to not to have id,owner_id etc
class ShowProduct(ItemBase):
    item_label: str
    item_probability: str
    item_price: str
    item_image: Optional[str]
    #date_created: date

    class Config():  # to convert non dict obj to json
        orm_mode = True