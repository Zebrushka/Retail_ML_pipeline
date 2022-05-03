from sqlalchemy.orm import Session
from db.models.item import Item
from schemas.item import ItemCreate


def create_new_item(item:ItemCreate, db: Session):
    #item_object = Item(**item.dict())
    item_object = Item(**item)
    db.add(item_object)
    db.commit()
    db.refresh(item_object)
    return item_object


def list_item(db: Session):
    item = db.query(Item).all()
    return item






