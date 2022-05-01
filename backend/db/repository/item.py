from sqlalchemy.orm import Session
from db.models.item import Item
from db.session import SessionLocal


def create_new_item(item, db: Session):
    item_object = Item(**item.dict())
    db.add(item_object)
    db.commit()
    db.refresh(item_object)
    return item_object


def list_item(db: Session):
    item = db.query(Item).all()
    return item






