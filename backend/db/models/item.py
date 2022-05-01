from sqlalchemy import Column, Integer, String, Boolean, Date, ForeignKey

from db.base_class import Base

# TODO сделать сохранение даты, фото, метки, вероятности, цены

class Item(Base):
    id = Column(Integer, primary_key = True, index=True)
    label = Column(String, nullable=False)
    probability = Column(String)
    price = Column(String, nullable = False)
    image = Column(String, nullable=False)
    date_query = Column(Date)