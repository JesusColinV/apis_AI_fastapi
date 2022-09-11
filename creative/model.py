from uuid import UUID
from sqlalchemy import Column
from sqlalchemy.sql.sqltypes import String, Integer, Float, Boolean
from database.base_class import Base
from fastapi_utils.guid_type import GUID
   
class TransactionModel(Base):
    __tablename__ = 'transaction_table'
    id = Column(GUID, primary_key=True, index = True)
    cost= Column(Float)
    start= Column(Float)
    end= Column(Float)
    done= Column(Boolean, nullable = False)
    
#SqlAlchemyBase.metadata.create_all(engine)
