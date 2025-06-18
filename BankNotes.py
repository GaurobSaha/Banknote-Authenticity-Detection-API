from pydantic import BaseModel, Field

class BankNote(BaseModel):
    variance:float
    skewness:float
    curtosis:float
    entropy:float