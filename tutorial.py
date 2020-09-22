"""This is a tutorial from the website https://fastapi.tiangolo.com/tutorial/first-steps"""

from fastapi import FastAPI, Query, Path, Body, Cookie, File, UploadFile, Depends, Header, HTTPException
from fastapi.security import OAuth2PasswordBearer
import numpy as np
from enum import Enum
from typing import Optional, List, Set
from pydantic import BaseModel, Field
from uuid import UUID
from datetime import datetime, time, timedelta
from fastapi.encoders import jsonable_encoder

class ModelName(str, Enum):
    alexNet = "alexNet"
    resNet = "resNet"
    leNet = "leNet"

class Item(BaseModel):
    name : str
    description : Optional[str] = None
    price : float
    tax : Optional[float] = None
    tag : Set[str] = set()


class User(BaseModel):
    username: str
    full_name: Optional[str] = Field(...,
    title="Full Name of the respondent",
    max_length=20,
    example = "Bahadur")
    dob : Optional[str] = Field(..., regex="^(0[1-9]|1[012])[-/.](0[1-9]|[12][0-9]|3[01])[-/.](19|20)\\d\\d$")

class UserOut(BaseModel):
    username: str
    dob : Optional[str] = Field(..., regex="^(0[1-9]|1[012])[-/.](0[1-9]|[12][0-9]|3[01])[-/.](19|20)\\d\\d$")


##Dependency functions for checking token and keys
async def verify_token(x_token: str = Header(...)):
    """These dependencies will be executed/solved the same way normal dependencies.
 But their value (if they return any) won't be passed to your path operation function"""
    if x_token != "fake-super-secret-token":
        raise HTTPException(status_code=400, detail="X-Token header invalid")


async def verify_key(x_key: str = Header(...)):
    if x_key != "fake-super-secret-key":
        raise HTTPException(status_code=400, detail="X-Key header invalid")
    #return x_key


app = FastAPI()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

fake_items_db = [{"item_name": "Foo"}, {"item_name": "Bar"}, {"item_name": "Baz"}]

@app.get("/")
async def root():
    return {"message": "Tutorial 101"}

@app.get("/calc/")
async def calc(create_seq: bool = False):    
    if not create_seq:
        return {"seq": 1}
    else:
        return {"seq": np.arange(1,10, 1)}

@app.get("/model/{model_name}")
async def get_model(model_name : ModelName):
    if model_name == ModelName.alexNet:
        return {"model_name": model_name, "message": "Deep Learning FTW!"}

    if model_name.value == "leNet":
        return {"model_name": model_name, "message": "LeCNN all the images"}

    return {"model_name": model_name, "message": "Have some residuals"}



@app.get("/items/")
async def read_item(skip: int = 0, limit: int = 10):
    return fake_items_db[skip : skip + limit]

@app.get("/items/{item_id}")
async def read_items(item_id : str, q : Optional[str] = None, logic : bool = False):
    item = {"item_id": item_id}
    if q:
        item.update({"q": q})
    if not logic:
        item.update(
            {"description": "This is an amazing item that has a long description"}
        )
    return item

@app.get("/items/test/{test_id}")
async def read_user_item(
    test_id: str, needy: str, skip: int = 0, limit: Optional[int] = None
):
    item = {"test_id": test_id, "needy": needy, "skip": skip, "limit": limit}
    return 

@app.post("/items_class/")
async def create_item(item_new: Item):
    item_dict = item_new.dict()
    if item_new.tax:
        total_price = item_new.price + item_new.tax
    item_dict.update({"Total Price" : total_price})
    return item_dict


@app.put("/items_all/{item_id}")
async def create_item(item_id: int, item: Item, 
q: Optional[str] = Query(None, min_length=3, max_length=50, regex="^fixedquery$")):
    result = {"item_id": item_id, **item.dict()}
    if q:
        result.update({"q": q})
    return result

@app.get("/sumIt/", tags=["Experiments"])
async def sum_it(numbers2 : Optional[List[float]] = Query(...,
 title = "List of integers",
 description="This will sum up all integers in the list",
 ge = 0,
 le = 1e3)):
    return {"Sum of Num": sum(numbers2)}

@app.put("/items/{item_id}", tags=["Class"])
async def update_item(
    *,
    item_id: int = Path(..., title="The ID of the item to get", ge=0, le=1000),
    q: Optional[str] = None,
    item: Optional[Item] = None,
    user: User = Body(..., embed=True),
    importance : int = Cookie(123),   ##stores a cookie query parameter
    body_part : str = Body(..., embed=True)
):
    results = {"item_id": item_id}
    if q:
        results.update({"q": q})
    if item:
        results.update({"item": item})
    if user:
        results.update({"User":user})
    if body_part:
        results.update({"body_part":body_part})
    return results

@app.put("/items_duration/{item_id}", tags=["Class"])
async def read_items(
    item_id: UUID,
    start_datetime: Optional[datetime] = Body(None),
    end_datetime: Optional[datetime] = Body(None),
    repeat_at: Optional[time] = Body(None),
    process_after: Optional[timedelta] = Body(None),
):
    start_process = start_datetime + process_after
    duration = end_datetime - start_process
    ret_mod =  {
        "item_id": item_id,
        "start_datetime": start_datetime,
        "end_datetime": end_datetime,
        "repeat_at": repeat_at,
        "process_after": process_after,
        "start_process": start_process,
        "duration": duration,
    }
    return jsonable_encoder(ret_mod)

@app.post("/files/")
async def create_files(files: List[bytes] = File(...)):
    return {"file_sizes": [len(file) for file in files]}


##Dependencies
"""You need to write a function and the embed it in the api call using FastApIs Depends function"""

async def common_parameters(q: Optional[str] = None, skip: int = 0, limit: int = 100):
    return {"q": q, "skip": skip, "limit": limit}


@app.get("/items_depends/")
async def read_items(commons: dict = Depends(common_parameters)):
    return commons

"""The function below would only execute if the dependencies are met.
That is, the verify_token and verify_key functions do not raise HTTPExceptions """
@app.get("/items_dependencies/", dependencies=[Depends(verify_token), Depends(verify_key)])
async def read_items():
    return [{"item": "Foo"}, {"item": "Bar"}]

@app.get("/items_auth/")
async def read_items(token: str = Depends(oauth2_scheme)):
    """This function should ask for authorization"""
    return {"token": token}    