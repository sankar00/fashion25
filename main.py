from fastapi import FastAPI, UploadFile
from starlette.responses import Response
from pydantic import BaseModel
from prediction import preprocess_image, predict_category
import uvicorn

app = FastAPI()

class Item(BaseModel):
    file: UploadFile

@app.get("/")
def index():
    return {"hello": "FastAPI"}

@app.get('/{name}')
def get_name(name: str):
    return {'message': f'hello, {name}'}

@app.post("/file/predict")
async def upload_file(item: UploadFile):
    file_extension = item.filename.split(".")[-1]
    if file_extension.lower() != "jpg":
        return {"error": "Invalid file format. Only JPG files are supported."}
    
    contents = await item.read()
    predicted_category = predict_category(contents)
    return {"predicted_category": predicted_category, "filename": item.filename}

@app.post("/vector_image")
async def vector_image(file: UploadFile):
    contents = await file.read()

    return Response(contents, media_type="image/png")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)

