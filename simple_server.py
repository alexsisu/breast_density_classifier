import aiohttp
import logging.handlers
from aiohttp import web
import sys

from multidict import MultiDict
from density_model_torch import prepare_image_for_processing, inference


async def store_image_handler(request):
    # WARNING: don't do that if you plan to receive large files!
    data = await request.post()

    print("BEGIN")
    contained_data = data['file']

    filename = contained_data.filename
    print("retrieved ", filename)

    all_content = contained_data.file

    print("retrieved 2", all_content)
    all_content = all_content.read()

    print("retrieved 3", all_content)

    with open(filename, "wb") as o:
        o.write(all_content)

    print("preapre image for processing")
    prepare_image_for_processing(filename)
    full_path_input_image = filename

    destination_folder = prepare_image_for_processing(full_path_input_image)
    image_path = destination_folder
    print("getting the destination folder", image_path)
    parameters_ = {
        "model_type": "cnn",
        "bins_histogram": 50,
        "model_path": "saved_models/BreastDensity_BaselineBreastModel/model.p",
        "device_type": "cpu",
        "image_path": image_path,
    }

    str_result = inference(parameters_)

    return web.Response(body="")


async def index(request):
    return web.FileResponse('./index.html')


app = web.Application(client_max_size=100 * (1024 ** 2))
app.router.add_post('/upload_image', store_image_handler)
app.router.add_get("/", index)
# app.router.add_static("/x",path="static",name="index.html")

web.run_app(app, host="0.0.0.0", port=15432)
