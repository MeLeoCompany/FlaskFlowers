from app import app, img_model, img_model_sentence, es
from flask import render_template, redirect, url_for, request, send_file
from app.searchForm import SearchForm
from app.inputFileForm import InputFileForm
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
import torch
import torchvision
from torchvision import transforms as T
from .machin_learning import model_init, param_init
import elasticsearch
import os
from PIL import Image

INDEX_IM_EMBED = 'my-image-embeddings'

HOST = app.config['ELASTICSEARCH_HOST']
AUTH = (app.config['ELASTICSEARCH_USER'], app.config['ELASTICSEARCH_PASSWORD'])
HEADERS = {'Content-Type': 'application/json'}

TLS_VERIFY = app.config['VERIFY_TLS']

app_models = {}

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def my_model_init():
    url_model_dir = 'static/models/full_model.pth'
    project_root = os.path.abspath(os.path.dirname(__file__))
    url_model = os.path.join(project_root, url_model_dir)
    model_new = model_init().to(DEVICE)
    _, optimizer_new, _ = param_init(model_new)
    checkpoint = torch.load(url_model, map_location=DEVICE)
    model_new.load_state_dict(checkpoint['model_state_dict'])
    optimizer_new.load_state_dict(checkpoint['optimizer_state_dict'])
    model_new.eval()
    return model_new

my_model_bbox = my_model_init()


@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
@app.route('/similar_image', methods=['GET', 'POST'])
def similar_image():
    index_name = INDEX_IM_EMBED
    global my_model_bbox
    if not es.indices.exists(index=index_name):
        return render_template('similar_image.html', title='Similar image', index_name=index_name, missing_index=True)

    # is_model_up_and_running(INFER_MODEL_IM_SEARCH)
    # if app_models.get(INFER_MODEL_IM_SEARCH) == 'started':
    form = InputFileForm()
    if request.method == 'POST':
        if form.validate_on_submit():
            if request.files['file'].filename == '' and form.searchbox.data == '' or form.searchbox.data is None:
                return render_template('similar_image.html', title='Similar image', form=form,
                                       err='No file selected', model_up=True)

            if request.files['file'].filename != '':
                filename = secure_filename(form.file.data.filename)
                print(filename)
                url_dir = 'static/tmp-uploads/'
                upload_dir = 'app/' + url_dir
                upload_dir_exists = os.path.exists(upload_dir)
                if not upload_dir_exists:
                    # Create a new directory because it does not exist
                    os.makedirs(upload_dir)

                # physical file-dir path
                file_path = upload_dir + filename
                # Save the image
                form.file.data.save(upload_dir + filename)
                image = Image.open(file_path)
                embedding, cropped_image = image_embedding(image, img_model, my_model_bbox)
                # relative file path for URL
                url_path_file = url_dir + '_cropped_'+ filename
                cropped_image.save(upload_dir + '_cropped_'+ filename)
                sentence_data = ''
            else:
                sentence_data = form.searchbox.data
                embedding = sentence_embedding(sentence_data)
                url_path_file = ''

            # Execute KN search over the image dataset
            search_response = knn_search_images(embedding.tolist())

            # search_results = [
            #     {"fields":{
            #         "image_id":["6502ce1b82f2e1a00b405f40"],
            #         "image_name":["Малиновый Пудинг"],
            #         "article":["609446"],
            #         "relative_path":["img1004.jpg"],
            #         "_score":'90'
            #     }},
            #     {"fields":{
            #         "image_id":["6502ce1b82f2e1a00b405f41"],
            #         "image_name":["Восхитительные Пионы"],
            #         "article":["608989"],
            #         "relative_path":["img1021.jpg"],
            #         "_score":'80'
            #     }},
            # ]
            # search_response = {
            #     "hits":{"hits":search_results}
            # }

            # Cleanup uploaded file after not needed
            # if os.path.exists(file_path):
            #     os.remove(file_path)

            similar_hits = search_response['hits']['hits']

            return render_template('similar_image.html', title='Similar image', form=form,
                                   search_results=similar_hits,
                                   original_file=url_path_file,
                                   sentence_data=sentence_data,
                                   similar_numbers=len(similar_hits), model_up=True)
        else:
            return redirect(url_for('similar_image'))
    else:
        return render_template('similar_image.html', title='Similar image', form=form, model_up=True)


@app.route('/image/<path:image_name>')
def get_image(image_name):
    try:
        # Use os.path.join to handle subdirectories
        image_path = os.path.join('./static/images/', image_name)
        return send_file(image_path, mimetype='image/jpg')
    except FileNotFoundError:
        return 'Image not found.'


@app.errorhandler(413)
@app.errorhandler(RequestEntityTooLarge)
def app_handle_413(e):
    return render_template('error.413.html', title=e.name, e_name=e.name, e_desc=e.description,
                           max_bytes=app.config["MAX_CONTENT_LENGTH"])


def sentence_embedding(query: str):
    encode = img_model_sentence.encode(query)
    return encode
    # response = es.ml.infer_trained_model(model_id=INFER_MODEL_IM_SEARCH, docs=[{"text_field": query}])
    # return response['inference_results'][0]


def knn_search_images(dense_vector: list):
    source_fields = ["image_id", "image_name", "img_url", "article"]
    query = {
        "field": "image_embedding",
        "query_vector": dense_vector,
        "k": 5,
        "num_candidates": 10
    }

    response = es.search(
        index=INDEX_IM_EMBED,
        fields=source_fields,
        knn=query, source=False)

    return response


def infer_trained_model(query: str, model: str):
    response = es.ml.infer_trained_model(model_id=model, docs=[{"text_field": query}])
    return response['inference_results'][0]


def image_embedding(image, model, my_model_bbox):
    transformer = T.ToTensor()
    img = transformer(image)
    output = my_model_bbox([img.to(DEVICE)])
    bbox_cor = output[0]['boxes'][0].detach().numpy()
    cropped_image = image.crop(bbox_cor)
    return model.encode(cropped_image), cropped_image

