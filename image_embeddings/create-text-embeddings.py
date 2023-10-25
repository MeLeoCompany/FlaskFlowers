import os
import sys
import time
import json
import argparse
from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch, SSLError
from elasticsearch.helpers import parallel_bulk
from PIL import Image
from datetime import datetime
from exif import Image as exifImage
import base64
import io


# a = 1+0.0117
# sum=0
# for i in range(0,36):
#     print(i)
#     sum += pow(a,i)

# x0 = 200000*sum/pow(a,36)
# print("x0:",x0)


ES_HOST = "https://127.0.0.1:9200/"
ES_USER = "elastic"
ES_PASSWORD = "changeme"
ES_TIMEOUT = 3600

DEST_INDEX = "my-text-embedding"
DELETE_EXISTING = True
CHUNK_SIZE = 100
INDEX_MAPING = "text-embeddings-mappings.json"

JSON_FILE = '..\\json_in\\pics_base_with_data.pics_data.json'

CA_CERT = '../app/conf/ess-cloud.cer'


parser = argparse.ArgumentParser()
parser.add_argument('--es_host', dest='es_host', required=False, default=ES_HOST,
                    help="Elasticsearch hostname. Must include HOST and PORT. Default: " + ES_HOST)
parser.add_argument('--es_user', dest='es_user', required=False, default=ES_USER,
                    help="Elasticsearch username. Default: " + ES_USER)
parser.add_argument('--es_password', dest='es_password', required=False, default=ES_PASSWORD,
                    help="Elasticsearch password. Default: " + ES_PASSWORD)
parser.add_argument('--verify_certs', dest='verify_certs', required=False, default=True,
                    action=argparse.BooleanOptionalAction,
                    help="Verify certificates. Default: True")
parser.add_argument('--thread_count', dest='thread_count', required=False, default=4, type=int,
                    help="Number of indexing threads. Default: 4")
parser.add_argument('--chunk_size', dest='chunk_size', required=False, default=CHUNK_SIZE, type=int,
                    help="Default: " + str(CHUNK_SIZE))
parser.add_argument('--timeout', dest='timeout', required=False, default=ES_TIMEOUT, type=int,
                    help="Request timeout in seconds. Default: " + str(ES_TIMEOUT))
parser.add_argument('--delete_existing', dest='delete_existing', required=False, default=True,
                    action=argparse.BooleanOptionalAction,
                    help="Delete existing indices if they are present in the cluster. Default: True")
parser.add_argument('--ca_certs', dest='ca_certs', required=False,  # default=CA_CERT,
                    help="Path to CA certificate.")  # Default: ../app/conf/ess-cloud.cer")
parser.add_argument('--extract_GPS_location', dest='gps_location', required=False, default=False,
                    action=argparse.BooleanOptionalAction,
                    help="[Experimental] Extract GPS location from photos if available. Default: False")

args = parser.parse_args()


def main():
    global args
    lst = []

    f_json = open(JSON_FILE, encoding='utf-8')
    data = json.load(f_json)
    img_model_sentence = None

    start_time = time.perf_counter()
    duration = time.perf_counter() - start_time
    print(f'Duration load model = {duration}')

    start_time = time.perf_counter()
    # for filename in tqdm(filenames, desc='Processing files', total=len(filenames)):
    for item in data:
        doc = {'image_id': item['_id']['$oid'],
                'image_name': item['title'], 'img_url': item['source_image_url'],
                'page_url': item['product_page_url'], 'article': item['article'], 
                'all_text_data':item['content']['all_text_data']+','+item['title'],
                'exif': {}
            }
        if img_model_sentence is None:
            img_model_sentence = SentenceTransformer('clip-ViT-B-32-multilingual-v1')
        embedding = sentence_embedding(doc['all_text_data'], img_model_sentence)
        doc['text_embedding'] = embedding.tolist()

        if len(lst)%513 == 0:
            print("loaded: "+str(len(lst)))
        lst.append(doc)

    duration = time.perf_counter() - start_time
    print(f'Duration creating image embeddings = {duration}')

    if args.ca_certs:
        es = Elasticsearch(
            hosts=[args.es_host],
            verify_certs=args.verify_certs,
            basic_auth=(args.es_user, args.es_password),
            ca_certs=args.ca_certs
        )
    else:
        es = Elasticsearch(
            hosts=[args.es_host],
            verify_certs=args.verify_certs,
            basic_auth=(args.es_user, args.es_password)
        )
    es.options(request_timeout=args.timeout)

    # index name to index data into
    index = DEST_INDEX
    try:
        with open(INDEX_MAPING, "r") as config_file:
            config = json.loads(config_file.read())
            if args.delete_existing:
                if es.indices.exists(index=index):
                    print("Deleting existing %s" % index)
                    es.indices.delete(index=index, ignore=[400, 404])

            print("Creating index %s" % index)
            es.indices.create(index=index,
                              mappings=config["mappings"],
                              settings=config["settings"],
                              ignore=[400, 404],
                              request_timeout=args.timeout)

        count = 0
        for success, info in parallel_bulk(
                client=es,
                actions=lst,
                thread_count=4,
                chunk_size=args.chunk_size,
                timeout='%ss' % 120,
                index=index
        ):
            if success:
                count += 1
                if count % args.chunk_size == 0:
                    print('Indexed %s documents' % str(count), flush=True)
                    sys.stdout.flush()
            else:
                print('Doc failed', info)

        print('Indexed %s documents' % str(count), flush=True)
        duration = time.perf_counter() - start_time
        print(f'Total duration = {duration}')
        print("Done!\n")
    except SSLError as e:
        if "SSL: CERTIFICATE_VERIFY_FAILED" in e.message:
            print("\nCERTIFICATE_VERIFY_FAILED exception. Please check the CA path configuration for the script.\n")
            raise
        else:
            raise


def sentence_embedding(query: str,img_model_sentence):
    encode = img_model_sentence.encode(query)
    return encode


def create_image_id(filename):
    # print("Image filename: ", filename)
    return os.path.splitext(os.path.basename(filename))[0]


def get_exif_date(filename):
    with open(filename, 'rb') as f:
        image = exifImage(f)
        taken = f"{image.datetime_original}"
        date_object = datetime.strptime(taken, "%Y:%m:%d %H:%M:%S")
        prettyDate = date_object.isoformat()
        return prettyDate


def get_exif_location(filename):
    with open(filename, 'rb') as f:
        image = exifImage(f)
        exif = {}
        lat = dms_coordinates_to_dd_coordinates(image.gps_latitude, image.gps_latitude_ref)
        lon = dms_coordinates_to_dd_coordinates(image.gps_longitude, image.gps_longitude_ref)
        return [lon, lat]


def dms_coordinates_to_dd_coordinates(coordinates, coordinates_ref):
    decimal_degrees = coordinates[0] + \
                      coordinates[1] / 60 + \
                      coordinates[2] / 3600

    if coordinates_ref == "S" or coordinates_ref == "W":
        decimal_degrees = -decimal_degrees

    return decimal_degrees


if __name__ == '__main__':
    main()
