import os
from os.path import join, isdir
from vars import *

def generateModelID(model_name):
    path_to_dir = LOGS_DIR + model_name
    id = 1
    if os.path.exists(path_to_dir):
        dir_content = os.listdir(path_to_dir)
        if len(dir_content) > 0:
            existing_model_names = [content for content in dir_content if isdir(join(path_to_dir, content))]
            existing_models_ids = [int(model.split('_')[-1]) for model in existing_model_names]
            id = max(existing_models_ids) + 1

    return id
