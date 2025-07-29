import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

INPUT_FOLDER = os.path.join(PROJECT_ROOT, 'images')
RESULTS_FOLDER = os.path.join(PROJECT_ROOT, 'results')
LOG_FOLDER = os.path.join(PROJECT_ROOT, 'logs')
LOG_FILE_PATH = os.path.join(LOG_FOLDER, 'image_classification.log')
