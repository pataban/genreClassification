from bidict import bidict


DATA_FIE_PATH = 'data\\'
RAW_DATA_FILE_NAME = 'bookSummaries'
RAW_DATA_FILE_EXTENSION = '.txt'
DATA_FILE_NAME = 'bookSummaries'
GENRE_INDEX = bidict({'Science Fiction': 0, 'Fantasy': 1,'Mystery': 2, 'Crime Fiction': 3})
# GENRE_INDEX = bidict({'Novel': 0, 'Science Fiction': 1, 'Fiction': 2, 'Fantasy': 3})
# GENRE_INDEX = bidict({'Novel': 0, 'Science Fiction': 1, 'Fiction': 2, "Children's literature": 3, 'Fantasy': 4})
# 'Young adult literature': 5, 'Historical novel': 6, 'Speculative fiction': 7, 'Crime Fiction': 8, 'Non-fiction': 9})
CLEAN_SUMMARY_MANUAL = False
TRAIN_SIZE = 1000
TEST_SIZE = 100

SUMMARY_MIN_LENGTH = 500    #TODO double same constant
SUMMARY_MAX_LENGTH = 1000
TEST_SPLIT = 0.2
RANDOM_STATE = 0

DROP_TOP = 10
KEEP_BOTTOM = 20000
SUMMARY_LENGTH_MIN = 50
SUMMARY_LENGTH_MAX = 200
