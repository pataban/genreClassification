from bidict import bidict


DATA_FIE_PATH = 'data/'
RAW_DATA_FILE_NAME = 'bookSummaries'
RAW_DATA_FILE_EXTENSION = '.txt'
DATA_FILE_NAME = 'bookSummaries'
GENRE_INDEX = bidict({'Science Fiction': 0, 'Fantasy': 1, 'Mystery': 2, 'Crime Fiction': 3})
# GENRE_INDEX = bidict({'Novel': 0, 'Science Fiction': 1, 'Fiction': 2, 'Fantasy': 3})
# GENRE_INDEX = bidict({'Novel': 0, 'Science Fiction': 1, 'Fiction': 2, "Children's literature": 3, 'Fantasy': 4})
# 'Young adult literature': 5, 'Historical novel': 6, 'Speculative fiction': 7, 'Crime Fiction': 8, 'Non-fiction': 9})
CLEAN_SUMMARY_MANUAL = False

EMBEDDING_FILE_PATH = 'data/glove.6B.100d.txt'
EMBEDDING_DIM = 100

SUMMARY_LENGTH_MIN = 200
SUMMARY_LENGTH_MAX = 1000
WORDS_DROP_TOP = 10
WORDS_KEEP_TOP = 20000

TRAIN_SIZE = 3000
TEST_SIZE = 300
