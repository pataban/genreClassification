from bidict import bidict


DATA_DIR_PATH = 'data/'
RAW_BOOKS_DATA_FILE_NAME = 'bookSummaries.txt'
RAW_MOVIES_DATA_FILE_NAME = ('movieSummaries1.txt', 'movieSummaries2.txt')
GENRE_INDEX = bidict({'drama': 0, 'documentary': 1, 'comedy': 2, 'horror': 3})
#GENRE_INDEX = bidict({'Science Fiction': 0, 'Fantasy': 1, 'Mystery': 2, 'Crime Fiction': 3})
# GENRE_INDEX = bidict({'Novel': 0, 'Science Fiction': 1, 'Fiction': 2, 'Fantasy': 3})
# GENRE_INDEX = bidict({'Novel': 0, 'Science Fiction': 1, 'Fiction': 2, "Children's literature": 3, 'Fantasy': 4})
# 'Young adult literature': 5, 'Historical novel': 6, 'Speculative fiction': 7, 'Crime Fiction': 8, 'Non-fiction': 9})

CLEAN_SUMMARY_MANUAL = False
EMBEDDING_FILE_PATH = 'data/glove.6B.100d.txt'
EMBEDDING_DIM = 100

SUMMARY_LENGTH_MIN = 50
SUMMARY_LENGTH_MAX = 500
WORDS_DROP_TOP = 20
WORDS_KEEP_TOP = 20000

TRAIN_SIZE = 5000
TEST_SIZE = 500
