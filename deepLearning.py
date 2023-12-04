import keras

from constants import *
from dataPreparation import *


def buildLSTM(vocabularySize, inputLength, embeddingMatrix):
    embedding = keras.layers.Embedding(
        vocabularySize, EMBEDDING_DIM, input_length=inputLength, mask_zero=True, trainable=False)
    model = keras.models.Sequential([
        embedding,
        keras.layers.LSTM(32),
        keras.layers.Dense(4, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                  metrics="sparse_categorical_accuracy")
    embedding.set_weights([embeddingMatrix])
    return model


def buildMLP(inputShape):
    model = keras.models.Sequential([
        keras.layers.Dense(4096, input_shape=inputShape,
            activation='relu', kernel_initializer='he_normal', bias_initializer='he_normal'),
        keras.layers.Dense(128,
            activation='relu', kernel_initializer='he_normal', bias_initializer='he_normal'),
        keras.layers.Dense(4, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                  metrics='sparse_categorical_accuracy')
    return model


def buildCNN(vocabularySize, inputLength, embeddingMatrix):
    embedding = keras.layers.Embedding(
        vocabularySize, EMBEDDING_DIM, input_length=inputLength, mask_zero=True, trainable=False)
    model = keras.models.Sequential([
        embedding,
        keras.layers.Conv1D(128, 3,
            activation='relu', kernel_initializer='he_normal', bias_initializer='he_normal'),
        keras.layers.MaxPooling1D(3),
        keras.layers.Conv1D(128, 3,
            activation='relu', kernel_initializer='he_normal', bias_initializer='he_normal'),
        keras.layers.MaxPooling1D(3),
        keras.layers.Conv1D(128, 3,
            activation='relu', kernel_initializer='he_normal', bias_initializer='he_normal'),
        keras.layers.MaxPooling1D(3),
        keras.layers.Conv1D(128, 5,
            activation='relu', kernel_initializer='he_normal', bias_initializer='he_normal'),
        keras.layers.MaxPooling1D(3),
        keras.layers.Conv1D(128, 3,
            activation='relu', kernel_initializer='he_normal', bias_initializer='he_normal'),
        keras.layers.GlobalMaxPooling1D(),
        keras.layers.Dense(128,
            activation='relu', kernel_initializer='he_normal', bias_initializer='he_normal'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(4, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                  metrics="sparse_categorical_accuracy")
    embedding.set_weights([embeddingMatrix])
    return model


def buildMNN(vocabularySize, inputCNNLength, embeddingMatrix, inputMLPShape, verbose=0):
    embedding = keras.layers.Embedding(
        vocabularySize, EMBEDDING_DIM, input_length=inputCNNLength, mask_zero=True, trainable=False)
    modelCNN = keras.models.Sequential([
        embedding,
        keras.layers.Conv1D(128, 3,
            activation='relu', kernel_initializer='he_normal', bias_initializer='he_normal'),
        keras.layers.MaxPooling1D(3),
        keras.layers.Conv1D(128, 3,
            activation='relu', kernel_initializer='he_normal', bias_initializer='he_normal'),
        keras.layers.MaxPooling1D(3),
        keras.layers.Conv1D(128, 3,
            activation='relu', kernel_initializer='he_normal', bias_initializer='he_normal'),
        keras.layers.MaxPooling1D(3),
        keras.layers.Conv1D(128, 5,
            activation='relu', kernel_initializer='he_normal', bias_initializer='he_normal'),
        keras.layers.MaxPooling1D(3),
        keras.layers.Conv1D(128, 3,
            activation='relu', kernel_initializer='he_normal', bias_initializer='he_normal'),
        keras.layers.GlobalMaxPooling1D(),
        keras.layers.Dense(128,
            activation='relu', kernel_initializer='he_normal', bias_initializer='he_normal'),
        keras.layers.Dropout(0.3)
    ], 'modelCNN')
    if 0 < verbose < 3:
        modelCNN.summary()

    modelMLP = keras.models.Sequential([
        keras.layers.Dense(2048, input_shape=inputMLPShape,
            activation='relu', kernel_initializer='he_normal', bias_initializer='he_normal'),
        keras.layers.Dense(128,
            activation='relu', kernel_initializer='he_normal', bias_initializer='he_normal'),
        keras.layers.Dropout(0.3)
    ], 'modelMLP')
    if 0 < verbose < 3:
        modelMLP.summary()

    layerMNN = keras.layers.Maximum()([modelCNN.output, modelMLP.output])
    layerMNN = keras.layers.Dense(16, activation='relu',
            kernel_initializer='he_normal', bias_initializer='he_normal')(layerMNN)
    layerMNN = keras.layers.Dense(4, activation='softmax')(layerMNN)
    modelMNN = keras.models.Model(inputs=[modelCNN.input, modelMLP.input],
            outputs=layerMNN, name='modelMNN')

    modelMNN.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                     metrics="sparse_categorical_accuracy")
    embedding.set_weights([embeddingMatrix])
    return modelMNN


def classify(modelName, genres, summaries, wordIndex, embeddingMatrix=None, verbose=0):
    print('\n--------------------------------------------------------------------------')
    print(modelName)
    print('--------------------------------------------------------------------------')

    xTrain, XTest = summaries
    yTrain, yTest = genres

    vocabularySize = max(wordIndex.values())+1
    if 0 < verbose < 3:
        print('vocabulary size = ', vocabularySize)

    if modelName == 'LSTM':
        model = buildLSTM(vocabularySize, xTrain.shape[1], embeddingMatrix)
    elif modelName == 'MLP':
        model = buildMLP(xTrain.shape[1:])
    elif modelName == 'CNN':
        model = buildCNN(vocabularySize, xTrain.shape[1], embeddingMatrix)
    elif modelName == 'MNN':
        model = buildMNN(vocabularySize, xTrain[0].shape[1], embeddingMatrix,
                         xTrain[1].shape[1:], verbose)
    if 0 < verbose < 3:
        model.summary()

    if verbose > 0:
        print("training...")
    model.fit(xTrain, yTrain, epochs=30, validation_split=0.1, verbose=verbose, batch_size=128,
              callbacks=keras.callbacks.EarlyStopping(monitor='val_sparse_categorical_accuracy',
                                                      patience=3, restore_best_weights=True))
    if 0 < verbose < 4:
        print('')

    print("evaluating...")
    res = model.evaluate(XTest, yTest, verbose=verbose, batch_size=128)

    print('loss: %.4f' % (res[0]))
    print('accuracy: %.4f' % (res[1]))


def classifyLSTM(genres, summaries, wordIndex, embeddingMatrix, verbose):
    classify('LSTM', genres, summaries, wordIndex, embeddingMatrix, verbose)


def classifyMLP(genres, summaries, wordIndex, verbose):
    classify('MLP', genres, summaries, wordIndex, verbose=verbose)


def classifyCNN(genres, summaries, wordIndex, embeddingMatrix, verbose):
    classify('CNN', genres, summaries, wordIndex,
             embeddingMatrix, verbose=verbose)


def classifyMNN(booksData, verbose):  # MultiNN
    train = [booksData['summaries'][0], booksData['summariesWP'][0]]
    test = [booksData['summaries'][1], booksData['summariesWP'][1]]
    classify('MNN', booksData['genres'], (train, test),
             booksData['wordIndex'], booksData['embeddingMatrix'], verbose=verbose)
