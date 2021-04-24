from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.applications.imagenet_utils import preprocess_input
from keras import optimizers, losses, activations, models
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, LearningRateScheduler
import os

def main():
    model_dir = 'untrained_HandNet_2.h5'
    handNet = models.load_model(model_dir)

    epochs = 120

    handNet.compile(loss='categorical_crossentropy', 
                optimizer = optimizers.Adam(learning_rate=0.0001),
                metrics=['accuracy'])
    handNet.summary()

    ROWS = 64
    COLS = 64
    generator = ImageDataGenerator(vertical_flip=False,
                               horizontal_flip=True,
                               height_shift_range=0.1,
                               width_shift_range=0.1,
                               preprocessing_function=preprocess_input)
    val_generator = ImageDataGenerator(vertical_flip=False,
                               preprocessing_function=preprocess_input)
    train_gen = generator.flow_from_directory(
    'output6_2_2/train',
    class_mode='categorical',
    target_size=(ROWS, COLS),
    batch_size = 32,
    shuffle= True
    )
    val_gen = val_generator.flow_from_directory(
    'output6_2_2/val',
    class_mode='categorical',
    target_size=(ROWS, COLS),
    batch_size = 32,
    shuffle= False 
    )
    test_gen = val_generator.flow_from_directory(
    'output6_2_2/test',
    class_mode='categorical',
    target_size=(ROWS, COLS),
    batch_size = 32,
    shuffle= False
    )

    NAME = 'HandNet_7'
    if not os.path.isdir("ckpts/"+ NAME):
        os.mkdir("ckpts/"+ NAME)
    tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))
    file_path = "ckpts/"+ NAME +"/weights.epoch_{epoch:02d}-val_accuracy_{val_accuracy:.4f}.h5"
    checkpoint = ModelCheckpoint(file_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max',period=1)
    early = EarlyStopping(monitor="val_accuracy", mode="max", patience=20)

    handNet.fit(train_gen, epochs=epochs, validation_data=val_gen, callbacks=[tensorboard, checkpoint, early])
    handNet.save('models/' + NAME + '.h5')
    score = handNet.evaluate(test_gen, verbose=1)
    print("%s: %.2f%%" % (handNet.metrics_names[1], score[1]*100))

if(__name__=="__main__") : 
    main()