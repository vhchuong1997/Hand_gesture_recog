from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.applications.imagenet_utils import preprocess_input
from keras import optimizers, losses, activations, models
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, LearningRateScheduler
import os

def main():
    model_dir = 'untrained_HandNet_1.h5'
    HandNet = models.load_model(model_dir)

    epochs = 60

    HandNet.compile(loss='categorical_crossentropy', 
               #optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
                optimizer = optimizers.Adam(learning_rate=0.0001, decay=0.0001/epochs),
                metrics=['accuracy'])
    HandNet.load_weights('ckpts/HandNet_4/weights.epoch_115-val_accuracy_0.9956.h5')
    HandNet.summary()

    ROWS = 64
    COLS = 64
    
    val_generator = ImageDataGenerator(vertical_flip=False,
                               preprocessing_function=preprocess_input)
    
    test_gen = val_generator.flow_from_directory(
    'output6_2_2/test',
    class_mode='categorical',
    target_size=(ROWS, COLS),
    batch_size = 32,
    shuffle= False
    )

    score = HandNet.evaluate(test_gen, verbose=1)
    print("%s: %.2f%%" % (HandNet.metrics_names[1], score[1]*100))
    # '''
    NAME = 'HandNet_4_1'
    HandNet.save('models/' + NAME + '.h5')
    # '''
if(__name__=="__main__") : 
    main()