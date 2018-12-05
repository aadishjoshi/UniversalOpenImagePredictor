
from keras.applications.inception_v3 import InceptionV3
from DataPreprocess import *


def ourInceptionV3():
    
    X_train = [np.array(load_img(r'E:\Datasets\validation\{}.jpg'.format(i),target_size=(139,139), grayscale=False))/255 for i in tqdm(Imageid[10000:20000])]
    X_Val = [np.array(load_img(r'E:\Datasets\validation\{}.jpg'.format(i),target_size=(139,139), grayscale=False))/255 for i in tqdm(Imageid[:2000])]
    
    myModel = InceptionV3(weights='imagenet', include_top=False,input_shape = (139,139,3))
    x = myModel.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(100, activation='relu')(x)
    predictions = Dense(20, activation='softmax')(x)
    model = Model(inputs=myModel.input, outputs=predictions)
    X_train = np.array(X_train).reshape((10000,139,139,3))
    X_Val = np.array(X_Val).reshape((2000,139,139,3))
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.RMSprop(lr=0.001),
                  metrics=['accuracy'])
    model.fit(X_train, Y_train, validation_data=(X_Val,Y_Val), batch_size=50, epochs=10, verbose=1)
    return model


if __name__ == "__main__":
    ourInceptionV3Model = ourInceptionV3()

