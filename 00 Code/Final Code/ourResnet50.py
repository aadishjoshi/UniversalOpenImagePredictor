from keras.applications.resnet50 import ResNet50
from DataPreprocess import *

def ourResnet():
    
    X_train = [np.array(load_img(r'E:/Datasets/validation/{}.jpg'.format(i),target_size=(200,200), grayscale=False))/255 for i in tqdm(Imageid[100:200])]
    X_Val = [np.array(load_img(r'E:/Datasets/validation/{}.jpg'.format(i),target_size=(200,200), grayscale=False))/255 for i in tqdm(Imageid[:20])]

    
    myModel = ResNet50(weights='imagenet', include_top=False,input_shape = (200,200,3))

    x = myModel.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(100, activation='relu')(x)
    predictions = Dense(20, activation='softmax')(x)
    model = Model(inputs=myModel.input, outputs=predictions)

    X_train = np.array(X_train).reshape((100,200,200,3))
    X_Val = np.array(X_Val).reshape((20,200,200,3))

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.RMSprop(lr=0.001),
                  metrics=['accuracy'])
    
    model.fit(X_train, Y_train, validation_data=(X_Val,Y_Val), batch_size=50, epochs=10, verbose=1)
    return model

# export code
if __name__ == "__main__":
    ourResnet50 = ourResnet()

