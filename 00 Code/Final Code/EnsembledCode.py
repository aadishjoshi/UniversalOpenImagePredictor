
# coding: utf-8

# In[1]:


from DataPreprocess import *
from ourModel import *
from ourInceptionV3 import *
from ourInceptionResnetV2 import *
from ourResnet50 import *


# In[2]:


#testing code
df = pd.read_csv('E:/Datasets/all/stage_1_sample_submission.csv', usecols=[0])
im = df.image_id.tolist()

for i in tqdm_notebook(im):
    try: 
        X_test = np.array(load_img('E:/Datasets/all/stage_1_test_images/{}.jpg'.format(i),target_size=(100,100), grayscale=True))/255
    except:
        pass
    
X_test = np.array(X_test).reshape((1,100,100,1))


# In[3]:


class_description = pd.read_csv('E:/Datasets/all/class-descriptions.csv', usecols=[0,1])
label_list = class_description.label_code.tolist()


# In[ ]:


#----------------------------------------------------prediction for ourModel20


# In[ ]:


ourModel20 = myModule()
prediction_ourModel = ourModel20.predict(X_test).argsort(1)[:,:5]


# In[ ]:


p1 = []
for it in tqdm(prediction_ourModel):
    p1.append([classes[int(i)] for i in it])
    
    
for i in p1[0]:
    for k in range(0,len(class_description.label_code)):
        j = class_description.label_code[k]
        if i == j:
            print(j)
            print(class_description.description[k])
            break;


# In[ ]:


#--------------------------------------------------------------ourInceptionResnetModel


# In[ ]:


#prediction for ourInceptionResnetModel
ourInceptionResnetModel = ourInceptionResnet()
prediction_ourInceptionResnet = ourInceptionResnetModel.predict(X_test).argsort(1)[:,:5]


# In[ ]:


p2 = []
for it in tqdm(prediction_ourModel):
    p2.append([classes[int(i)] for i in it])
    
    
for i in p2[0]:
    for k in range(0,len(class_description.label_code)):
        j = class_description.label_code[k]
        if i == j:
            print(j)
            print(class_description.description[k])
            break;


# In[ ]:


#--------------------------------------------------------------ourResnet50Model


# In[ ]:


#prediction for ourResnet50Model
ourResnet50Model = ourResnet()
prediction_ourResnet50 = ourResnet50Model.predict(X_test).argsort(1)[:,:5]


# In[ ]:


p3 = []
for it in tqdm(prediction_ourModel):
    p3.append([classes[int(i)] for i in it])
    
    
for i in p3[0]:
    for k in range(0,len(class_description.label_code)):
        j = class_description.label_code[k]
        if i == j:
            print(j)
            print(class_description.description[k])
            break;


# In[ ]:


#--------------------------------------------------------------ourInceptionV3Model


# In[ ]:


#prediction for ourInceptionV3Model
ourInceptionV3Model = ourInceptionV3()
prediction_ourInceptionV3 = ourInceptionV3Model.predict(X_test).argsort(1)[:,:5]


# In[ ]:


p4 = []
for it in tqdm(prediction_ourModel):
    p4.append([classes[int(i)] for i in it])
    
    
for i in p4[0]:
    for k in range(0,len(class_description.label_code)):
        j = class_description.label_code[k]
        if i == j:
            print(j)
            print(class_description.description[k])
            break;


# In[ ]:


def ensemble(output1,output2, output3, output4):
    print(numpy.average([output1, output2, output3, output4], axis = 1).

ensemble(p1,p2,p3,p4)

