import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle
import pdb



#>>> df.keys()
#Index(['Pick Longitude', 'Pick Latitude', 'Pick Elevation', 'Phase',
 #      'Pick Time', 'Source Longitude', 'Source Latitude', 'Source Depth',
#       'Source Magnitude', 'Source Time'],


df=pickle.load(open("training_data_10000.pkl",'rb'))

def write_pickle():
    skip_background=True
    events=[]
    e_i=[0]
    et=df['Source Time'].iloc[0]
#    assert(et!=0)
    for point in range(1,len(df)):
      if point % 1000==0:print (point)
      if df['Source Time'].iloc[point] == et:
          e_i.append(point)
      elif df['Source Time'].iloc[point]==0:
          e_i.append(point)
          
      else:
          if et==0: #This is need because the first point is background point!
              #e_i.append(point)
              et=df['Source Time'].iloc[point]
          else:
              events.append(np.array(df.iloc[e_i]))
              if not (df['Source Time'].iloc[e_i]!=0).any():
                  pdb.set_trace()
              e_i=[point]
              
              et=df['Source Time'].iloc[point]
    pickle.dump(events, open('event_p.pkl','wb'))

    return events    
    pass
write_pickle()

def process_event(data):
    source_time=np.max(data[:,-1])
    source_offset=np.random.uniform(-500,500)
    noise=np.zeros((len(data),1))
    for entry_i,v in enumerate(data): 
        if v[4]==0 or v[9]==0:
            data[entry_i,4]=np.random.uniform(-500,500)
            data[entry_i,9]=0
        else:
            data[entry_i,[4,9]]-=(source_time+source_offset)
            noise[entry_i,0]=1
    data=np.concatenate([data,noise],axis=-1)
#    np.random.shuffle(data)
    return data
    
events=pickle.load(open('event_p.pkl','rb'))


def data_generator(batch_size=32):
    events=pickle.load(open('event_p.pkl','rb'))
    while True:
        batch=[]
        b_choice=[np.random.choice(len(events),1+np.random.poisson(1.5),replace=False) for i in range(batch_size)]        
        for evt in b_choice:
            data=np.concatenate([process_event(events[i]) for i in evt],axis=0)
            np.random.shuffle(data)
            batch.append(data)

            
        data=tf.keras.preprocessing.sequence.pad_sequences(batch,dtype='float32')
                  
        x=data[:,:,:5]
        y=data[:,:,5:]

        x[:,:,0:3]=x[:,:,0:3]/100.
        y[:,:,0:3]=y[:,:,0:3]/100.

        x[:,:,4]=x[:,:,4]/100.
        y[:,:,4]=y[:,:,4]/100.

        print(x.shape,y.shape)
        yield x,y
    


                   

      
def data_faker_generator(batch_size):

    while True:
        batch_x=[]
        batch_y=[]
        for trial in range(batch_size):
            events=[]
            for i in range(np.random.poisson(3)):
                event_x=np.random.uniform(-1,1)
                event_y=np.random.uniform(-1,1)
                event_t=np.random.uniform(-10,10)
                events.append([event_x,event_y,event_t])
            stations=np.random.uniform(-2,2,size=(10,2))
            picks=[]
            targets=[]
            for event in events:
                for sta in stations:
                    time=((sta[0]-event[0])**2+(sta[1]-event[1])**2)**.5+event[2] 
                    picks.append([(sta[0],sta[1],time),event])

            picks.sort(key=lambda k: k[0][2])
            batch_x.append([p[0] for p in picks])
            batch_y.append([p[1] for p in picks])
            

            
        batch_x=tf.keras.preprocessing.sequence.pad_sequences(
                batch_x, maxlen=None, dtype='float32', padding='post', truncating='post',
                value=0.0
            )

        batch_y=tf.keras.preprocessing.sequence.pad_sequences(
                batch_y, maxlen=None, dtype='float32', padding='post', truncating='post',
                value=0.0
            )
        print(batch_x.shape)
        yield batch_x,batch_y



input_layer=tf.keras.layers.Input(shape=(None,5))
lstm_f=tf.keras.layers.LSTM(20,return_sequences=True)(input_layer)
lstm_b=tf.keras.layers.LSTM(20,go_backwards=True,return_sequences=True)(input_layer)
lstm=tf.keras.layers.Concatenate()([lstm_f,lstm_b])
output=tf.keras.layers.Dense(6)(lstm)


def loss(true,pred):
    
    c_pred=tf.math.sigmoid(pred[:,:,-1])
    true_label=true[:,:,-1]
    
    c_loss=  (1-true_label)*tf.math.log(1-c_pred)+(true_label)*tf.math.log(c_pred)

    
    reg_loss= tf.reduce_sum(tf.square(true[:,:,:4]-pred[:,:,:4]),axis=-1)*true_label

    
    return tf.reduce_sum(reg_loss)+-1*tf.reduce_sum(c_loss)
    

    


model =tf.keras.models.Model(input_layer,output)
opt=tf.keras.optimizers.Adam(1e-4)
model.compile(loss=loss,optimizer=opt)
print('here')

        
if __name__=="__main__":
#    write_pickle()
    d=data_generator(32)
    for i in range(10):
        x,y=next(d)
    checkpoint = tf.keras.callbacks.ModelCheckpoint('asso_best.tf', monitor='loss',save_weights_only=True, verbose=1, save_best_only=True, mode='min')

    model.load_weights('asso_best.tf')   
    model.fit_generator(d,1000,100,callbacks=[checkpoint])
    model.load_weights('asso_best.tf')
    
    pred=model.predict(x)

    for i in range(10):
        points=np.concatenate([np.expand_dims(p,0) for p in pred[i] if p[5]>0 ],axis=0)
        events=np.concatenate([np.expand_dims(p,0) for p in y[i] if p[5]!=0 ],axis=0)
        
        plt.scatter(points[:,0],points[:,1],c=points[:,4] )
        plt.scatter(events[:,0],events[:,1],c=events[:,4],marker='x')
        plt.show()
    
