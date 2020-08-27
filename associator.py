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

    events=[]
    e_i=[0]
    et=df['Source Time'].iloc[0]
#    assert(et!=0)
    for point in range(1,len(df)):
      if point % 1000==0:print (point)
      if df['Source Time'].iloc[point] == et or df['Source Time'].iloc[point]==0:
          e_i.append(point)
      else:
          if et==0: #This is need because the first point is background point!
              e_i.append(point)
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


def data_generator(batch_size=32):
    events=pickle.load(open('event_p.pkl','rb'))
    while True:
        b_choice=np.random.choice(len(events),batch_size,replace=False)
        data=[events[i] for i in b_choice]        
        for i,e in enumerate(data):
            source_time=e[-1,-1]
#            print(source_time,data[i][:,[4,9]])
            source_offset=np.random.uniform(-500,500)
            
            data[i][:,[4,9]]-=(source_time+source_offset)
            del_index=[i for i in range(len(e)) if e[i,4] > 100]
            
            data[i]=np.delete(data[i][:],del_index,axis=0)
            np.random.shuffle(data[i])
            
        
        print(np.mean([len(d) for d in data]))
        data=tf.keras.preprocessing.sequence.pad_sequences(data,dtype='float32')
          
        
        x=data[:,:,:5]/100.
        y=data[:,:,5:]/100.
        pdb.set_trace()
        
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
        yield batch_x,batch_y



input_layer=tf.keras.layers.Input(shape=(None,5))
lstm_f=tf.keras.layers.LSTM(20,return_sequences=True)(input_layer)
lstm_b=tf.keras.layers.LSTM(20,go_backwards=True,return_sequences=True)(input_layer)
lstm=tf.keras.layers.Concatenate()([lstm_f,lstm_b])
output=tf.keras.layers.Dense(5)(lstm)
model =tf.keras.models.Model(input_layer,output)
model.compile(loss='mse',optimizer='adam')


        
if __name__=="__main__":
    write_pickle()
    d=data_generator(32)
    x,y=next(d)
    checkpoint = tf.keras.callbacks.ModelCheckpoint('asso_best.tf', monitor='loss',save_weights_only=True, verbose=1, save_best_only=True, mode='min')

    
#    model.fit_generator(d,1000,10,callbacks=[checkpoint])
    model.load_weights('asso_best.tf')
    
    pred=model.predict(x)

    for i in range(10):
        plt.scatter(pred[i,:,0],pred[i,:,1],c=pred[i,:,4] )
        plt.scatter(y[i,:,0],y[i,:,1],c=y[i,:,4],marker='x' )
        plt.show()
    
