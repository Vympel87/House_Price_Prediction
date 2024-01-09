import json, pickle
import numpy as np

__location = None
__data_columns = None
__model = None

def get_estimated_price(location, sqft, bhk, bath):
   try:
       locIndex = __data_columns.index(location.lower())
   except:
       locIndex = -1

   x = np.zeros(len(__data_columns))
   x[0]= sqft
   x[1]= bath
   x[2]= bhk

   if locIndex >= 0:
       x[locIndex] = 1
       
   return round(__model.predict([x])[0], 2)

def get_location_names():
   return __location

def load_saved_artifacts():
  print("Loading....start")
  global __data_columns
  global __location
  global __model

  with open("./server/artifacts/columns.json", 'r') as f:
     __data_columns = json.load(f)['data_columns']
     __location = __data_columns[3:]

  with open("./server/artifacts/banglore_house_prices_model.pickle", 'rb') as f:
       __model = pickle.load(f)
  print("Loading....done")