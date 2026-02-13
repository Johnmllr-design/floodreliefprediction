import numpy as np
import torch
import random

class clean_dataset():
    def clean_data(self, data_path):
        try:
            dataset = np.load(data_path, allow_pickle=True)
            new_dataset = []
            event_dictionary = {}
            for row in dataset:
                new_row = row
                if new_row['floodEvent'] not in event_dictionary:
                    event_dictionary[new_row['floodEvent']] = [random.random(), random.random(), random.random()]
                    new_row['floodEvent'] = event_dictionary[new_row['floodEvent']]
                else:
                    new_row['floodEvent'] = event_dictionary[new_row['floodEvent']]
                if new_row['waterDepth'] == None:
                    new_row['waterDepth'] = 0
                if new_row['floodWaterDuration'] == None:
                    new_row['floodWaterDuration'] = 0
                if new_row['causeOfDamage'] == None:
                    new_row['causeOfDamage'] = 0
                del new_row['floodCharacteristicsIndicator']
                input_value = [float(new_row['waterDepth']), float(new_row['floodWaterDuration']), float(new_row['causeOfDamage']), float(new_row['floodEvent'][0]), float(new_row['floodEvent'][1]), float(new_row['floodEvent'][2]), float(new_row['ground_truth'])]
                
                new_dataset.append(input_value)
            
            np.save('new_dataset', new_dataset)
            np.save('event_dictionary', event_dictionary)
        except:
            print("couldn't load that path")




# obj = clean_dataset()
# obj.clean_data('flood_dataset.npy')

arr = np.load('new_dataset.npy', allow_pickle=True)
dicti = np.load('event_dictionary.npy', allow_pickle=True)
print(dicti)


