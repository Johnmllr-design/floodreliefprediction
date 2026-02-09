import torch
import os
import requests
import numpy as np




class flood_data_curator():

    def __init__(self):
        response = requests.get("https://www.fema.gov/api/open/v2/FimaNfipClaims")
        response = response.json()
        claims = response['FimaNfipClaims']
        dataset = []
        for row in claims:
            input_hash = {'waterDepth' :  row['waterDepth'], 
                          'floodWaterDuration' : row['floodWaterDuration'], 
                          'causeOfDamage' : row['causeOfDamage'], 
                          'floodEvent' : row['floodEvent'], 
                          'floodCharacteristicsIndicator' : row['floodCharacteristicsIndicator'],
                          'ground_truth' : row['netBuildingPaymentAmount']}
            dataset.append(input_hash)
        np.save('flood_dataset', dataset)


print(np.load('flood_dataset.npy', allow_pickle=True))