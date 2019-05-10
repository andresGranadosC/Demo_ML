import pandas as pd
import numpy as np

from .serviceManager import *

def loadDataSetWithFeatures(selectedFeatures = [] ):
    
    dataset = pd.read_csv('static/UCI_Credit_Card.csv')
    target = dataset['default.payment.next.month']
    features = dataset.drop(columns=['ID','default.payment.next.month'])
    if all([(x in features.columns) for x in selectedFeatures ]):
        return (target, features[selectedFeatures] )
    else:
        raise InvalidFeatureNames

def loadDataSet():
    
    dataset = pd.read_csv('static/UCI_Credit_Card.csv')
    target = dataset['default.payment.next.month']
    features = dataset.drop(columns=['ID','default.payment.next.month'])
    return (target, features )
