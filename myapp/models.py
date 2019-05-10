from django.db import models


class Snippet(models.Model):
    name = models.CharField(max_length=100)
    body = models.TextField()

    def __str__(self):
        return self.name

class CreditCardDefault(models.Model):
    
    limitBal = models.CharField(max_length=7)
    gender = models.CharField(max_length=1)
    education = models.CharField(max_length=1)
    marriage = models.CharField(max_length=1)
    age = models.CharField(max_length=2)
    pay0 = models.CharField(max_length=7)
    pay2 = models.CharField(max_length=7)
    pay3 = models.CharField(max_length=7)
    pay4 = models.CharField(max_length=7)
    pay5 = models.CharField(max_length=7)
    pay6 = models.CharField(max_length=7)
    billAmt1 = models.CharField(max_length=7)
    billAmt2 = models.CharField(max_length=7)
    billAmt3 = models.CharField(max_length=7)
    billAmt4 = models.CharField(max_length=7)
    billAmt5 = models.CharField(max_length=7)
    billAmt6 = models.CharField(max_length=7)
    payAmt1 = models.CharField(max_length=7)
    payAmt2 = models.CharField(max_length=7)
    payAmt3 = models.CharField(max_length=7)
    payAmt4 = models.CharField(max_length=7)
    payAmt5 = models.CharField(max_length=7)
    payAmt6 = models.CharField(max_length=7)

    def __str__(self):
        return self.age
    
    def __init__(self, data = {}, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = data

    def extractSelectedFeatures(self, dic):
        selectedFeatures = {}
        for k in dic:
            if dic[k] != '':
                selectedFeatures[k] = int(dic[k])
                
        return selectedFeatures
