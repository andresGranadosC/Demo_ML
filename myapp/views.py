import numpy as np
import io
import pandas as pd
from django.shortcuts import render
from django.http import HttpResponse

from .datasetManager import loadDataSet
from .forms import ContactForm, SnippetForm, CcDefaultForm
from .models import CreditCardDefault
from .reportsManager import ReportManager
from .predictionModelManager import ModelManager, ModelTestManager, selectBestEstimator

# from celery.decorators import task
# from celery import task
from celery import shared_task
import time


def contact(request):
    
    if request.method == 'POST':
        form = ContactForm(request.POST)
        if form.is_valid():
            name=form.cleaned_data['name']
            email=form.cleaned_data['email']
    
    else:
        form = ContactForm()
    return render(request, 'form.html', {'form': form})


def snippet_detail(request):

    if request.method == 'POST':
        form = SnippetForm(request.POST)
        if form.is_valid():
            form.save()
    
    form = SnippetForm()
    return render(request, 'form.html', {'form': form})


def ccdefault_detail(request):
    if request.method == 'POST':
        form = CcDefaultForm(request.POST)
        if form.is_valid():
            creditCardDefault = CreditCardDefault(form.cleaned_data)
            
    else:
        form = CcDefaultForm()
    return render(request, 'cc_default_form.html', {'form': form})


def welcome( request ):
    return render(request, 'welcome.html')


def visualizedata( request,  processing = 0 ):
    print("---------------------- request.build_absolute_uri() ----------------------: ", request.build_absolute_uri())
    print("---------------------- request.POST ----------------------:", request.POST )
    if request.method == 'POST':
        form = CcDefaultForm(request.POST)
        if form.is_valid():
            creditCardDefault = CreditCardDefault(form.cleaned_data)
            enteredFeatures = creditCardDefault.extractSelectedFeatures(form.cleaned_data)

            if request.POST.get('submit2'):

                result = generateHistograms.delay( enteredFeatures )
                
                return render(request, 'generalreport.html', context={'imageNames': list( enteredFeatures ), 'task_id': result.task_id, 'info': None })

            else:

                # df = pd.DataFrame(enteredFeatures, index=[0])
                # modelManager = ModelManager(df, selectedFeatures = enteredFeatures )

                # result = modelManager.makeModelTest()

                modelTestManager = ModelTestManager()
                # result = modelTestManager.makeModelTest.delay(enteredFeatures)
                result = modelTestManager.makeModelTest.delay(enteredFeatures)
                # print("------------------------result-----------------------: ", list( dir(result) ))

                predictionString = '...'
                if result == 'PREDICITON DONE':
                    prediction, bestEstimator, bestMethodRecall = selectBestEstimator()
                    predictionString = "No"
                    if prediction:
                        predictionString = "Yes"
                    return render( request, 'cc_default_form.html', { 'form': form, 'prediction': predictionString, 'bestEstimator': bestEstimator, 'bestMethodRecall': bestMethodRecall, 'task_id': result.task_id } )
                else:
                    prediction, bestEstimator, bestMethodRecall = ('...', '...', '...')
                    return render( request, 'cc_default_form.html', { 'form': form, 'prediction': predictionString, 'bestEstimator': bestEstimator, 'bestMethodRecall': bestMethodRecall, 'task_id': result.task_id } )
                
                

        return render(request, 'cc_default_form.html', {'form': form, 'prediction': None})
    
    return render(request, 'generalreport.html')

@shared_task(bind=True)
def generateHistograms(self, enteredFeatures):

    progress_recorder = ProgressRecorder(self)
    
    report = ReportManager()
    target, features = loadDataSet()

    for i, featureName in enumerate( enteredFeatures ):
        progress_recorder.set_progress(i + 1, len(enteredFeatures))
        data1 = features[featureName][target == 0]
        data2 = features[featureName][target == 1]
        report.twoHistogram(data1, data2, featureName, enteredFeatures[featureName] , featureName+'.png')
        
    return 'done'


def visualizePredictionResults(request):
    print("------------------------ request.method: ----------------------- ", type(request.method) )
    return render(request, 'generalreport.html', context={'imageNames': [], 'info': 'info prepared' })
    """
    if request.method == 'GET':
        form = CcDefaultForm(request.POST)
        if form.is_valid():
            creditCardDefault = CreditCardDefault(form.cleaned_data)
            enteredFeatures = creditCardDefault.extractSelectedFeatures(form.cleaned_data)
            print("------------------------ visualizePredictionResults -----------------------: ", enteredFeatures )
            # result = generateHistograms.delay( enteredFeatures )

            return render(request, 'generalreport.html', context={'imageNames': list( enteredFeatures ), 'info': 'info prepared' })


        return render(request, 'cc_default_form.html', {'form': form, 'prediction': None})
    
    return render(request, 'generalreport.html')
    """
