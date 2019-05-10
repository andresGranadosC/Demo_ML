from .datasetManager import loadDataSet
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import io


class ReportManager:


    features, target = None, None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.target, self.features = loadDataSet()
        
    
    def testplot1(self, data):
        fig = Figure()
        buf = io.BytesIO()
        canvas = FigureCanvas(fig)
        #canvas.get_default_filename = lambda: 'testplot1.png'
        ax = fig.add_subplot(111)
        x=[1, 2, 3, 4, 5, 6, 7, 8, 9]
        y=[1, 2, 3, 4, 5, 6, 7, 8, 9]
        ax.plot(x, y)
        canvas.print_png(buf)
        # response=HttpResponse(buf.getvalue(),content_type='image/png')
        fig.savefig('myapp/static/testplot1.png')
        fig.clear()
        # response['Content-Length'] = str(len(response.content))
        # return response

    def testplot2(self, data):
        fig = Figure()
        buf = io.BytesIO()
        canvas = FigureCanvas(fig)
        #canvas.get_default_filename = lambda: 'testplot2.png'
        ax = fig.add_subplot(111)
        x=[1, 2, 3, 4, 5, 6, 7, 8, 9]
        y=[1, 2, 3, 4, 0, 1, 7, 8, 9]
        ax.plot(x, y)
        canvas.print_png(buf)
        # response=HttpResponse(buf.getvalue(),content_type='image/png')
        fig.savefig('myapp/static/testplot2.png')
        fig.clear()
        # response['Content-Length'] = str(len(response.content))
        # return response

    def twoHistogram(self, data1, data2, featureName, featureValue, figName='plot.png'):
        try:
            fig = Figure()
            buf = io.BytesIO()
            canvas = FigureCanvas(fig)
            ax = fig.add_subplot(111)
            ax.set_title('Histogram '+featureName)
            
            counts1, bins1, ignored1 = ax.hist(data1, 50, alpha = 0.4, density = False, label='Not default')
            counts2, bins2, ignored2 = ax.hist(data2, 50, alpha = 0.4, density = False, label='Default')
            ax.axvline(x=featureValue, c='red', label='dataEntered')
            if featureName in ['LIMIT_BAL', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6','BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6','PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6']:
                ax.set_yscale('log')
            ax.legend()
            canvas.print_png(buf)
            # fig.savefig('myapp/static/'+figName)
            # fig.savefig('/Users/sqdmsqdm/ML_project/mysite/myapp/static/'+figName)
            fig.savefig('static/'+figName)
            fig.clear()
        except:
            return None