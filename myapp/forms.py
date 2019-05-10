from crispy_forms.helper import FormHelper
from crispy_forms.layout import Layout, Submit
from django import forms
from .models import Snippet, CreditCardDefault
from django.core.validators import RegexValidator

class NameWidget(forms.MultiWidget):

    def __init__(self, attrs=None):
        super().__init__([
            forms.TextInput(),
            forms.TextInput()
        ], attrs)
    def decompress(self, value):
        # 'firstvalue secondvalue'
        if value:
            return value.split(' ')
        return ['A name', 'A lastname']
        # ['firstvalue', 'secondvalue']

class NameField(forms.MultiValueField):

    widget = NameWidget
    
    def __init__(self, *args, **kwargs):
        
        fields = (
            forms.CharField(validators=[
                RegexValidator(r'[a-zA-Z]+', 'Enter a valid first name')
            ]),  # test
            forms.CharField(validators=[
                RegexValidator(r'[a-zA-Z]+', 'Enter a valid second name')
            ])   # none
        )
        super().__init__(fields, *args, **kwargs)

    def compress(self, data_list):
        # data_list = ['first value', 'second value']
        return f'{data_list[0]} {data_list[1]}'
        # 'firstvalue secondvalue'

class ContactForm(forms.Form):
    name = NameField()
    email = forms.EmailField(label='E-Mail')
    category = forms.ChoiceField(choices=[('questions', 'Question'), ('other', 'Other')])
    subject = forms.CharField(required=False)
    body = forms.CharField(widget=forms.Textarea)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.helper = FormHelper
        self.helper.form_method = 'post'
        self.helper.layout = Layout(
            'name',
            'email',
            'category',
            'subject',
            'body',
            Submit('submit', 'Submit', css_class='btn-success')
        )


class SnippetForm(forms.ModelForm):

    class Meta:
        model = Snippet
        fields = {'name', 'body'}

class CcDefaultForm(forms.Form):
    
    first_form = SnippetForm()
    
    LIMIT_BAL = forms.CharField(required=True, validators=[
        RegexValidator(r'[0-9]+', 'Ingrese un valor de crédito en dólares')
    ])
    SEX = forms.ChoiceField(choices=[('1', 'Male'), ('2', 'Female')])
    EDUCATION = forms.ChoiceField(choices=[('1', 'Graduate school'), ('2', 'University'), ('3', 'High school'), ('4', 'Others'), ('5', 'Unknown1'), ('6', 'Unknown2') ])
    MARRIAGE = forms.ChoiceField(choices=[('1', 'Married'), ('2', 'Single'), ('3', 'Others')])
    AGE = forms.CharField(required=True, validators=[
        RegexValidator(r'[0-9]+', 'Ingrese su edad en años')
    ])
    PAY_0 = forms.CharField(required=False, validators=[
        RegexValidator(r'[0-9]+', 'Ingrese el retraso en pago el primer mes')
    ])
    PAY_2 = forms.CharField(required=False)
    PAY_3 = forms.CharField(required=False)
    PAY_4 = forms.CharField(required=False)
    PAY_5 = forms.CharField(required=False)
    PAY_6 = forms.CharField(required=False)
    BILL_AMT1 = forms.CharField(required=False)
    BILL_AMT2 = forms.CharField(required=False)
    BILL_AMT3 = forms.CharField(required=False)
    BILL_AMT4 = forms.CharField(required=False)
    BILL_AMT5 = forms.CharField(required=False)
    BILL_AMT6 = forms.CharField(required=False)
    PAY_AMT1 = forms.CharField(required=False)
    PAY_AMT2 = forms.CharField(required=False)
    PAY_AMT3 = forms.CharField(required=False)
    PAY_AMT4 = forms.CharField(required=False)
    PAY_AMT5 = forms.CharField(required=False)
    PAY_AMT6 = forms.CharField(required=False)
    
    def __init__(self, *args, **kwargs):
        
        self.helper = FormHelper
        self.helper.form_method = 'post'
        self.helper.form_action = 'visualizedata'
        self.helper.layout = Layout(
            'LIMIT_BAL',
            'SEX',
            'EDUCATION',
            'MARRIAGE',
            'AGE',
            'PAY_0',
            'PAY_2',
            'PAY_3',
            'PAY_4',
            'PAY_5',
            'PAY_6',
            'BILL_AMT1',
            'BILL_AMT2',
            'BILL_AMT3',
            'BILL_AMT4',
            'BILL_AMT5',
            'BILL_AMT6',
            'PAY_AMT1',
            'PAY_AMT2',
            'PAY_AMT3',
            'PAY_AMT4',
            'PAY_AMT5',
            'PAY_AMT6',
            Submit('submit1', 'Predicción de incumplimiento', css_class='btn-danger'),
            Submit('submit2', 'Visualizar datos', css_class='btn-success')
        )
        super().__init__(*args, **kwargs)

        
    class Meta:
        model = CreditCardDefault
        fields = '__all__'
