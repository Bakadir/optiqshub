from django import forms
from django.forms import formset_factory
from .models import LayerData, LEDSpectrumData, MultilayerFilmData
class LightSourceForm(forms.Form):
    led_choice = forms.ChoiceField(label='Select Layer Data', choices=[])

    def __init__(self, *args, **kwargs):
        super(LightSourceForm, self).__init__(*args, **kwargs)
        self.fields['led_choice'].choices = [(layer.id, layer.name) for layer in LEDSpectrumData.objects.all()]

class MultilayerForm(forms.Form):
    multilayer_choice = forms.ChoiceField(label='Select Layer Data', choices=[])

    def __init__(self, *args, **kwargs):
        super(MultilayerForm, self).__init__(*args, **kwargs)
        self.fields['multilayer_choice'].choices = [(layer.id, layer.name) for layer in MultilayerFilmData.objects.all()]
    
class InputForm(forms.Form):

    number_of_layers = forms.IntegerField(required=False,initial=1)
   
class Angle(forms.Form):
    incidence_angle = forms.FloatField(required=False, initial=0)
    
    def __init__(self, *args, **kwargs):
        wavelengths = kwargs.pop('wavelengths')
        super(Angle, self).__init__(*args, **kwargs)

        if wavelengths:
            WAVELENGTH_CHOICES = [(wl, str(wl)) for wl in wavelengths]
            min_wl = min(wavelengths)
            max_wl = max(wavelengths)
        else:
            WAVELENGTH_CHOICES = []
            min_wl = None
            max_wl = None
        
        self.fields['min_wavelength'] = forms.ChoiceField(choices=WAVELENGTH_CHOICES, initial=min_wl)
        self.fields['max_wavelength'] = forms.ChoiceField(choices=WAVELENGTH_CHOICES, initial=max_wl)
        #self.fields['incidence_angle'] = forms.FloatField(required=False, initial=0)
class Wavelength(forms.Form):
    
    def __init__(self, *args, **kwargs):
        wavelengths = kwargs.pop('wavelengths', [])
        super(Wavelength, self).__init__(*args, **kwargs)

        if wavelengths:
            WAVELENGTH_CHOICES = [(wl, str(wl)) for wl in wavelengths]
        else:
            WAVELENGTH_CHOICES = []
            
        
        self.fields['chosen_wavelength'] = forms.ChoiceField(choices=WAVELENGTH_CHOICES)
        ANGLE_CHOICES = [(i, str(i)) for i in range(0, 91)]
        self.fields['min_angle'] = forms.ChoiceField(choices=ANGLE_CHOICES, initial=0)
        self.fields['max_angle'] = forms.ChoiceField(choices=ANGLE_CHOICES, initial=90)
        self.fields['step_angle'] = forms.IntegerField(initial=1000, min_value=1)


class Thickness(forms.Form):
    incidence_angle_thick = forms.FloatField(required=False, initial=0)
    min_thick = forms.FloatField(required=False, initial=0)
    max_thick = forms.FloatField(required=False, initial=100)
    step_thick = forms.IntegerField(required=False, initial=1000)
    def __init__(self, *args, **kwargs):
        wavelengths = kwargs.pop('wavelengths', [])
        layers = kwargs.pop('layers', [])
        super(Thickness, self).__init__(*args, **kwargs)

        if wavelengths:
            WAVELENGTH_CHOICES = [(wl, str(wl)) for wl in wavelengths]
        else:
            WAVELENGTH_CHOICES = []
            
        
        self.fields['chosen_wavelength_thick'] = forms.ChoiceField(choices=WAVELENGTH_CHOICES)
        if layers:
            LAYER_CHOICES = [(index, f"Layer {index + 1} - {layer['book']}") for index, layer in enumerate(layers)]
            last_layer = layers[-1]
            LAYER_CHOICES[-1] = (len(layers) - 1, f"Substrate - {last_layer['book']}")
        else:
            LAYER_CHOICES = []
        
        self.fields['chosen_layer'] = forms.ChoiceField(choices=LAYER_CHOICES, initial=len(layers) - 1)

class LayerForm(forms.Form):
        
    book = forms.CharField(max_length=1000, required=False)
    #forms.ChoiceField(choices=[('', '--Select Material--')] + [(name, name) for name in BOOK_NAMES], required=False)
    page = forms.CharField(max_length=1000, required=False)
    thickness = forms.FloatField(required=False, initial=1)

MatrixFormSet = formset_factory(LayerForm, extra=1)

class LayerFormSet(forms.Form):
    name = forms.ChoiceField(choices=[], required=False)
    thickness = forms.FloatField(required=False, initial=1)
    def __init__(self, *args, **kwargs):
        # Call the superclass' __init__ method
        super().__init__(*args, **kwargs)
        
        # Update choices for the name field
        choices = [(layer.id, layer.name) for layer in LayerData.objects.all()]
        self.fields['name'].choices = choices
MatrixLayerFormSet = formset_factory(LayerFormSet, extra=1)

class LayerDataForm(forms.Form):
    name = forms.CharField(max_length=100,required=False, initial='')
    data_file = forms.FileField(required=False, initial='')
    description = forms.CharField(max_length=200,required=False, initial='')

class GraphWavelength(forms.Form):
    
    def __init__(self, *args, **kwargs):
        wavelengths = kwargs.pop('wavelengths')
        super(GraphWavelength, self).__init__(*args, **kwargs)

        if wavelengths:
            WAVELENGTH_CHOICES = [(wl, str(wl)) for wl in wavelengths]
            min_wl = min(wavelengths)
            max_wl = max(wavelengths)
        else:
            WAVELENGTH_CHOICES = []
            min_wl = None
            max_wl = None
        
        self.fields['min_wavelength'] = forms.ChoiceField(choices=WAVELENGTH_CHOICES, initial=min_wl)
        self.fields['max_wavelength'] = forms.ChoiceField(choices=WAVELENGTH_CHOICES, initial=max_wl)
        self.fields['incidence_angle'] = forms.FloatField(required=False, initial=0)
class GraphAngle(forms.Form):
    
    def __init__(self, *args, **kwargs):
        wavelengths = kwargs.pop('wavelengths', [])
        super(GraphAngle, self).__init__(*args, **kwargs)

        if wavelengths:
            WAVELENGTH_CHOICES = [(wl, str(wl)) for wl in wavelengths]
        else:
            WAVELENGTH_CHOICES = []
            
        
        self.fields['chosen_wavelength'] = forms.ChoiceField(choices=WAVELENGTH_CHOICES)
        ANGLE_CHOICES = [(i, str(i)) for i in range(0, 91)]
        self.fields['min_angle'] = forms.ChoiceField(choices=ANGLE_CHOICES, initial=0)
        self.fields['max_angle'] = forms.ChoiceField(choices=ANGLE_CHOICES, initial=90)
        self.fields['step_angle'] = forms.IntegerField(initial=1000, min_value=1)

class GraphThickness(forms.Form):
    incidence_angle_thick = forms.FloatField(required=False, initial=0)
    min_thick = forms.FloatField(required=False, initial=0)
    max_thick = forms.FloatField(required=False, initial=100)
    step_thick = forms.IntegerField(required=False, initial=1000)
    def __init__(self, *args, **kwargs):
        wavelengths = kwargs.pop('wavelengths', [])
        layers = kwargs.pop('layers', [])
        super(GraphThickness, self).__init__(*args, **kwargs)

        if wavelengths:
            WAVELENGTH_CHOICES = [(wl, str(wl)) for wl in wavelengths]
        else:
            WAVELENGTH_CHOICES = []
            
        
        self.fields['chosen_wavelength_thick'] = forms.ChoiceField(choices=WAVELENGTH_CHOICES)
        if layers:
            LAYER_CHOICES = [(index, f"Layer {index + 1} - {layer.name}") for index, layer in enumerate(layers)]
            last_layer = layers[-1]
            LAYER_CHOICES[-1] = (len(layers) - 1, f"Substrate - {last_layer.name}")
        else:
            LAYER_CHOICES = []
        
        self.fields['chosen_layer'] = forms.ChoiceField(choices=LAYER_CHOICES, initial=len(layers) - 1)