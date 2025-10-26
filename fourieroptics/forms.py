from django import forms
from django.forms import formset_factory

class FourierDiff(forms.Form):
    
    APERTURE_CHOICES = [
        ('N-Slit', 'N-Slit'),
        ('Circular', 'Circular'),
    ]

    aperture_type = forms.ChoiceField(choices=APERTURE_CHOICES,widget=forms.RadioSelect,  initial='N-Slit')

    #add_lens=forms.BooleanField(required=False,initial=False)

    qwt_width = forms.FloatField(required=False,initial=1) 
    qwt_height = forms.FloatField(required=False,initial=1) 

    number_of_slits = forms.IntegerField(required=False,initial=1)  

    aperture_radius = forms.FloatField(required=False,initial=1.0)
    
    slit_width = forms.FloatField(required=False,initial=0.5) 
    slit_height = forms.FloatField(required=False,initial=2)
    distance_between_slits = forms.FloatField(required=False,initial=1)  
    
    #focal_length = forms.FloatField(required=False,initial=50)
    #distance_lens_to_aperture = forms.FloatField(required=False,initial=15) 
    #wavelength = forms.FloatField(initial=660.0)
     
    screen_width = forms.FloatField(required=False,initial=5)  
    screen_height = forms.FloatField(required=False,initial=5)  
    distance_screen_to_aperture = forms.FloatField(required=False,initial=30) 
    resolution = forms.IntegerField(initial=200, min_value=100, max_value=250 )
    animation_frames = forms.IntegerField(initial=10, min_value=2, max_value=20)
    #animation_framerate = forms.IntegerField(initial=10)

""" class WavelengthIntensityForm(forms.Form):
    wavelength = forms.FloatField(initial=660.0)
    intensity = forms.FloatField(initial=1.0)

WavelengthIntensityFormSet = formset_factory(WavelengthIntensityForm, extra=1)
 """