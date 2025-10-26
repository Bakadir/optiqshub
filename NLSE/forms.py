
from django import forms



class SimulationParameters(forms.Form):
    number_of_points = forms.IntegerField(required=False,initial=5000, min_value=100, max_value=10000)
    duration_k = forms.FloatField(required=False,initial=2**7)
    time_resolution_1 = forms.FloatField(required=False,initial=0.1)
    
    time_resolution_2 = forms.FloatField(required=False,initial=-12)

class PulseForm(forms.Form):
    PULSE_CHOICES = [
        ('sech', 'Sech pulse (Soliton)'),
        ('gaussian', 'Gaussian Pulse'),
        ('sinc', 'Sinc pulse'),
        
        ('cw', 'Continuous Wave (CW)'),
        
    ]
    add_noise=forms.BooleanField(required=False,initial=False)
    noise_amplitude=forms.FloatField(required=False,initial=10)

    pulse_type = forms.ChoiceField(choices=PULSE_CHOICES)
    amplitude = forms.FloatField(required=False,initial=1)
    chirp = forms.FloatField(required=False,initial=0)
    order = forms.IntegerField(required=False,initial=1)
    testCarrierFreq = forms.FloatField(required=False,initial=0)
class FiberParametersForm(forms.Form):
    

    
    Length_a = forms.FloatField(required=False,initial=1)
    Length_b = forms.FloatField(required=False,initial=3)

    nsteps = forms.IntegerField(required=False,initial=50 , min_value=10, max_value=100)

    gamma_a = forms.FloatField(required=False,initial=10)
    gamma_b = forms.FloatField(required=False,initial=-3)

    beta2_a = forms.FloatField(required=False,initial=-100)
    beta2_b = forms.FloatField(required=False,initial=3)

    alpha_dB_per_m_a = forms.FloatField(required=False,initial=0)
    alpha_dB_per_m_b = forms.FloatField(required=False,initial=-3)

    amplitude_charac = forms.FloatField(required=False,initial=2)
    amplitude_bool=forms.BooleanField(required=False,initial=False)
    length_bool=forms.BooleanField(required=False,initial=False)
    length_charac = forms.FloatField(required=False,initial=1.5)

