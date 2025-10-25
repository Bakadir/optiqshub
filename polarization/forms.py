from django import forms
from django.forms import formset_factory

class VectorForm(forms.Form):
    VECTOR_CHOICES = [
        ('Linear Polarization', 'Linear Polarization'),
        ('Left Circular Polarization', 'Left Circular Polarization'),
        ('Right Circular Polarization', 'Right Circular Polarization'),
        ('Elliptical Polarization', 'Elliptical Polarization'),
    ]

    vector_type = forms.ChoiceField(choices=VECTOR_CHOICES)
    angle = forms.FloatField(required=False, initial=0)
    azimuth = forms.FloatField(required=False, initial=0)
    elliptic_angle = forms.FloatField(required=False, initial=0)

class StokesVectorForm(forms.Form):
    VECTOR_CHOICES = [
        ('Unpolarized', 'Unpolarized'),
        ('Linear Polarization', 'Linear Polarization'),
        ('Left Circular Polarization', 'Left Circular Polarization'),
        ('Right Circular Polarization', 'Right Circular Polarization'),
        ('Elliptical Polarization', 'Elliptical Polarization'),
    ]

    vector_type = forms.ChoiceField(choices=VECTOR_CHOICES)

    angle = forms.FloatField(required=False, initial=0)

    azimuth = forms.FloatField(required=False, initial=0)
    elliptic_angle = forms.FloatField(required=False, initial=0)
    degree_of_polarization = forms.FloatField(required=False, initial=1)



class MatrixForm(forms.Form):
    MATRIX_CHOICES = [
        ('Linear Polarizer', 'Linear Polarizer'),
        ('Retarder', 'Retarder'),
        ('Attenuator', 'Attenuator'),
        ('Mirror', 'Mirror'),
        ('Quarter Wave Plate', 'Quarter Wave Plate'),
        ('Half Wave Plate', 'Half Wave Plate'),
        ('Fresnel Reflection', 'Fresnel Reflection'),
        ('Fresnel Transmission', 'Fresnel Transmission'),
    ]

    matrix_type = forms.ChoiceField(choices=MATRIX_CHOICES)
    matrix_angle = forms.FloatField(required=False, initial=0)
    fast_axis_angle = forms.FloatField(required=False, initial=0)
    retardance = forms.FloatField(required=False, initial=0)
    optical_density = forms.FloatField(required=False, initial=0)
    im_index_of_refraction = forms.FloatField(required=False, initial=0)
    re_index_of_refraction = forms.FloatField(required=False, initial=1)
    incidence_angle = forms.FloatField(required=False, initial=0)

# Create a formset from the JonesMatrixForm
MatrixFormSet = formset_factory(MatrixForm, extra=1)



    


