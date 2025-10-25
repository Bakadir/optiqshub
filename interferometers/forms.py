from django import forms

class MichelsonInterferometerForm(forms.Form):
    # Laser parameters
    wavelength = forms.FloatField(initial=632.8)  # He-Ne laser (nm)
    laser_radius = forms.FloatField(initial=1.0)  # Beam radius (mm)

    # Grid parameters
    grid_size = forms.FloatField(initial=20)  # Grid size (mm)
    resolution = forms.IntegerField(initial=512)  # Resolution (N x N)

    # Beamsplitter parameters
    beamsplitter_reflection = forms.FloatField(initial=0.5)  # Reflection coefficient (0-1)
    laser_to_beamsplitter = forms.FloatField(initial=5)  # Distance (cm)

    # Arm lengths (ensure a path difference)
    arm1_length = forms.FloatField(initial=10.0)  # Arm 1 length (cm)
    arm2_length = forms.FloatField(initial=10.1)  # Arm 2 length (cm)

    # Screen distance
    beamsplitter_to_screen = forms.FloatField(initial=20)  # Distance (cm)

    # Mirror tilt (non-zero tilt for fringes)
    tilt_x = forms.FloatField(initial=1.0)  # Tilt in x (mrad)
    tilt_y = forms.FloatField(initial=0.0)  # Tilt in y (mrad)
    
"""     # Compensating Plate Parameters
    add_plate=forms.BooleanField(required=False,initial=False)

    plate_thickness = forms.FloatField(initial=0.1, required=False)  # Thickness of the compensating plate in cm
    plate_refractive_real = forms.FloatField(initial=1.5, required=False)  # Real part of refractive index
    plate_refractive_imag = forms.FloatField(initial=0.0, required=False)  # Imaginary part of refractive index


    # Tilt Angle (in mrad)
    plate_tilt = forms.FloatField(initial=0.0)  # Tilt angle of the compensating plate in mrad
    # Distance between Plate and Mirror 1
    plate_to_mirror1 = forms.FloatField(initial=5.0)  # Distance between plate and Mirror 1 (cm) """