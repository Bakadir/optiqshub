from django.shortcuts import render
from .forms import *


import numpy as np
import matplotlib.pyplot as plt
import io
import base64
import plotly.graph_objects as go
import json
import plotly.utils
import plotly.io as pio

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from django.shortcuts import render
from .forms import MichelsonInterferometerForm

def michelson_interferometers(request):
    if request.method == 'POST':
        form = MichelsonInterferometerForm(request.POST)
        if form.is_valid():
            # Extract and convert form data
            wavelength = form.cleaned_data['wavelength'] * 1e-9  # nm → m
            size = form.cleaned_data['grid_size'] * 1e-3  # mm → m
            N = form.cleaned_data['resolution']
            R = form.cleaned_data['laser_radius'] * 1e-3  # mm → m
            z1 = form.cleaned_data['arm1_length'] * 1e-2  # cm → m
            z2 = form.cleaned_data['arm2_length'] * 1e-2  # cm → m
            z3 = form.cleaned_data['laser_to_beamsplitter'] * 1e-2  # cm → m
            z4 = form.cleaned_data['beamsplitter_to_screen'] * 1e-2  # cm → m
            Rbs = form.cleaned_data['beamsplitter_reflection']
            tx = form.cleaned_data['tilt_x'] * 1e-3  # mrad → rad
            ty = form.cleaned_data['tilt_y'] * 1e-3  # mrad → rad

            # Wavenumber and grid setup
            k = 2 * np.pi / wavelength
            x = np.linspace(-size/2, size/2, N)
            y = np.linspace(-size/2, size/2, N)
            xv, yv = np.meshgrid(x, y)
            dx = x[1] - x[0]

            # Gaussian beam
            beam = np.exp(-(xv**2 + yv**2) / (R**2))

            # Angular spectrum propagation
            def propagate(field, z):
                fx = np.fft.fftfreq(N, dx)
                fy = np.fft.fftfreq(N, dx)
                FX, FY = np.meshgrid(fx, fy)
                kz = np.sqrt(k**2 - (2*np.pi*FX)**2 - (2*np.pi*FY)**2 + 0j)
                F_ft = np.fft.fft2(field)
                F_ft *= np.exp(1j * kz * z)
                return np.fft.ifft2(F_ft)

            # Propagate to beamsplitter
            F = propagate(beam, z3)

            # Split beams
            F1 = np.sqrt(Rbs) * F  # Reflected (M1)
            F2 = np.sqrt(1 - Rbs) * F  # Transmitted (M2)

            # Propagate M1 (round trip with tilt)
            F1 = propagate(F1, z1)
            F1 *= np.exp(1j * (tx * xv + ty * yv))  # Apply tilt
            F1 = propagate(F1, z1)  # Return trip
            F1 *= np.sqrt(1 - Rbs)  # Attenuation
            F1 *= np.exp(1j * np.pi/2)  # π/2 phase shift

            # Prepare animation frames for moving M2
            z2_values = np.linspace(0, z2, 10)
            frames = []
            for z2_current in z2_values:
                F2_temp = np.sqrt(1 - Rbs) * F  # Start fresh for each frame
                F2_temp = propagate(F2_temp, z2_current)
                F2_temp = propagate(F2_temp, z2_current)
                F2_temp *= np.sqrt(Rbs)
                F_final = F1 + F2_temp
                F_screen = propagate(F_final, z4)
                I = np.abs(F_screen)**2
                I = (I - np.min(I)) / (np.max(I) - np.min(I))
                frames.append(go.Frame(data=[go.Heatmap(
                    z=I, x=xv[0]*1e3, y=yv[:,0]*1e3, colorscale='Inferno'
                )], name=f"{z2_current*100:.2f} cm"))

            # Initial frame
            initial_I = np.abs(propagate(F1 + F2, z4))**2
            initial_I = (initial_I - np.min(initial_I)) / (np.max(initial_I) - np.min(initial_I))

            # Create interactive Plotly figure with slider
            fig = go.Figure(
                data=[go.Heatmap(
                    z=initial_I, x=xv[0]*1e3, y=yv[:,0]*1e3, colorscale='Inferno'
                )],
                frames=frames
            )

            fig.update_layout(
                title="Michelson Interference Pattern — Mirror M₂ Motion",
                xaxis=dict(title="X (mm)", scaleanchor="y"),
                yaxis=dict(title="Y (mm)"),
                width=600,
                height=600,
                sliders=[{
                    "steps": [
                        {
                            "args": [[f"{z2_current*100:.2f} cm"], {"frame": {"duration": 0, "redraw": True}}],
                            "label": f"{z2_current*100:.2f}",
                            "method": "animate"
                        }
                        for z2_current in z2_values
                    ],
                    "transition": {"duration": 0},
                    "x": 0.1,
                    "xanchor": "left",
                    "y": -0.1,
                    "currentvalue": {"prefix": "OM₂ (cm): ", "font": {"size": 14}},
                }]
            )

            plot_data = pio.to_json(fig)
            return render(request, 'interferometers/michelson.html', {'plot_data': plot_data, 'form': form})
    else:
        form = MichelsonInterferometerForm()
    return render(request, 'interferometers/michelson.html', {'form': form})


def michelson_interferometer(request):
    if request.method == 'POST':
        form = MichelsonInterferometerForm(request.POST)
        if form.is_valid():
            # Extract and convert form data
            wavelength = form.cleaned_data['wavelength'] * 1e-9  # nm → m
            size = form.cleaned_data['grid_size'] * 1e-3  # mm → m
            N = form.cleaned_data['resolution']
            R = form.cleaned_data['laser_radius'] * 1e-3  # mm → m
            z1 = form.cleaned_data['arm1_length'] * 1e-2  # cm → m
            z2 = form.cleaned_data['arm2_length'] * 1e-2  # cm → m
            z3 = form.cleaned_data['laser_to_beamsplitter'] * 1e-2  # cm → m
            z4 = form.cleaned_data['beamsplitter_to_screen'] * 1e-2  # cm → m
            Rbs = form.cleaned_data['beamsplitter_reflection']
            tx = form.cleaned_data['tilt_x'] * 1e-3  # mrad → rad
            ty = form.cleaned_data['tilt_y'] * 1e-3  # mrad → rad

            # Wavenumber and grid setup
            k = 2 * np.pi / wavelength
            x = np.linspace(-size/2, size/2, N)
            y = np.linspace(-size/2, size/2, N)
            xv, yv = np.meshgrid(x, y)
            dx = x[1] - x[0]

            # Gaussian beam
            beam = np.exp(-(xv**2 + yv**2) / (R**2))

            # Angular spectrum propagation
            def propagate(field, z):
                fx = np.fft.fftfreq(N, dx)
                fy = np.fft.fftfreq(N, dx)
                FX, FY = np.meshgrid(fx, fy)
                kz = np.sqrt(k**2 - (2*np.pi*FX)**2 - (2*np.pi*FY)**2 + 0j)
                F_ft = np.fft.fft2(field)
                F_ft *= np.exp(1j * kz * z)
                return np.fft.ifft2(F_ft)

            # Propagate to beamsplitter
            F = propagate(beam, z3)

            # Split beams
            F1 = np.sqrt(Rbs) * F  # Reflected (M1)
            F2 = np.sqrt(1 - Rbs) * F  # Transmitted (M2)

            # Propagate M1 (round trip with tilt)
            F1 = propagate(F1, z1)
            F1 *= np.exp(1j * (tx * xv + ty * yv))  # Apply tilt
            F1 = propagate(F1, z1)  # Return trip
            F1 *= np.sqrt(1 - Rbs)  # Attenuation

            # Propagate M2 (round trip)
            F2 = propagate(F2, z2)
            F2 = propagate(F2, z2)  # Return trip
            F2 *= np.sqrt(Rbs)  # Attenuation

            # Recombine with phase shift
            F1 *= np.exp(1j * np.pi/2)  # π/2 phase shift
            F_final = F1 + F2

            # Propagate to screen
            F_final = propagate(F_final, z4)

            # Compute intensity (log scale)
            I = np.abs(F_final)**2 
            I = (I - np.min(I)) / (np.max(I) - np.min(I))  # Normalize to [0, 1]

            # Create heatmap
            fig = go.Figure(data=go.Heatmap(
                z=I, x=xv[0]*1e3, y=yv[:,0]*1e3, colorscale='Inferno'
            ))
            fig.update_layout(
                title="Michelson Interference Pattern",
                xaxis=dict(title="X (mm)", scaleanchor="y"),
                yaxis=dict(title="Y (mm)"),
                width=600,
                height=600,
            )
            plot_data = pio.to_json(fig)
            context = {'plot_data': plot_data, 'form': form}
            #return render(request, 'interferometers/michelson.html', {'plot_data': plot_data, 'form': form})
        else:
            context = {'form': form}
    else:
        form = MichelsonInterferometerForm()
        context = {'form': form}

    if request.headers.get("HX-Request") == "true":
        return render(request, "interferometers/partials/michelson_results.html",context)
    else:
    
        return render(request, 'interferometers/michelson.html', {'form': form})

def run_michelson_simulation(wavelength, size, N, R, z1, z2, z3, z4, Rbs, tx, ty):
    """Simulates the Michelson Interferometer and returns the intensity pattern."""
    
    # Create computational grid
    dx = size / N  # Pixel size
    k = 2 * np.pi / wavelength  # Wavenumber
    x = np.linspace(-size/2, size/2, N)
    X, Y = np.meshgrid(x, x)

    # Generate Gaussian beam
    beam = np.exp(-(X**2 + Y**2) / (R**2))

    # Fresnel propagation function (Angular Spectrum)
    def propagate(field, z):
        fx = np.fft.fftfreq(N, dx)
        FX, FY = np.meshgrid(fx, fx)
        kz = np.sqrt(k**2 - (2*np.pi*FX)**2 - (2*np.pi*FY)**2, dtype=np.complex128)
        F_ft = np.fft.fft2(field)
        F_ft *= np.exp(1j * kz * z)
        return np.fft.ifft2(F_ft)

    # Propagate to the beamsplitter
    F = propagate(beam, z3)

    # Split beam: Reflective part (M1) and Transmitted part (M2)
    F1 = np.sqrt(Rbs) * F
    F2 = np.sqrt(1 - Rbs) * F

    # Propagate to and from M1 (Mirror 1)
    F1 = propagate(F1, 2 * z1)  # Round-trip
    F1 *= np.sqrt(1 - Rbs)  # Attenuation at the beamsplitter

    # Propagate to and from M2 (Mirror 2) with tilt
    F2 = propagate(F2, z2)
    F2 *= np.exp(1j * (tx * X + ty * Y))  # Apply tilt phase shift
    F2 = propagate(F2, z2)  # Round-trip
    F2 *= np.sqrt(Rbs)  # Attenuation at the beamsplitter

    # Recombine beams at the beamsplitter
    F_final = F1 + F2

    # Propagate to the screen
    F_final = propagate(F_final, z4)

    # Compute intensity
    return np.abs(F_final) ** 2


def fabry_perot_interferometer(request):
    return render(request, 'interferometers/fabry_perot.html')

def mach_zehnder_interferometer(request):
    return render(request, 'interferometers/mach_zehnder.html')
