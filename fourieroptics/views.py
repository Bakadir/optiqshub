import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import plotly.graph_objects as go
import plotly.io as pio
from scipy.fft import fft2, ifft2, fftfreq, fftshift
from scipy import fft
import shutil

from django.shortcuts import render, redirect
from django.http import JsonResponse, HttpResponse, FileResponse
from django import forms

from .forms import *

def download_animation(request):
    file_path = 'static/fourieroptics/rgb_animation.gif'
    return FileResponse(open(file_path, 'rb'), as_attachment=True, filename='rgb_animation.gif')
def download_animation_intensity(request):
    file_path = 'static/fourieroptics/intensity_animation.gif'
    return FileResponse(open(file_path, 'rb'), as_attachment=True, filename='intensity_animation.gif')

from django.http import JsonResponse
from PIL import Image
from pathlib import Path
import time

from PIL import Image
import numpy as np
from pathlib import Path
import os
import shutil
from django.core.files import File
def ApertureFromImage(image_path, Nx, Ny):
    """
    Resizes an image to fit within an aperture of a given size and inserts it into a black screen
    of size (Nx, Ny).

    Parameters:
        image_path (str): Path to the image file.
        Nx (int): Number of pixels in the x-dimension of the screen.
        Ny (int): Number of pixels in the y-dimension of the screen.
        qwt_width_mm (float): Width of the aperture in millimeters.
        qwt_height_mm (float): Height of the aperture in millimeters.
        screen_width_mm (float): Physical width of the screen in millimeters.
        screen_height_mm (float): Physical height of the screen in millimeters.

    Returns:
        np.ndarray: Grayscale aperture with the resized image centered on a black background.
    """
    # Load the image
    img = Image.open(Path(image_path))
    img = img.convert("RGB")

    # Convert the desired aperture size from millimeters to pixels
    
    # Rescale the image to the desired aperture size in pixels
    rescaled_img = img.resize((Nx, Ny), Image.Resampling.LANCZOS)
    imgRGB = np.asarray(rescaled_img) / 255.0

    # Convert the image to grayscale
    t = 0.2990 * imgRGB[:, :, 0] + 0.5870 * imgRGB[:, :, 1] + 0.1140 * imgRGB[:, :, 2]

 
    # Flip vertically if required (depends on coordinate system)
    t = np.flip(t, axis=0)

    return t

import uuid
def home(request):
    
 
    if request.method == "POST":
        form = FourierDiff(request.POST)
        if form.is_valid() :
            aperture_type = form.cleaned_data['aperture_type'] 
                
            #wavelengths = [form.cleaned_data['wavelength'] * 1e-9 ]
            #intensities = [int(1.0)]
            wavelengths = []
            intensities = []

            for key, value in request.POST.items():
                if key.startswith("wavelength-"):
                    wavelengths.append(float(value)* 1e-9)
                elif key.startswith("intensity-"):
                    intensities.append(float(value)* 1e-9)

            print("Wavelengths:", wavelengths)
            print("Intensities:", intensities)


            number_of_slits = form.cleaned_data['number_of_slits']
            distance_between_slits = form.cleaned_data['distance_between_slits'] * 1e-3  
            slit_width = form.cleaned_data['slit_width'] * 1e-3 
            slit_height = form.cleaned_data['slit_height'] * 1e-3 


            aperture_radius = form.cleaned_data['aperture_radius'] * 1e-3  

            distance_screen_to_aperture = form.cleaned_data['distance_screen_to_aperture'] * 1e-2 
            screen_width = form.cleaned_data['screen_width'] * 1e-3 
            screen_height = screen_width 

            resolution = form.cleaned_data['resolution']   
            animation_frames = form.cleaned_data['animation_frames']  
            
            x = np.linspace(-screen_width / 2, screen_width / 2, resolution)
            y = np.linspace(-screen_height / 2, screen_height / 2, resolution)
            xv, yv = np.meshgrid(x, y)

            if aperture_type == 'N-Slit':
                slit_positions = np.linspace(
                    -(number_of_slits - 1) * distance_between_slits / 2,
                    (number_of_slits - 1) * distance_between_slits / 2,
                    number_of_slits
                )
                U0 = np.zeros_like(xv)
                for position in slit_positions:
                    U0 += (np.abs(xv - position) < slit_width / 2) & (np.abs(yv) < slit_height / 2)

            elif aperture_type == "Circular":
                
                U0 = (xv**2 + yv**2 <= aperture_radius**2).astype(float)
            elif aperture_type == "QWT":
                image_path = 'static/fourieroptics/QWT.png'
                
                
                U0 = ApertureFromImage(image_path, xv.shape[0], yv.shape[1])
            
            #rgb_images = []
            #intensity_images = []
            frames_heatmap = []
            frames_3d = []
            frames_lines = []
            frames_rgb = []
            
            num_frames = animation_frames # 10
            rgb_colors = [wavelength_to_rgb(wl * 1e9) for wl in wavelengths]
            # Compute images for all frames
            #k = 2 * np.pi / wl 

            Nx, Ny = U0.shape

            # Sampling intervals in x and y
            dx = xv[0, 1] - xv[0, 0]  # dx
            dy = yv[1, 0] - yv[0, 0]  # dy

            # Spatial frequency coordinates in x and y (kx and ky)
            kx = 2 * np.pi * fftfreq(Nx, dx)  # Frequency coordinates in x
            ky = 2 * np.pi * fftfreq(Ny, dy)
            kxv, kyv = np.meshgrid(kx, ky)
            A = fft2(U0)
            for frame in range(num_frames):
                screen_distance = distance_screen_to_aperture * frame / (num_frames - 1)

                # Initialize images
                rgb_image = np.zeros((*xv.shape, 3), dtype=np.float64)
                intensity_image = np.zeros_like(xv, dtype=np.float64)
                    # Convert to nm for RGB mapping
                U_total = np.zeros_like(xv, dtype=np.complex128)

                for wl, intensity, rgb_color in zip(wavelengths, intensities, rgb_colors):
                    # Compute the field at the screen for the current wavelength
                    #U_screen_temp = compute_U(U0, xv, yv, wl, screen_distance)

                    k = 2 * np.pi / wl 
                    transfer_function = np.exp(1j * screen_distance * np.sqrt(k**2 - kxv**2 - kyv**2))
                    U_screen_temp = ifft2(A * transfer_function)


                    U_total += U_screen_temp * intensity

                    # Accumulate intensity and RGB contributions
                    intensity_image += np.abs(U_screen_temp) ** 2 * intensity
                    rgb_image[:, :, 0] += np.abs(U_screen_temp) ** 2 * intensity * rgb_color[0]
                    rgb_image[:, :, 1] += np.abs(U_screen_temp) ** 2 * intensity * rgb_color[1]
                    rgb_image[:, :, 2] += np.abs(U_screen_temp) ** 2 * intensity * rgb_color[2]

                

                intensity_distribution = np.abs(U_total) ** 2
                intensity_distribution /= np.max(intensity_distribution)

                # Normalize RGB image
                max_value = np.max(rgb_image)
                if max_value > 0:  # Only normalize if max value is greater than 0
                    rgb_image /= max_value
                

                # Store precomputed images
                #intensity_images.append(intensity_distribution)
                #rgb_images.append(rgb_image)

                frame_data_heatmap = go.Heatmap(z=intensity_distribution, x=xv[0] * 1e3, y=yv[:, 0] * 1e3, colorscale='Inferno')
                frames_heatmap.append(go.Frame(data=[frame_data_heatmap], name=str(frame)))

                # 3D surface frame
                frame_data_3d = go.Surface(z=intensity_distribution, x=xv[0] * 1e3, y=yv[:, 0] * 1e3, colorscale='Inferno')
                frames_3d.append(go.Frame(data=[frame_data_3d], name=str(frame)))

                # Line plots
                intensity_y0 = intensity_distribution[len(y)//2, :]

                frame_data_lines = [
                    go.Scatter(x=x * 1e3, y=intensity_y0, mode='lines', name="y = 0"),
                ]
                frames_lines.append(go.Frame(data=frame_data_lines, name=str(frame)))
                
                rgb_image = np.flip(rgb_image, axis=0)
                
                fig_rgb = px.imshow(
                    rgb_image,
                    x=xv[0] * 1e3,  # Convert to mm
                    y=yv[:, 0] * 1e3,  # Convert to mm
                    origin='lower',
                    aspect='auto',
                )
                fig_rgb.update_layout(
                    title='RGB Intensity Distribution at the Screen Plane',
                    xaxis=dict(
                        title="X-Position [mm]",
                        scaleanchor="y"  # Ensure the x-axis and y-axis have the same scale
                    ),
                    yaxis=dict(
                        title="Y-Position [mm]",
                        autorange='reversed'  # Reverse the y-axis
                    ),
                    width=600,
                    height=600,
                )
                
                frames_rgb.append(go.Frame(data=fig_rgb.data, name=str(frame)))


            num_frames = animation_frames
            screen_distances = np.linspace(0, distance_screen_to_aperture, num_frames)

            slider_steps = [
                {
                    "args": [[str(i)], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
                    "label": f"{screen_distance * 100:.2f} cm",  # Convert to cm
                    "method": "animate"
                } for i, screen_distance in enumerate(screen_distances)
            ]

            # Heatmap animation
            fig_heatmap = go.Figure(
                data=frames_heatmap[0].data,
                layout=go.Layout(
                    title="Intensity Distribution at the Screen Plane",
                    xaxis=dict(
                        title="X-Position [mm]",
                        scaleanchor="y"  # Ensure the x-axis and y-axis have the same scale
                    ),
                    yaxis=dict(
                        title="Y-Position [mm]"
                    ),
                    width=500,
                    height=500,
                    sliders=[{
                        "currentvalue": {
                            "prefix": "Screen Distance: ",
                            "font": {"size": 12}
                        },
                        "steps": slider_steps,
                        "transition": {"duration": 300},
                        "x": 0.1,
                        "y": -0.1,  # Position the slider below the x-axis
                        "len": 1
                    }]
                ),
                frames=frames_heatmap
            )
            anim_heatmap_plot_data = pio.to_json(fig_heatmap)

            # RGB animation
            fig_rgb = go.Figure(
                data=frames_rgb[0].data,
                layout=go.Layout(
                    title="RGB Intensity Distribution at the Screen Plane",
                    xaxis=dict(
                        title="X-Position [mm]",
                        scaleanchor="y"  # Ensure the x-axis and y-axis have the same scale
                    ),
                    yaxis=dict(
                        title="Y-Position [mm]"
                    ),
                    width=500,
                    height=500,
                    sliders=[{
                        "currentvalue": {
                            "prefix": "Screen Distance: ",
                            "font": {"size": 12}
                        },
                        "steps": slider_steps,
                        "transition": {"duration": 300},
                        "x": 0.1,
                        "y": -0.1,  # Position the slider below the x-axis
                        "len": 1
                    }]
                ),
                frames=frames_rgb
            )
            anim_rgb_plot_data = pio.to_json(fig_rgb)

            # 3D surface plot animation
            fig_3d = go.Figure(
                data=frames_3d[0].data,
                layout=go.Layout(
                    title="3D Surface Plot of Intensity at the Screen Plane",
                    scene=dict(
                        xaxis_title="X-Position [mm]",
                        yaxis_title="Y-Position [mm]",
                        zaxis_title="Intensity"
                    ),
                    width=500,
                    height=500,
                    sliders=[{
                        "currentvalue": {
                            "prefix": "Screen Distance: ",
                            "font": {"size": 12}
                        },
                        "steps": slider_steps,
                        "transition": {"duration": 300},
                        "x": 0.1,
                        "y": -0.1,  # Position the slider below the x-axis
                        "len": 1
                    }]
                ),
                frames=frames_3d
            )
            anim_3d_plot_data = pio.to_json(fig_3d)

            # Line plots animation
            fig_lines = go.Figure(
                data=frames_lines[0].data,
                layout=go.Layout(
                    title="Intensity Distributions Along X=0",
                    xaxis_title="Position [mm]",
                    yaxis_title="Intensity",
                    width=500,
                    height=500,
                    sliders=[{
                        "currentvalue": {
                            "prefix": "Screen Distance: ",
                            "font": {"size": 12}
                        },
                        "steps": slider_steps,
                        "transition": {"duration": 300},
                        "x": 0.1,
                        "y": -0.1,  # Position the slider below the x-axis
                        "len": 1
                    }]
                ),
                frames=frames_lines
            )
            anim_lines_plot_data = pio.to_json(fig_lines)


            
            context = {
                "form":form, 
                'anim_heatmap_plot_data':anim_heatmap_plot_data,
                'anim_3d_plot_data':anim_3d_plot_data,
                'anim_lines_plot_data':anim_lines_plot_data,
                'anim_rgb_plot_data':anim_rgb_plot_data,
                        }

    else:
        form = FourierDiff()
        
        context = {'form': form}

    if request.headers.get("HX-Request") == "true":
        return render(request, "fourieroptics/partials/fourier_results.html",context)
    else:
        return render(request, "fourieroptics/home.html", context)
    




from django.shortcuts import render, get_object_or_404
from django.urls import reverse
from django.http import JsonResponse

import plotly.express as px

def wavelength_to_rgb(wavelength):
    gamma = 0.8
    intensity_max = 255
    factor = 0.0
    r = g = b = 0

    if (wavelength >= 380) and (wavelength < 440):
        r = -(wavelength - 440) / (440 - 380)
        g = 0.0
        b = 1.0
    elif (wavelength >= 440) and (wavelength < 490):
        r = 0.0
        g = (wavelength - 440) / (490 - 440)
        b = 1.0
    elif (wavelength >= 490) and (wavelength < 510):
        r = 0.0
        g = 1.0
        b = -(wavelength - 510) / (510 - 490)
    elif (wavelength >= 510) and (wavelength < 580):
        r = (wavelength - 510) / (580 - 510)
        g = 1.0
        b = 0.0
    elif (wavelength >= 580) and (wavelength < 645):
        r = 1.0
        g = -(wavelength - 645) / (645 - 580)
        b = 0.0
    elif (wavelength >= 645) and (wavelength <= 750):
        r = 1.0
        g = 0.0
        b = 0.0
    else:
        r = g = b = 0.0

    if (wavelength >= 380) and (wavelength < 420):
        factor = 0.3 + 0.7*(wavelength - 380) / (420 - 380)
    elif (wavelength >= 420) and (wavelength < 645):
        factor = 1.0
    elif (wavelength >= 645) and (wavelength <= 750):
        factor = 0.3 + 0.7*(750 - wavelength) / (750 - 645)
    else:
        factor = 0.0

    if r != 0:
        r = round(intensity_max * ((r * factor) ** gamma))
    if g != 0:
        g = round(intensity_max * ((g * factor) ** gamma))
    if b != 0:
        b = round(intensity_max * ((b * factor) ** gamma))

    return (r, g, b)
