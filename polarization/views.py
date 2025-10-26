

from django.shortcuts import render



# views.py
from django.forms import formset_factory
from .forms import VectorForm, MatrixForm, MatrixFormSet,StokesVectorForm
import os
import numpy as np
import matplotlib.pyplot as plt

import matplotlib
from matplotlib import animation
from matplotlib.animation import PillowWriter


import plotly.graph_objects as go
from plotly.offline import plot
matplotlib.use('Agg')

def get_jones_vector(vector_type, angle, azimuth, elliptic_angle):

    if vector_type == 'Linear Polarization':

        jones_vector = np.array([np.cos(angle), np.sin(angle)])

    elif vector_type == 'Left Circular Polarization':

        jones_vector = 1 / np.sqrt(2) * np.array([1, -1j])

    elif vector_type == 'Right Circular Polarization':

        jones_vector = 1 / np.sqrt(2) * np.array([1, 1j])

    elif vector_type == 'Elliptical Polarization':
    
        A = np.cos(elliptic_angle)
        B = np.sin(elliptic_angle)
        C = np.cos(azimuth)
        D = np.sin(azimuth)
        J = np.array([C * A - D * B * 1j, D * A + C * B * 1j])
        jones_vector = J * np.exp(1j * (- np.angle(J[0])))

    return jones_vector

def get_jones_matrix(matrix_type,matrix_angle,retarder_fast_axis_angle,fast_axis_angle,retardance,optical_density,index_of_refraction,incidence_angle): 
    if matrix_type == 'Linear Polarizer':
        jones_matrix = np.array([[np.cos(matrix_angle)**2, np.sin(matrix_angle) * np.cos(matrix_angle)],
                                 [np.sin(matrix_angle) * np.cos(matrix_angle), np.sin(matrix_angle)**2]])
    elif matrix_type == 'Retarder':
        P = np.exp(+retardance / 2 * 1j)
        Q = np.exp(-retardance / 2 * 1j)
        D = np.sin(retardance / 2) * 2j
        C = np.cos(retarder_fast_axis_angle)
        S = np.sin(retarder_fast_axis_angle)
        jones_matrix = np.array([[C * C * P + S * S * Q, C * S * D],
                                 [C * S * D, C * C * Q + S * S * P]])
 
    elif matrix_type == 'Attenuator':
        f = np.sqrt(optical_density)
        jones_matrix = np.array([[f, 0], [0, f]])

    elif matrix_type == 'Mirror':
        jones_matrix = np.array([[1, 0], [0, -1]])

    elif matrix_type == 'Quarter Wave Plate':
        retardance = np.pi / 2
        A = np.exp(+retardance / 2 * 1j)
        B = np.exp(-retardance / 2 * 1j)
        D = np.sin(retardance / 2) * 2j
        C = np.cos(fast_axis_angle)
        S = np.sin(fast_axis_angle)
        jones_matrix = np.array([[C * C * A + S * S * B, C * S * D],
                                 [C * S * D, C * C * B + S * S * A]])
   

    elif matrix_type == 'Half Wave Plate':
        retardance = np.pi
        A = np.exp(+retardance / 2 * 1j)
        B = np.exp(-retardance / 2 * 1j)
        D = np.sin(retardance / 2) * 2j
        C = np.cos(fast_axis_angle)
        S = np.sin(fast_axis_angle)
        jones_matrix = np.array([[C * C * A + S * S * B, C * S * D],
                                 [C * S * D, C * C * B + S * S * A]])
    elif matrix_type == 'Fresnel Reflection':
        jones_matrix = np.array([[r_par(index_of_refraction, incidence_angle, 1), 0],
                                 [0, r_per(index_of_refraction, incidence_angle, 1)]])

    elif matrix_type == 'Fresnel Transmission':
    
        tpar = t_par(index_of_refraction, incidence_angle,1)
        tper = t_per(index_of_refraction, incidence_angle,1)
        jones_matrix = np.array([[tper, 0], [0, tpar]])

    return jones_matrix

def get_stokes_vector(vector_type,angle,azimuth,elliptic_angle,DOP): 
    if vector_type == 'Linear Polarization':
        stokes_vector = np.array([1,np.cos(2 * angle),np.sin(2 * angle),0])
    elif vector_type == 'Left Circular Polarization':
        stokes_vector = np.array([1, 0, 0, -1])
    elif vector_type == 'Right Circular Polarization':
        stokes_vector = np.array([1, 0, 0, 1])
    elif vector_type == 'Elliptical Polarization':
        omega = np.arctan(elliptic_angle)
        cw = np.cos(2 * omega)
        sw = np.sin(2 * omega)
        ca = np.cos(2 * azimuth)
        sa = np.sin(2 * azimuth)

        unpolarized = np.array([1 - DOP, 0, 0, 0])
        polarized = DOP * np.array([1, cw * ca, cw * sa, sw])
        stokes_vector = unpolarized + polarized

    elif vector_type == 'Unpolarized':
        stokes_vector = np.array([1, 0, 0, 0])
    return stokes_vector

def get_mueller_matrix(matrix_type,matrix_angle,retarder_fast_axis_angle,fast_axis_angle,retardance,optical_density,index_of_refraction,incidence_angle):
    if matrix_type == 'Linear Polarizer':
        C2 = np.cos(2 * matrix_angle)
        S2 = np.sin(2 * matrix_angle)
        lp = np.array( [[1, C2, S2, 0],
                        [C2, C2**2, C2 * S2, 0],
                        [S2, C2 * S2, S2 * S2, 0],
                        [0, 0, 0, 0]])
        mueller_matrix = 0.5 * lp
    elif matrix_type == 'Retarder':
        C2 = np.cos(2 * retarder_fast_axis_angle)
        S2 = np.sin(2 * retarder_fast_axis_angle)
        C = np.cos(retardance)
        S = np.sin(retardance)
        mueller_matrix =  np.array([[1, 0, 0, 0],
                                    [0, C2**2 + C * S2**2, (1 - C) * S2 * C2, -S * S2],
                                    [0, (1 - C) * C2 * S2, S2**2 + C * C2**2, S * C2],
                                    [0, S * S2, -S * C2, C]])
    elif matrix_type == 'Attenuator':
        mueller_matrix = np.array([ [optical_density, 0, 0, 0],
                                    [0, optical_density, 0, 0],
                                    [0, 0, optical_density, 0],
                                    [0, 0, 0, optical_density]])
    elif matrix_type == 'Mirror':
        mueller_matrix = np.array([ [1, 0, 0, 0],
                                    [0, 1, 0, 0],
                                    [0, 0, -1, 0],
                                    [0, 0, 0, -1]])
    elif matrix_type == 'Quarter Wave Plate':
        C2 = np.cos(2 * fast_axis_angle)
        S2 = np.sin(2 * fast_axis_angle)
        mueller_matrix =  np.array([[1, 0, 0, 0],
                                    [0, C2**2, C2 * S2, -S2],
                                    [0, C2 * S2, S2 * S2, C2],
                                    [0, S2, -C2, 0]])
    elif matrix_type == 'Half Wave Plate':
        C2 = np.cos(2 * fast_axis_angle)
        S2 = np.sin(2 * fast_axis_angle)
        mueller_matrix = np.array([     [1, 0, 0, 0],
                                    [0, C2**2 - S2**2, 2 * C2 * S2, 0],
                                    [0, 2 * C2 * S2, S2 * S2 - C2**2, 0],
                                    [0, 0, 0, -1]])
    elif matrix_type == 'Fresnel Reflection':
        
        R_p = R_par(index_of_refraction, incidence_angle, 1)
        R_s = R_per(index_of_refraction, incidence_angle, 1)
        mueller_matrix = np.array([
        [0.5 * (R_p + R_s), 0.5 * (R_p - R_s), 0, 0],
        [0.5 * (R_p - R_s), 0.5 * (R_p + R_s), 0, 0],
        [0, 0, np.sqrt(R_p * R_s), 0],
        [0, 0, 0, np.sqrt(R_p * R_s)]
    ])

    elif matrix_type == 'Fresnel Transmission':
        
        T_p = T_par(index_of_refraction, incidence_angle, 1)
        T_s = T_per(index_of_refraction, incidence_angle, 1)

        mueller_matrix = np.array([
            [0.5 * (T_p + T_s), 0.5 * (T_p - T_s), 0, 0],
            [0.5 * (T_p - T_s), 0.5 * (T_p + T_s), 0, 0],
            [0, 0, np.sqrt(T_p * T_s), 0],
            [0, 0, 0, np.sqrt(T_p * T_s)]
        ])

    return mueller_matrix

def get_jones_stateinfo(J,after_optical_element,num):
    #ellipse_azimuth
    Ex0, Ey0 = np.abs(J)
    delta = np.angle(J[..., 1]) - np.angle(J[..., 0])
    numer = 2 * Ex0 * Ey0 * np.cos(delta)
    denom = Ex0**2 - Ey0**2
    ellipse_azimuth = 0.5 * np.arctan2(numer, denom)
    #ellipse_axes
    Ex0, Ey0 = np.abs(J)
    alpha = ellipse_azimuth
    C = np.cos(alpha)
    S = np.sin(alpha)
    asqr = (Ex0 * C)**2 + (Ey0 * S)**2 + 2 * Ex0 * Ey0 * C * S * np.cos(delta)
    bsqr = (Ex0 * S)**2 + (Ey0 * C)**2 - 2 * Ex0 * Ey0 * C * S * np.cos(delta)
    a = np.sqrt(abs(asqr))
    b = np.sqrt(abs(bsqr))
    if a < b:
        ellipse_axes = round(b,3), round(a,3)
        if abs(b) >= abs(a):
            epsilon = np.arctan2(a, b)
        else:
            epsilon = np.arctan2(b, a)

        if delta < 0:
            ellipticity = -epsilon
        else:
            ellipticity = epsilon
    else:
        ellipse_axes = round(a,3),round(b,3)

    #ellipticity
        if abs(a) >= abs(b):
            epsilon = np.arctan2(b, a)
        else:
            epsilon = np.arctan2(a, b)

        if delta < 0:
            ellipticity = -epsilon
        else:
            ellipticity = epsilon

 

    return  {
            "polarization": after_optical_element,
            "animation":f'{num}_jones_animation.gif',
            "jones_vector": np.round(J,3).tolist(),
            "intensity": round(abs(J[..., 0])**2 + abs(J[..., 1])**2, 3),
            "phase": round(np.degrees(np.angle(J[..., 1]) - np.angle(J[..., 0])), 3),
            "ellipse_azimuth": round(np.degrees(ellipse_azimuth), 3),
            "ellipticity": round(np.degrees(ellipticity), 3),
            "ellipse_axes": ellipse_axes,
        }

def jones_to_stokes(J):
    Ex = abs(J[0])
    Ey = abs(J[1])
    phi = np.angle(J[1]) - np.angle(J[0])

    S0 = Ex**2 + Ey**2
    S1 = Ex**2 - Ey**2
    S2 = 2 * Ex * Ey * np.cos(phi)
    S3 = 2 * Ex * Ey * np.sin(phi)
    return np.array([S0, S1, S2, S3])

def stokes_to_jones(S):

    if S[0] == 0:
        return np.array([0, 0])

    Ip = np.sqrt(S[1]**2 + S[2]**2 + S[3]**2)

    Q = S[1] / Ip
    U = S[2] / Ip
    V = S[3] / Ip
    E_0 = np.sqrt(Ip)

    if Q == -1:
        return np.array([0, E_0])

    A = np.sqrt((1 + Q) / 2)
    J = E_0 * np.array([A, complex(U, V) / (2 * A)])

    return J

def jonescalculus(request):
    
    if request.method == 'POST':

        

        vector_type = request.POST.get('vector_type')

        angle = np.radians(float(request.POST.get('angle', 0)))
        azimuth = np.radians(float(request.POST.get('azimuth', 0)))
        elliptic_angle = np.radians(float(request.POST.get('elliptic_angle', 0)))

  

        jones_vector = get_jones_vector(vector_type,angle,azimuth,elliptic_angle)

        directory_path = "media/polarization/"
        if os.path.exists(directory_path):
            for filename in os.listdir(directory_path):
                file_path = os.path.join(directory_path, filename)
                os.unlink(file_path)

        input_filename = 'media/polarization/0_jones_animation.gif'
        fig, ax = plt.subplots(figsize=(8, 8))

        ani = matplotlib.animation.FuncAnimation(fig, animation_update,
                                    frames=np.linspace(0, -2 * np.pi,64),
                                    fargs=(jones_vector, ax))
        ani.save(input_filename, writer=PillowWriter(fps=30))
        plt.close()
        animation = True
        J1 = jones_vector
        input_polarization = f"input : {vector_type}"
        if vector_type == 'Linear Polarization':
            input_polarization = f"input : {vector_type} at {request.POST.get('angle', 0)}°"
        
        elif vector_type == 'Elliptical Polarization':
            input_polarization = f"input : {vector_type} with azimut = {request.POST.get('azimuth', 0)}° & elliptic angle = {request.POST.get('elliptic_angle', 0)}°"
        
        
        states = []
        
        states.append(get_jones_stateinfo(J1,input_polarization,0))
        
        animation = True
        jones_vecs =[jones_vector]

        output_jones_vector = jones_vector
        num=1


        matrix_rows = []
        i = 1
        while f'matrix_type-{i}' in request.POST:
            matrix_type = request.POST.get(f'matrix_type-{i}')
            matrix_angle = np.radians(float(request.POST.get(f'matrix_angle-{i}', 0)))
            retarder_fast_axis_angle = np.radians(float(request.POST.get(f'retarder_fast_axis_angle-{i}', 0)))
            fast_axis_angle = np.radians(float(request.POST.get(f'fast_axis_angle-{i}', 0)))
            retardance = np.radians(float(request.POST.get(f'retardance-{i}', 0)))
            optical_density = float(request.POST.get(f'optical_density-{i}', 0))
            re_index_of_refraction = float(request.POST.get(f're_index_of_refraction-{i}', 1))
            im_index_of_refraction = float(request.POST.get(f'im_index_of_refraction-{i}', 0))

            index_of_refraction = complex(re_index_of_refraction, im_index_of_refraction) 
            incidence_angle = np.radians(float(request.POST.get(f'incidence_angle-{i}', 0)))

            jones_matrix = get_jones_matrix(matrix_type,matrix_angle,retarder_fast_axis_angle,fast_axis_angle,retardance,optical_density,index_of_refraction,incidence_angle) #,index_of_refraction,incidence_angle

            output_jones_vector = jones_matrix @ output_jones_vector
            filename =  f'media/polarization/{num}_jones_animation.gif'
            fig, ax = plt.subplots(figsize=(8, 8))

            ani = matplotlib.animation.FuncAnimation(fig, animation_update,
                                        frames=np.linspace(0, -2 * np.pi,64),
                                        fargs=(output_jones_vector, ax))
            ani.save(filename, writer=PillowWriter(fps=30))
            plt.close()
            after_optical_element = f"after {matrix_type}"
            if matrix_type == 'Linear Polarizer':
                after_optical_element = f"after {matrix_type} at {request.POST.get(f'matrix_angle-{i}', 0)}°"
            elif matrix_type == 'Mirror':
                after_optical_element = f"after {matrix_type} "

            elif matrix_type == 'Attenuator':
                after_optical_element = f"after {matrix_type} with opticat density = {optical_density}"
            elif matrix_type == 'Quarter Wave Plate' or matrix_type == 'Half Wave Plate':
                after_optical_element = f"after {matrix_type} with fast axis at {request.POST.get(f'fast_axis_angle-{i}', 0)}°"
            elif matrix_type == 'Retarder':
                after_optical_element = f"after {matrix_type} with fast axis at {request.POST.get(f'retarder_fast_axis_angle-{i}', 0)}° & retardance = {request.POST.get(f'retardance-{i}', 0)}°"
            elif matrix_type == 'Fresnel Reflection' or matrix_type == 'Fresnel Transmission':
                after_optical_element = f"after {matrix_type} with incidence angle at {request.POST.get(f'incidence_angle-{i}', 0)}° & index of refraction = {index_of_refraction}"

            J=output_jones_vector

            states.append(get_jones_stateinfo(J,after_optical_element,num))
            num+=1
            jones_vecs.append(output_jones_vector)

            i += 1

        jones_to_stokes_list = []
        for vect in jones_vecs:
            jones_to_stokes_list.append(jones_to_stokes(vect))
        jones_vecs = jones_to_stokes_list
        fig = go.Figure()

        sphere_data = draw_empty_sphere()
        fig.add_trace(sphere_data['data'][0]) 

        annotations = sphere_data['layout']['scene']['annotations']

        scatter, ann = draw_stokes_poincare(jones_vecs[0], label='input', color='red')
        fig.add_trace(scatter)
        annotations.extend(ann)

        for i in range(len(jones_vecs) - 1):
            if i == len(jones_vecs) - 2:
                scatter, ann = draw_stokes_poincare(jones_vecs[i + 1], label='output', color='green')
                fig.add_trace(scatter)
                annotations.extend(ann)
            else:
                scatter, ann = draw_stokes_poincare(jones_vecs[i + 1], label=None, color='blue')
                fig.add_trace(scatter)

            fig.add_trace(join_stokes_poincare(jones_vecs[i], jones_vecs[i + 1], color='blue', linestyle='solid'))

        fig.update_layout(
            scene=dict(
                xaxis_title='S₁',
                yaxis_title='S₂',
                zaxis_title='S₃',
                aspectmode='cube',
                
                annotations=sphere_data['layout']['scene']['annotations']
            ),
            margin=dict(l=0, r=0, b=0,t=0),
            height=800,
            
        )
        filename =  f'media/polarization/interactive_poincare_sphere_plot.html'
        fig.write_html(filename)
        plot_html = fig.to_html(full_html=False)

        

        context = { 'animation': animation,'states':states,'plot_html':plot_html}

    else:
        context = {'animation': False}

    if request.headers.get("HX-Request") == "true":
        return render(request, "polar/partials/jonescalculus_results.html",context)
    else:
        return render(request, "polar/jonescalculus.html", context)


def muellercalculus(request):
    if request.method == 'POST':

        directory_path = "media/polarization/"
        if os.path.exists(directory_path):
            for filename in os.listdir(directory_path):
                file_path = os.path.join(directory_path, filename)
                os.unlink(file_path)


        vector_type = request.POST.get('vector_type')

        angle = np.radians(float(request.POST.get('angle', 0)))
        azimuth = np.radians(float(request.POST.get('azimuth', 0)))
        elliptic_angle = np.radians(float(request.POST.get('elliptic_angle', 0)))
        dop = np.radians(float(request.POST.get('degree_of_polarization', 0)))
        

        # vector_type = vector_form.cleaned_data.get("vector_type")
        # angle = np.radians(vector_form.cleaned_data.get("angle"))
        # azimuth = np.radians(vector_form.cleaned_data.get("azimuth"))
        # elliptic_angle = np.radians(vector_form.cleaned_data.get("elliptic_angle"))
        # dop  = vector_form.cleaned_data.get("degree_of_polarization")

        stokes_vector = get_stokes_vector(vector_type,angle,azimuth,elliptic_angle,dop)

        input_filename = 'media/polarization/input_mueller_animation.gif'
        input_anim = 'input_mueller_animation.gif'

        input_polarization = f"input : {vector_type}"
        if vector_type == 'Linear Polarization':
            input_polarization = f"input : {vector_type} at {request.POST.get('angle', 0)}°"
        
        elif vector_type == 'Elliptical Polarization':
            input_polarization = f"input : {vector_type} with DOP = {request.POST.get('degree_of_polarization', 0)} & azimut = {request.POST.get('azimuth', 0)}° & elliptic angle = {request.POST.get('elliptic_angle', 0)}°"
        
        if vector_type != "Unpolarized":
    
            fig, ax = plt.subplots(figsize=(8, 8))

            ani = matplotlib.animation.FuncAnimation(fig, animation_update,
                                        frames=np.linspace(0, -2 * np.pi,64),
                                        fargs=(stokes_to_jones(stokes_vector), ax))
            ani.save(input_filename, writer=PillowWriter(fps=30))
            plt.close()
        else:
            input_anim = None
                
        states = []
        Sv = stokes_vector
        input_info = {
            "polarization": input_polarization,
            "animation": input_anim,
            "stokes_vector": np.round(Sv,3).tolist(),
            "intensity": np.round(Sv[0], 3),
            "degree_of_polarization": np.round(np.sqrt(Sv[1]**2 + Sv[2]**2 + Sv[3]**2) / Sv[0], 3),
            "ellipse_orientation": np.round(np.degrees(1 / 2 * np.arctan2(Sv[2], Sv[1])), 3),
            "ellipse_ellipticity": np.round(np.degrees(1 / 2 * np.arcsin(Sv[3] / Sv[0])), 3),
            "ellipse_axes": (np.round(np.sqrt((Sv[0] + np.sqrt(Sv[1]**2 + Sv[2]**2)) / 2), 2),
                            np.round(np.sqrt((Sv[0] - np.sqrt(Sv[1]**2 + Sv[2]**2)) / 2), 2)),
        }
        out_stokes_vector = stokes_vector
        states.append(input_info)
        num=1
        animation = True
        stokes_vecs =[stokes_vector]
        

        i = 1
        while f'matrix_type-{i}' in request.POST:
            matrix_type = request.POST.get(f'matrix_type-{i}')
            matrix_angle = np.radians(float(request.POST.get(f'matrix_angle-{i}', 0)))
            retarder_fast_axis_angle = np.radians(float(request.POST.get(f'retarder_fast_axis_angle-{i}', 0)))
            fast_axis_angle = np.radians(float(request.POST.get(f'fast_axis_angle-{i}', 0)))
            retardance = np.radians(float(request.POST.get(f'retardance-{i}', 0)))
            optical_density = float(request.POST.get(f'optical_density-{i}', 0))
            re_index_of_refraction = float(request.POST.get(f're_index_of_refraction-{i}', 1))
            im_index_of_refraction = float(request.POST.get(f'im_index_of_refraction-{i}', 0))

            index_of_refraction = complex(re_index_of_refraction, im_index_of_refraction) 
            incidence_angle = np.radians(float(request.POST.get(f'incidence_angle-{i}', 0)))

            mueller_matrix = get_mueller_matrix(matrix_type,matrix_angle,retarder_fast_axis_angle,fast_axis_angle,retardance,optical_density,index_of_refraction,incidence_angle)

            out_stokes_vector = mueller_matrix @ out_stokes_vector
            stokes_vecs.append(out_stokes_vector)

            
            filename =  f'media/polarization/{num}_mueller_animation.gif'
            fig, ax = plt.subplots(figsize=(8, 8))

            ani = matplotlib.animation.FuncAnimation(fig, animation_update,
                                        frames=np.linspace(0, -2 * np.pi),
                                        fargs=(stokes_to_jones(out_stokes_vector), ax))
            ani.save(filename, writer=PillowWriter(fps=30))
            plt.close()
        

            after_optical_element = f"after {matrix_type}"
            if matrix_type == 'Linear Polarizer':
                after_optical_element = f"after {matrix_type} at {request.POST.get(f'matrix_angle-{i}', 0)}°"
            elif matrix_type == 'Mirror':
                after_optical_element = f"after {matrix_type} "

            elif matrix_type == 'Attenuator':
                after_optical_element = f"after {matrix_type} with opticat density = {optical_density}"
            elif matrix_type == 'Quarter Wave Plate' or matrix_type == 'Half Wave Plate':
                after_optical_element = f"after {matrix_type} with fast axis at {request.POST.get(f'fast_axis_angle-{i}', 0)}°"
            elif matrix_type == 'Retarder':
                after_optical_element = f"after {matrix_type} with fast axis at {request.POST.get(f'retarder_fast_axis_angle-{i}', 0)}° & retardance = {request.POST.get(f'retardance-{i}', 0)}°"
            elif matrix_type == 'Fresnel Reflection' or matrix_type == 'Fresnel Transmission':
                after_optical_element = f"after {matrix_type} with incidence angle at {request.POST.get(f'incidence_angle-{i}', 0)}° & index of refraction = {index_of_refraction}"



            state_info = {
                "polarization": after_optical_element,
                "animation":f'{num}_mueller_animation.gif',
                "stokes_vector": np.round(out_stokes_vector,3).tolist(),
                "intensity": np.round(out_stokes_vector[..., 0],3),
                "degree_of_polarization":np.round(np.sqrt(out_stokes_vector[1]**2 + out_stokes_vector[2]**2 + out_stokes_vector[3]**2) / out_stokes_vector[0],3),
                "ellipse_orientation": np.round(np.degrees(1 / 2 * np.arctan2(out_stokes_vector[..., 2], out_stokes_vector[..., 1])),3),
                "ellipse_ellipticity": np.round(np.degrees(1 / 2 * np.arcsin(out_stokes_vector[..., 3] / out_stokes_vector[..., 0])),3),
                "ellipse_axes": ( np.round(np.sqrt((out_stokes_vector[..., 0] + np.sqrt(out_stokes_vector[..., 1]**2 + out_stokes_vector[..., 2]**2)) / 2),2),
                                    np.round(np.sqrt((out_stokes_vector[..., 0] - np.sqrt(out_stokes_vector[..., 1]**2 + out_stokes_vector[..., 2]**2)) / 2),2) )

                
                }
            
            states.append(state_info)

            num+=1
            i+=1

        fig = go.Figure()
        sphere_data = draw_empty_sphere()
        fig.add_trace(sphere_data['data'][0]) 
        annotations = sphere_data['layout']['scene']['annotations']

        scatter, ann = draw_stokes_poincare(stokes_vecs[0], label='input', color='red')
        fig.add_trace(scatter)
        annotations.extend(ann)

        for i in range(len(stokes_vecs) - 1):
            if i == len(stokes_vecs) - 2:
                scatter, ann = draw_stokes_poincare(stokes_vecs[i + 1], label='output', color='green')
                fig.add_trace(scatter)
                annotations.extend(ann)
            else:
                scatter, ann = draw_stokes_poincare(stokes_vecs[i + 1], label='None', color='blue')
                fig.add_trace(scatter)

            fig.add_trace(join_stokes_poincare(stokes_vecs[i], stokes_vecs[i + 1], color='blue', linestyle='solid'))

        fig.update_layout(
            scene=dict(
                xaxis_title='S₁',
                yaxis_title='S₂',
                zaxis_title='S₃',
                aspectmode='cube',
                
                annotations=sphere_data['layout']['scene']['annotations']
            ),
            margin=dict(l=0, r=0, b=0, t=0),
            height=800,
        )
        filename = f'media/polarization/interactive_poincare_sphere_plot.html'
        fig.write_html(filename)
        plot_html = fig.to_html(full_html=False)

            

        context =  {'animation': True,'plot_html': plot_html,'states':states,}

    
    else:
        context = {'animation': False}

    

    if request.headers.get("HX-Request") == "true":
        return render(request, "polar/partials/muellercalculus_results.html",context)
    else:
        return render(request, "polar/muellercalculus.html", context)
    


def pythoncode(request):

    return render(request, 'polar/pythoncode.html')

def t_par(m, theta_i, n_i):

    m2 = (m / n_i)**2
    c = np.cos(theta_i)
    s = np.sin(theta_i)
    d = np.sqrt(m2 - s * s, dtype=complex)  
    if m.imag == 0: 
        d = np.conjugate(d)
    m2 = (m / n_i)**2
    tp = 2 * c * (m / n_i) / (m2 * c + d)
    return np.real_if_close(tp)

def t_per(m, theta_i, n_i):

    m2 = (m / n_i)**2
    c = np.cos(theta_i)
    s = np.sin(theta_i)
    d = np.sqrt(m2 - s * s, dtype=complex)  
    if m.imag == 0: 
        d = np.conjugate(d)
  
    ts = 2 * d / (m / n_i) / (c + d)
    return np.real_if_close(ts)

def T_par(m, theta_i, n_i):

    m2 = (m / n_i)**2
    c = np.cos(theta_i)
    s = np.sin(theta_i)
    d = np.sqrt(m2 - s * s, dtype=complex) 
    if m.imag == 0: 
        d = np.conjugate(d)
    tp = 2 * c * (m / n_i) / ((m / n_i)**2 * c + d)
    return np.abs(d / c * np.abs(tp)**2)

def T_per(m, theta_i, n_i):
  
    m2 = (m / n_i)**2
    c = np.cos(theta_i)
    s = np.sin(theta_i)
    d = np.sqrt(m2 - s * s, dtype=complex) 
    if m.imag == 0: 
        d = np.conjugate(d)
    ts = 2 * c / (c + d)
    return np.abs(d / c * abs(ts)**2)
    
def T_unpolarized(m, theta_i, n_i):
  
    return (T_par(m, theta_i, n_i) + T_per(m, theta_i, n_i)) / 2
def r_par(m, theta_i, n_i):
 
    m2 = (m / n_i)**2
    c = np.cos(theta_i)
    s = np.sin(theta_i)
    d = np.sqrt(m2 - s * s, dtype=complex)
    m2 = (m / n_i)**2
    rp = (m2 * c - d) / (m2 * c + d)
    return np.real_if_close(rp)

def r_per(m, theta_i, n_i):
  
    m2 = (m / n_i)**2
    c = np.cos(theta_i)
    s = np.sin(theta_i)
    d = np.sqrt(m2 - s * s, dtype=complex)
    rs = (c - d) / (c + d)
    return np.real_if_close(rs)

def R_par(m, theta_i, n_i):
  
    return np.abs(r_par(m, theta_i, n_i))**2

def R_per(m, theta_i, n_i):
  
    return np.abs(r_per(m, theta_i, n_i))**2
    
def R_unpolarized(m, theta_i, n_i):
  
    return (R_par(m, theta_i, n_i) + R_per(m, theta_i, n_i)) / 2

def animation_update(frame, J, ax):
    
    ax.clear()
    h_amp, v_amp = np.abs(J)
    h_phi, v_phi = np.angle(J)
    the_max = max(h_amp, v_amp) * 1.1

    ax.plot([-the_max, the_max], [0, 0], 'g')
    ax.plot([0, 0], [-the_max, the_max], 'b')

    t = np.linspace(0, 2 * np.pi, 100)
    x = h_amp * np.cos(t - h_phi + frame)
    y = v_amp * np.cos(t - v_phi + frame)
    ax.plot(x, y, 'k')

    x = h_amp * np.cos(h_phi + frame)
    y = v_amp * np.cos(v_phi + frame)
    ax.plot(x, y, 'ro')
    ax.plot([x, x], [0, y], 'g--')
    ax.plot([0, x], [y, y], 'b--')
    ax.plot([0, x], [0, y], 'r')

    ax.set_xlim(-the_max, the_max)
    ax.set_ylim(-the_max, the_max)
    ax.set_aspect('equal')
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(0, 1, "y", ha="center")
    ax.text(1, 0, "x", va="center")

def draw_empty_sphere():
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))

    sphere = go.Surface(x=x, y=y, z=z, opacity=0.2, showscale=False)
    annotations = [
        dict(
            showarrow=False,
            x=1.15, y=0, z=0,
            text='horizontal 0°', 
            xanchor='center', yanchor='bottom',
            font=dict(size=12, color='black'),
        ),
        dict(
            showarrow=False,
            x=0, y=1.25, z=0,
            text='45°', 
            xanchor='center', yanchor='bottom',
            font=dict(size=12, color='black'),
        ),
        dict(
            showarrow=False,
            x=0, y=0, z=1.15,
            text='Right Circular', 
            xanchor='center', yanchor='bottom',
            font=dict(size=12, color='black'),
        ),
        dict(
            showarrow=False,
            x=0, y=0, z=-1.15,
            text='Left Circular', 
            xanchor='center', yanchor='bottom',
            font=dict(size=12, color='black'),
        ),
        dict(
            showarrow=False,
            x=-1.15, y=0, z=0,
            text='vertical 90°', 
            xanchor='center', yanchor='bottom',
            font=dict(size=12, color='black'),
        ),
    ]

    return {'data': [sphere], 'layout': {'scene': {'annotations': annotations}}}

def great_circle_points(ax, ay, az, bx, by, bz):
    delta = np.arccos(ax * bx + ay * by + az * bz)
    psi = np.linspace(0, delta)
    sinpsi = np.sin(psi)
    cospsi = np.cos(psi)
    sindelta = np.sin(delta)
    if sindelta == 0:
        sindelta = 1e-5
    elif abs(sindelta) < 1e-5:
        sindelta = 1e-5 * np.sign(sindelta)
    x = cospsi * ax + sinpsi * ((az**2 + ay**2) * bx - (az * bz + ay * by) * ax) / sindelta
    y = cospsi * ay + sinpsi * ((az**2 + ax**2) * by - (az * bz + ax * bx) * ay) / sindelta
    z = cospsi * az + sinpsi * ((ay**2 + ax**2) * bz - (ay * by + ax * bx) * az) / sindelta
    return x, y, z

def join_stokes_poincare(S1, S2, color='blue', lw=2, linestyle='dash'):
    SS1 = np.sqrt(S1[1]**2 + S1[2]**2 + S1[3]**2)
    SS2 = np.sqrt(S2[1]**2 + S2[2]**2 + S2[3]**2)
    x, y, z = great_circle_points(S1[1] / SS1, S1[2] / SS1, S1[3] / SS1, S2[1] / SS2, S2[2] / SS2, S2[3] / SS2) 
    line = go.Scatter3d(x=x, y=y, z=z, mode='lines', line=dict(color=color, width=lw, dash=linestyle),showlegend=False)
    return line

def draw_stokes_poincare(S, label, color):
    SS = np.sqrt(S[1]**2 + S[2]**2 + S[3]**2)
    x = S[1] / SS
    y = S[2] / SS
    z = S[3] / SS
    marker = dict(symbol='circle', size=6, color=color, line=dict(color='black', width=1))
    scatter = go.Scatter3d(x=[x], y=[y], z=[z], mode='markers', marker=marker, name=label, showlegend=False)
    annotations = []
    if label:
        annotations.append(dict(
            showarrow=True,
            x=x, y=y, z=z,
            text=label,
            xanchor='left',
            yanchor='middle',
            font=dict(size=12, color=color if label in ['input','output']  else 'black'),
            ax=20,
            ay=-20,
        ))
    return scatter, annotations