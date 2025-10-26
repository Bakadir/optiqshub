from django.shortcuts import render, redirect

# Create your views here.
from numpy import pi, cos, sin, arcsin, sqrt
import numpy as np
from django.http import HttpResponse, Http404
from .forms import LayerForm, InputForm,Angle,Wavelength,Thickness,LayerFormSet,LayerDataForm,GraphWavelength,GraphAngle,GraphThickness

from django.forms import formset_factory
import os
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import json
import plotly.utils
import pandas as pd
import numpy as np
from numpy import pi, cos, sin, arcsin
import yaml
import re
from django.http import JsonResponse
from io import BytesIO
import requests
import xlsxwriter
from functools import reduce
from scipy.interpolate import interp1d
from .models import LayerData, LEDSpectrumData, MultilayerFilmData
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.path import Path
import pandas as pd

from scipy.spatial import Delaunay
import matplotlib
matplotlib.use('Agg')

def parse_catalog_yaml(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)

def remove_html_tags(text):
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)

def process_catalog_content(content):
    items = []
    current_divider = None
    for item in content:
        if 'DIVIDER' in item:
            current_divider = {'label': item['DIVIDER'], 'books': []}
            items.append(current_divider)
        elif 'BOOK' in item:
            if current_divider:
                book_name = remove_html_tags(item['name'])
                current_divider['books'].append({'book': item['BOOK'], 'name': book_name,  'content': item['content']})
            else:
                book_name = remove_html_tags(item['name'])
                items.append({'label': '', 'books': [{'book': item['BOOK'], 'name': book_name,  'content': item['content']}]})
    return items

import os
import yaml

from django.http import JsonResponse
import os

def get_categories_and_materials(base_path):
    categories = {}
    try:
        for category in os.listdir(base_path):
            category_path = os.path.join(base_path, category)
            if os.path.isdir(category_path):
                materials = {}
                for material in os.listdir(category_path):
                    material_path = os.path.join(category_path, material)
                    if os.path.isdir(material_path):
                        nk_files = []
                        nk_path = os.path.join(material_path, "nk")
                        if os.path.exists(nk_path):
                            nk_files = [
                                f for f in os.listdir(nk_path) if f.endswith(".yml")
                            ]
                        materials[material] = nk_files
                categories[category] = materials
    except Exception as e:
        print(f"An error occurred: {e}")
    return categories
from django.http import JsonResponse
import os

""" def update_pages(request):
    material = request.GET.get('material')
    base_path = "static/thinfilms"
    nk_files = []

    if material:
        # Find the nk files for the selected material
        for category in os.listdir(base_path):
            category_path = os.path.join(base_path, category)
            if os.path.isdir(category_path):
                material_path = os.path.join(category_path, material)
                if os.path.isdir(material_path):
                    nk_path = os.path.join(material_path, "nk")
                    if os.path.exists(nk_path):
                        nk_files = [
                            f for f in os.listdir(nk_path) if f.endswith(".yml")
                        ]

    return JsonResponse({'nk_files': nk_files})

 """
def update_pages(request):
    book_id = request.GET.get('book')
    file_path = f'static/thinfilms/catalog.yml'
    catalog = parse_catalog_yaml(file_path)
    catalog_content = process_catalog_content(catalog)

    pages = []
    current_divider = None
    for group in catalog_content:
        for book in group['books']:
            if book['book'] == book_id:
                for item in book['content']:
                    if 'DIVIDER' in item:
                        current_divider = item['DIVIDER']
                    elif 'PAGE' in item:
                        pages.append({'label': current_divider, 'page': item['PAGE'], 'name': item['name']})

    # Group pages by label
    grouped_pages = {}
    for page in pages:
        label = page['label']
        if label not in grouped_pages:
            grouped_pages[label] = []
        grouped_pages[label].append({'page': page['page'], 'name': page['name']})

    return JsonResponse({'pages': grouped_pages})
def home(request):
    base_path = "static/thinfilms"
    categories_and_materials = get_categories_and_materials(base_path)

    file_path = f'static/thinfilms/catalog-nk.yml'
    catalog = parse_catalog_yaml(file_path)
    catalog_content = process_catalog_content(catalog)

    if request.method == 'POST' and 'calculate' in request.POST:
        try:
            number_of_layers = int(request.POST.get('number_of_layers', 1))
        except ValueError:
            number_of_layers = 1

        layers_data = []

        # Loop through each dynamic layer using 1-based indexing
        for i in range(1, number_of_layers + 1):  # starts at 1
            book = request.POST.get(f'book-{i}', '')
            page = request.POST.get(f'page-{i}', '')
            thickness = request.POST.get(f'thickness-{i}', '0')

            layer_data = {
                'material': book,
                'nk_file': page,
                'thickness': thickness
            }
            layers_data.append(layer_data)

        print("Layers Data:", layers_data)
        request.session['layers_data'] = layers_data

        return redirect('thinfilms:result')

    # For GET requests, just render the page
    return render(request, 'thinfilms/home.html', {
        'categories_and_materials': categories_and_materials,
        'catalog': catalog_content,
    })

def homeee(request):
    base_path = "static/thinfilms"
    categories_and_materials = get_categories_and_materials(base_path)

    file_path = f'static/thinfilms/catalog.yml'
    catalog = parse_catalog_yaml(file_path)
    catalog_content = process_catalog_content(catalog)


    if request.method == 'POST':
        input_form = InputForm(request.POST)
        MatrixFormSet = formset_factory(LayerForm, extra=1)
        matrix_formset = MatrixFormSet(request.POST)

        if input_form.is_valid() and matrix_formset.is_valid():
            number_of_layers = input_form.cleaned_data['number_of_layers']
            MatrixFormSet = formset_factory(LayerForm, extra=number_of_layers)
            matrix_formset = MatrixFormSet()

            if 'calculate' in request.POST:

                layers_data = []
                for i in range(number_of_layers):
                    book = request.POST.get(f"book-{i}")
                    page = request.POST.get(f"page-{i}")
                    thickness = request.POST.get(f"form-{i}-thickness")
                    layer_data = [book,page,thickness]
                    
                    layers_data.append(layer_data)

                request.session['layers_data'] = layers_data
                
                # Redirect to the result view
                return redirect('thinfilms:result')
            
    else:
        input_form = InputForm()
        MatrixFormSet = formset_factory(LayerForm, extra=1)
        matrix_formset = MatrixFormSet()

    return render(request, 'thinfilms/home.html', {
        'input_form': input_form,
        'matrix_formset': matrix_formset,
        'categories_and_materials': categories_and_materials,
        'catalog': catalog_content,
    })


def result(request):
    layers_data = request.session.get('layers_data', [])
    
    if not layers_data:
        return HttpResponse("No layers data found.", status=404)
    
    base_github_url = 'https://github.com/polyanskiy/refractiveindex.info-database/tree/master/database/data/main'
    refractiveindex_url = 'https://refractiveindex.info/?shelf=main'
    
    detailed_layers_data = []
    
    all_wavelengths = []
    all_n_values = []
    all_k_values = []
    wavelength_nk_map = []
    multilayer_name = []
    for layer in layers_data:

        book = layer.get('material', '')
        page = layer.get('nk_file', '')
        thickness = layer.get('thickness', '0')

        github_link = f"{base_github_url}/{book}/nk/{page}.yml"
        refractiveindex_link = f"{refractiveindex_url}&book={book}&page={page}"


        

        try_url = f"https://github.com/polyanskiy/refractiveindex.info-database/blob/master/database/data/main/{book}/nk/{page}.yml"
        raw_url = f"https://raw.githubusercontent.com/polyanskiy/refractiveindex.info-database/master/database/data/main/{book}/nk/{page}.yml"


        response = requests.get(raw_url)

        if response.status_code != 200:
            error_message = (
                f"Error fetching the data from GitHub. "
                f"Attempted URL: <a href='{try_url}' target='_blank'>{try_url}</a>"
            )
            return HttpResponse(error_message, status=404)
                
        yaml_data = yaml.safe_load(response.text)
        
        # Extract the relevant data section (type: tabulated nk)
        data_section = None
        for data in yaml_data['DATA']:
            if data['type'] == 'tabulated nk':
                data_section = data['data']
                break
        
        if not data_section:
            error_message = (
                "No tabulated nk data found in the YAML file."
                f"Attempted URL: {try_url} "
            )
            return HttpResponse(error_message, status=404)
        
        # Split the data into lines and then into columns
        data_lines = data_section.strip().split("\n")
        parsed_data = [line.split() for line in data_lines]
        wavelengths = []
        n_values = []
        k_values = []
        rounded_wavelengths = []
        wavelength_to_nk = {}
        
        for parts in parsed_data:
            if len(parts) >= 3:
                try:
                    wavelength = float(parts[0])
                    rounded_wavelength = round(wavelength*1e3, 3)
                    n_value = float(parts[1])
                    k_value = float(parts[2])
                    wavelengths.append(wavelength*1e3)
                    rounded_wavelengths.append(rounded_wavelength)
                    n_values.append(n_value)
                    k_values.append(k_value)
                    wavelength_to_nk[rounded_wavelength] = (n_value, k_value)
                except ValueError:
                    continue
        
        multilayer_name.append(f"{book}({thickness}nm)")
        min_wavelength = min(wavelengths) if wavelengths else None
        max_wavelength = max(wavelengths) if wavelengths else None
        num_values = len(wavelengths)
        
        all_wavelengths.append(set(rounded_wavelengths))
        all_n_values.append(n_values)
        all_k_values.append(k_values)
        wavelength_nk_map.append(wavelength_to_nk)
        
        detailed_layers_data.append({
            'book': book,
            'page': page,
            'thickness': thickness,
            'github_link': github_link,
            'refractiveindex_link': refractiveindex_link,
            'parsed_data': list(zip(wavelengths, n_values, k_values)),
            'min_wavelength': min_wavelength,
            'max_wavelength': max_wavelength,
            'num_values': num_values
        })
    
    # Find common wavelengths across all layers
    common_wavelengths = sorted(reduce(set.intersection, all_wavelengths)) if all_wavelengths else []
    common_wavelengths_all =common_wavelengths 
    final_data = []
    for wavelength in common_wavelengths:
        nk_row = []
        for i in range(len(layers_data)):
            n, k = wavelength_nk_map[i].get(wavelength, (None, None))
            nk_row.append((n, k))
        final_data.append((wavelength, nk_row))
    layers = []
    for i, layer in enumerate(detailed_layers_data):
        thickness_nm = float(layer['thickness'])  # Ensure thickness is a float
        nk_data = [complex(wavelength_nk_map[i][wavelength][0], wavelength_nk_map[i][wavelength][1]) for wavelength in common_wavelengths]
        layers.append([thickness_nm, common_wavelengths, nk_data])
    multilayer_name_str = ''.join(multilayer_name)

    N =len(common_wavelengths)
    books = [layer['book'] for layer in detailed_layers_data]
    
    Rp_list = []
    Tp_list = []
    Ap_list = []

    Rs_list = []
    Ts_list = []
    As_list = []

    R_list = []
    T_list = []
    A_list = []

    if request.method == 'POST':
        form = Angle(request.POST, wavelengths=common_wavelengths)
        form1 = Wavelength(request.POST, wavelengths=common_wavelengths)
        form3 = Thickness(request.POST, wavelengths=common_wavelengths, layers=detailed_layers_data)
        if form.is_valid() and form1.is_valid() and form3.is_valid() :
            incidence_angle = form.cleaned_data.get('incidence_angle', 0)
            min_wavelength = form.cleaned_data.get('min_wavelength')
            max_wavelength = form.cleaned_data.get('max_wavelength')

            chosen_wavelength = form1.cleaned_data.get('chosen_wavelength')
            min_angle = float(form1.cleaned_data.get('min_angle'))
            max_angle = float(form1.cleaned_data.get('max_angle'))
            step_angle = int(form1.cleaned_data.get('step_angle'))


            chosen_layer_thick = int(form3.cleaned_data.get('chosen_layer'))
            incidence_angle_thick = form3.cleaned_data.get('incidence_angle_thick', 0)
            chosen_wavelength_thick = float(form3.cleaned_data.get('chosen_wavelength_thick'))
            min_thick = form3.cleaned_data.get('min_thick')
            max_thick = form3.cleaned_data.get('max_thick')
            step_thick = form3.cleaned_data.get('step_thick')

            # Filter common_wavelengths based on min and max values
            common_wavelengths = [wl for wl in common_wavelengths 
                                    if float(min_wavelength) <= wl <= float(max_wavelength)]
            

    else:
        form = Angle(wavelengths=common_wavelengths)
        form1 = Wavelength(wavelengths=common_wavelengths)
        form3 = Thickness(wavelengths=common_wavelengths, layers=detailed_layers_data)
        incidence_angle = 0
        chosen_wavelength = min_wavelength
        min_angle = 0
        max_angle = 90
        step_angle = 1000
        
        chosen_layer_thick = len(layers)-1 
        chosen_wavelength_thick = min_wavelength
        incidence_angle_thick = 0
        min_thick = 0
        max_thick = 100
        step_thick = 1000
    for wl in common_wavelengths:
        Rp, Tp, Ap = calculate_RT(layers, wl, "p", np.radians(incidence_angle)) 
        Rs, Ts, As = calculate_RT(layers, wl, "s", np.radians(incidence_angle))
        Rp_list.append(Rp)
        Tp_list.append(Tp)
        Ap_list.append(Ap)

        Rs_list.append(Rs)
        Ts_list.append(Ts)
        As_list.append(As)

        R_list.append((Rp+Rs)/2)
        T_list.append((Tp+Ts)/2)
        A_list.append((Ap+As)/2)
    
    # Convert lists to numpy arrays for further processing
    Rp_array = np.array(Rp_list)
    Tp_array = np.array(Tp_list)
    Ap_array = np.array(Ap_list)
    RTA_p = zip(common_wavelengths, Rp_array, Tp_array, Ap_array)
    
    Rs_array = np.array(Rs_list)
    Ts_array = np.array(Ts_list)
    As_array = np.array(As_list)
    RTA_s = zip(common_wavelengths, Rs_array, Ts_array, As_array)
    
    R_array = np.array(R_list)
    T_array = np.array(T_list)
    A_array = np.array(A_list)
    RTA_unpolarized = zip(common_wavelengths, R_array, T_array, A_array)


    # Generate plots for each polarization
    fig_p = go.Figure()
    fig_p.add_trace(go.Scatter(x=common_wavelengths, y=Rp_array, mode='lines+markers', name='Reflectance'))
    fig_p.add_trace(go.Scatter(x=common_wavelengths, y=Tp_array, mode='lines+markers', name='Transmittance'))
    fig_p.add_trace(go.Scatter(x=common_wavelengths, y=Ap_array, mode='lines+markers', name='Absorbance'))
    fig_p.update_layout(
        title=f'p-polarized at {incidence_angle}° on {multilayer_name_str} ',
        xaxis_title='Wavelength (nm)',
        yaxis_title='%',
    )
    graph_p_json = json.dumps(fig_p, cls=plotly.utils.PlotlyJSONEncoder)
    
    fig_s = go.Figure()
    fig_s.add_trace(go.Scatter(x=common_wavelengths, y=Rs_array, mode='lines+markers', name='Reflectance'))
    fig_s.add_trace(go.Scatter(x=common_wavelengths, y=Ts_array, mode='lines+markers', name='Transmittance'))
    fig_s.add_trace(go.Scatter(x=common_wavelengths, y=As_array, mode='lines+markers', name='Absorbance'))
    fig_s.update_layout(
        title=f's-polarized at {incidence_angle}° on {multilayer_name_str}',
        xaxis_title='Wavelength (nm)',
        yaxis_title='%',
        height=400 
    )
    graph_s_json = json.dumps(fig_s, cls=plotly.utils.PlotlyJSONEncoder)
    
    fig_unpolarized = go.Figure()
    fig_unpolarized.add_trace(go.Scatter(x=common_wavelengths, y=R_array, mode='lines+markers', name='Reflectance'))
    fig_unpolarized.add_trace(go.Scatter(x=common_wavelengths, y=T_array, mode='lines+markers', name='Transmittance'))
    fig_unpolarized.add_trace(go.Scatter(x=common_wavelengths, y=A_array, mode='lines+markers', name='Absorbance'))
    fig_unpolarized.update_layout(
        title=f'unpolarized at {incidence_angle}° on {multilayer_name_str}',
        xaxis_title='Wavelength (nm)',
        yaxis_title='%',
        height=400 
    )
    graph_unpolarized_json = json.dumps(fig_unpolarized, cls=plotly.utils.PlotlyJSONEncoder)
    thickness_range = np.linspace(min_thick, max_thick, step_thick)  
    Rp_thick = []
    Tp_thick = []
    Ap_thick = []

    Rs_thick = []
    Ts_thick = []
    As_thick= []

    R_thick = []
    T_thick = []
    A_thick = []
    multilayer_name_thick = multilayer_name
    book = detailed_layers_data[chosen_layer_thick]['book']
    multilayer_name_thick[chosen_layer_thick] = f"{book}(thickness nm)"
    multilayer_name_thick =  ''.join(multilayer_name_thick)
    layers_thick = layers
    for thick in thickness_range:
        layers_thick[chosen_layer_thick][0] = thick 
        Rp, Tp, Ap = calculate_RT(layers_thick, chosen_wavelength_thick, "p", np.radians(incidence_angle_thick)) 
        Rs, Ts, As = calculate_RT(layers_thick, chosen_wavelength_thick, "s", np.radians(incidence_angle_thick))

        Rp_thick.append(Rp)
        Tp_thick.append(Tp)
        Ap_thick.append(Ap)

        Rs_thick.append(Rs)
        Ts_thick.append(Ts)
        As_thick.append(As)

        R_thick.append((Rp + Rs) / 2)
        T_thick.append((Tp + Ts) / 2)
        A_thick.append((Ap + As) / 2)
    
    angles = np.linspace(min_angle, max_angle, step_angle)  # Example angles from 0 to 90 degrees
    Rp_angle = []
    Tp_angle = []
    Ap_angle = []

    Rs_angle = []
    Ts_angle = []
    As_angle = []

    R_angle = []
    T_angle = []
    A_angle = []
    
    for angle in angles:

        Rp, Tp, Ap = calculate_RT(layers, float(chosen_wavelength), "p", np.radians(angle)) # calculate_RT_p(layers, wl, 0)
        Rs, Ts, As = calculate_RT(layers, float(chosen_wavelength), "s", np.radians(angle))
        Rp_angle.append(Rp)
        Tp_angle.append(Tp)
        Ap_angle.append(Ap)

        Rs_angle.append(Rs)
        Ts_angle.append(Ts)
        As_angle.append(As)

        R_angle.append((Rp+Rs)/2)
        T_angle.append((Tp+Ts)/2)
        A_angle.append((Ap+As)/2)

    RTA_thick_p = zip(thickness_range, Rp_thick, Tp_thick, Ap_thick)
    RTA_thick_s = zip(thickness_range, Rs_thick, Ts_thick, As_thick)
    RTA_thick_unpolarized = zip(thickness_range, R_thick, T_thick, A_thick)

    # Plot the data
    fig_thick_unpolarized = go.Figure()
    fig_thick_unpolarized.add_trace(go.Scatter(x=thickness_range, y=np.array(R_thick), mode='lines+markers', name='Reflectance'))
    fig_thick_unpolarized.add_trace(go.Scatter(x=thickness_range, y=np.array(T_thick), mode='lines+markers', name='Transmittance'))
    fig_thick_unpolarized.add_trace(go.Scatter(x=thickness_range, y=np.array(A_thick), mode='lines+markers', name='Absorbance'))
    fig_thick_unpolarized.update_layout(
        title=f'Unpolarized at {chosen_wavelength_thick} nm for {multilayer_name_thick}',
        xaxis_title='Thickness (nm)',
        yaxis_title='%'
    )
    graph_thick_unpolarized_json = json.dumps(fig_thick_unpolarized, cls=plotly.utils.PlotlyJSONEncoder)

    fig_thick_s = go.Figure()
    fig_thick_s.add_trace(go.Scatter(x=thickness_range, y=np.array(Rs_thick), mode='lines+markers', name='Reflectance'))
    fig_thick_s.add_trace(go.Scatter(x=thickness_range, y=np.array(Ts_thick), mode='lines+markers', name='Transmittance'))
    fig_thick_s.add_trace(go.Scatter(x=thickness_range, y=np.array(As_thick), mode='lines+markers', name='Absorbance'))
    fig_thick_s.update_layout(
        title=f'S-polarized at {chosen_wavelength_thick} nm for {multilayer_name_thick}',
        xaxis_title='Thickness (nm)',
        yaxis_title='%'
    )
    graph_thick_s_json = json.dumps(fig_thick_s, cls=plotly.utils.PlotlyJSONEncoder)

    fig_thick_p = go.Figure()
    fig_thick_p.add_trace(go.Scatter(x=thickness_range, y=np.array(Rp_thick), mode='lines+markers', name='Reflectance'))
    fig_thick_p.add_trace(go.Scatter(x=thickness_range, y=np.array(Tp_thick), mode='lines+markers', name='Transmittance'))
    fig_thick_p.add_trace(go.Scatter(x=thickness_range, y=np.array(Ap_thick), mode='lines+markers', name='Absorbance'))
    fig_thick_p.update_layout(
        title=f'P-polarized at {chosen_wavelength_thick} nm for {multilayer_name_thick}',
        xaxis_title='Thickness (nm)',
        yaxis_title='%'
    )
    graph_thick_p_json = json.dumps(fig_thick_p, cls=plotly.utils.PlotlyJSONEncoder)


    RTA_angle_p = zip(angles, Rp_angle, Tp_angle, Ap_angle)
    RTA_angle_s = zip(angles, Rs_angle, Ts_angle, As_angle)
    RTA_angle_unpolarized = zip(angles, R_angle, T_angle, A_angle)
    
    # Plot the data
    fig_angle_unpolarized = go.Figure()
    fig_angle_unpolarized.add_trace(go.Scatter(x=angles, y=np.array(R_angle), mode='lines+markers', name='Reflectance'))
    fig_angle_unpolarized.add_trace(go.Scatter(x=angles, y=np.array(T_angle), mode='lines+markers', name='Transmittance'))
    fig_angle_unpolarized.add_trace(go.Scatter(x=angles, y=np.array(A_angle), mode='lines+markers', name='Absorbance'))
    fig_angle_unpolarized.update_layout(
        title=f'Unpolarized at {chosen_wavelength} nm for {multilayer_name_str}',
        xaxis_title='Angle (degrees)',
        yaxis_title='%'
    )
    graph_angle_unpolarized_json = json.dumps(fig_angle_unpolarized, cls=plotly.utils.PlotlyJSONEncoder)

    fig_angle_s = go.Figure()
    fig_angle_s.add_trace(go.Scatter(x=angles, y=np.array(Rs_angle), mode='lines+markers', name='Reflectance'))
    fig_angle_s.add_trace(go.Scatter(x=angles, y=np.array(Ts_angle), mode='lines+markers', name='Transmittance'))
    fig_angle_s.add_trace(go.Scatter(x=angles, y=np.array(As_angle), mode='lines+markers', name='Absorbance'))
    fig_angle_s.update_layout(
        title=f'S-polarized at {chosen_wavelength} nm for {multilayer_name_str}',
        xaxis_title='Angle (degrees)',
        yaxis_title='%'
    )
    graph_angle_s_json = json.dumps(fig_angle_s, cls=plotly.utils.PlotlyJSONEncoder)

    brewster_index = np.argmin(Rp_angle)
    brewster_angle = angles[brewster_index]
    fig_angle_p = go.Figure()
    fig_angle_p.add_trace(go.Scatter(x=angles, y=np.array(Rp_angle), mode='lines+markers', name='Reflectance'))
    fig_angle_p.add_trace(go.Scatter(x=angles, y=np.array(Tp_angle), mode='lines+markers', name='Transmittance'))
    fig_angle_p.add_trace(go.Scatter(x=angles, y=np.array(Ap_angle), mode='lines+markers', name='Absorbance'))
    fig_angle_p.update_layout(
        title=f'P-polarized at {chosen_wavelength} nm for {multilayer_name_str}',
        xaxis_title='Angle (degrees)',
        yaxis_title='%',
        annotations=[dict(x=brewster_angle, y=Rp_angle[brewster_index], 
                      xref='x', yref='y', 
                      text=f'Brewster Angle: {brewster_angle:.2f}°', 
                      showarrow=True, arrowhead=2)]
    )
    graph_angle_p_json = json.dumps(fig_angle_p, cls=plotly.utils.PlotlyJSONEncoder)
    # Save data to session for later use in download
    request.session['final_data'] = final_data
    request.session['books'] = books

    request.session['common_wavelengths'] = common_wavelengths

    request.session['Rp_list'] = Rp_list
    request.session['Tp_list'] = Tp_list
    request.session['Ap_list'] = Ap_list

    request.session['Rs_list'] = Rs_list
    request.session['Ts_list'] = Ts_list
    request.session['As_list'] = As_list

    request.session['R_list'] = R_list
    request.session['T_list'] = T_list
    request.session['A_list'] = A_list
   
    request.session['angles'] = angles.tolist()
    request.session['Rp_angle'] = Rp_angle
    request.session['Tp_angle'] = Tp_angle
    request.session['Ap_angle'] = Ap_angle

    request.session['Rs_angle'] = Rs_angle
    request.session['Ts_angle'] = Ts_angle
    request.session['As_angle'] = As_angle

    request.session['R_angle'] = R_angle
    request.session['T_angle'] = T_angle
    request.session['A_angle'] = A_angle

    request.session['thickness_range'] = thickness_range.tolist()
    request.session['Rp_thick'] = Rp_thick
    request.session['Tp_thick'] = Tp_thick
    request.session['Ap_thick'] = Ap_thick

    request.session['Rs_thick'] = Rs_thick
    request.session['Ts_thick'] = Ts_thick
    request.session['As_thick'] = As_thick

    request.session['R_thick'] = R_thick
    request.session['T_thick'] = T_thick
    request.session['A_thick'] = A_thick

    context = {
        'detailed_layers_data': detailed_layers_data,
        'multilayer_name': multilayer_name_str,
        'common_wavelengths': common_wavelengths,
        'final_data': final_data,
        'number': N,
        'books': books,
        'RTA_p': RTA_p,
        'RTA_s': RTA_s,
        'RTA_unpolarized': RTA_unpolarized,
        'RTA_angle_p': RTA_angle_p,
        'RTA_angle_s': RTA_angle_s,
        'RTA_angle_unpolarized': RTA_angle_unpolarized,
        'graph_p_json': graph_p_json,
        'graph_s_json': graph_s_json,
        'graph_unpolarized_json': graph_unpolarized_json,
        'form':form,
        'form1':form1,
        'form3':form3,
        'graph_angle_unpolarized_json':graph_angle_unpolarized_json,
        'graph_angle_s_json':graph_angle_s_json,
        'graph_angle_p_json':graph_angle_p_json,
        'incidence_angle':incidence_angle,
        'common_wavelengths_all':common_wavelengths_all,
        'graph_thick_unpolarized_json': graph_thick_unpolarized_json,
        'graph_thick_s_json': graph_thick_s_json,
        'graph_thick_p_json': graph_thick_p_json,
        'RTA_thick_p': RTA_thick_p,
        'RTA_thick_s': RTA_thick_s,
        'RTA_thick_unpolarized': RTA_thick_unpolarized,
        'min_angle':min(angles),
        'max_angle':max(angles),
        'angles':angles,
        'wavelength_angle': chosen_wavelength,
        'multilayer_name_thick':multilayer_name_thick,

    }
    return render(request, 'thinfilms/result.html', context)

def download_excel_common(request):
    final_data = request.session.get('final_data', [])
    books = request.session.get('books', [])
    
    columns = ["Wavelength (nm)"]
    for book in books:
        columns.extend([f"{book} n", f"{book} k"])

    data = []
    for wavelength, nk_row in final_data:
        row = [wavelength]
        for n, k in nk_row:
            row.extend([n, k])
        data.append(row)

    df = pd.DataFrame(data, columns=columns)

    # Create a BytesIO buffer to hold the Excel file
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Common Data')

    # Rewind the buffer
    buffer.seek(0)

    # Return the Excel file as a response
    response = HttpResponse(buffer, content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    response['Content-Disposition'] = 'attachment; filename=common_data.xlsx'

    return response
def download_excel(request, book, page):
    # Construct the GitHub URL
    github_url = f"https://raw.githubusercontent.com/polyanskiy/refractiveindex.info-database/master/database/data-nk/main/{book}/{page}.yml"
    
    # Fetch the YAML data from GitHub
    response = requests.get(github_url)
    if response.status_code != 200:
        return HttpResponse("Error fetching the data from GitHub.", status=404)
    
    yaml_data = yaml.safe_load(response.text)
    
    # Extract the relevant data section (type: tabulated nk)
    data_section = None
    for data in yaml_data['DATA']:
        if data['type'] == 'tabulated nk':
            data_section = data['data']
            break
    
    if not data_section:
        return HttpResponse("No tabulated nk data found in the YAML file.", status=404)
    
    # Split the data into lines and then into columns
    data_lines = data_section.strip().split("\n")
    parsed_data = [line.split() for line in data_lines]
    
    # Convert the parsed data to a DataFrame
    df = pd.DataFrame(parsed_data, columns=['wavelength (um)', 'n', 'k'])
    
    # Convert the columns to the appropriate data types
    df = df.astype({'wavelength (um)': float, 'n': float, 'k': float})

    df['wavelength (um)'] = df['wavelength (um)'] * 1e3
    
    # Rename the wavelength column to 'wavelength (nm)'
    df.rename(columns={'wavelength (um)': 'wavelength (nm)'}, inplace=True)
    df.rename(columns={'n': f'{book}_n'}, inplace=True)
    df.rename(columns={'k': f'{book}_k'}, inplace=True)


    # Convert DataFrame to Excel
    excel_buffer = BytesIO()
    with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False)
    
    # Prepare the response with the Excel file
    response = HttpResponse(
        excel_buffer.getvalue(),
        content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )
    response['Content-Disposition'] = f'attachment; filename={book}_{page}.xlsx'
    
    return response

def download_excel_RTA_p(request):
    common_wavelengths = request.session.get('common_wavelengths', [])
    R_list = request.session.get('Rp_list', [])
    T_list = request.session.get('Tp_list', [])
    A_list = request.session.get('Ap_list', [])

    data = {
        'Wavelength (nm)': common_wavelengths,
        'Reflectance (R)': R_list,
        'Transmittance (T)': T_list,
        'Absorbance (A)': A_list
    }

    df = pd.DataFrame(data)

    # Create a BytesIO buffer to hold the Excel file
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='RTA Data p-polarized')

    # Rewind the buffer
    buffer.seek(0)

    # Return the Excel file as a response
    response = HttpResponse(buffer, content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    response['Content-Disposition'] = 'attachment; filename=RTA_data_p_polarized.xlsx'

    return response

def download_excel_RTA_s(request):
    common_wavelengths = request.session.get('common_wavelengths', [])
    R_list = request.session.get('Rs_list', [])
    T_list = request.session.get('Ts_list', [])
    A_list = request.session.get('As_list', [])

    data = {
        'Wavelength (nm)': common_wavelengths,
        'Reflectance (R)': R_list,
        'Transmittance (T)': T_list,
        'Absorbance (A)': A_list
    }

    df = pd.DataFrame(data)

    # Create a BytesIO buffer to hold the Excel file
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='RTA Data s-polarized')

    # Rewind the buffer
    buffer.seek(0)

    # Return the Excel file as a response
    response = HttpResponse(buffer, content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    response['Content-Disposition'] = 'attachment; filename=RTA_data_s_polarized.xlsx'

    return response

def download_excel_RTA_unpolarized(request):
    common_wavelengths = request.session.get('common_wavelengths', [])
    R_list = request.session.get('R_list', [])
    T_list = request.session.get('T_list', [])
    A_list = request.session.get('A_list', [])

    data = {
        'Wavelength (nm)': common_wavelengths,
        'Reflectance (R)': R_list,
        'Transmittance (T)': T_list,
        'Absorbance (A)': A_list
    }

    df = pd.DataFrame(data)

    # Create a BytesIO buffer to hold the Excel file
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='RTA Data unpolarized')

    # Rewind the buffer
    buffer.seek(0)

    # Return the Excel file as a response
    response = HttpResponse(buffer, content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    response['Content-Disposition'] = 'attachment; filename=RTA_data_unpolarized.xlsx'

    return response

def calculate_RT(layers, wl_um, polarization_type, incidence_angle):
    eps0 = 8.85e-12
    mu0 = 4e-7 * np.pi
    Yfs = np.sqrt(eps0 / mu0)

    n0 = 1 

    wl_data = np.array(layers[-1][1])
    nk_data = np.array(layers[-1][2])
    ns = np.interp(wl_um, wl_data, nk_data)


    #last_layer_nk = layers[-1][2]  
    #ns_complex = [nk for wl, nk in zip(layers[-1][1], last_layer_nk) if wl == wl_um]

    #ns = ns_complex[0]
    #ns_tuple = ns_complex[0]  # This is expected to be a tuple (real, imag)
    #ns = complex(*ns_tuple)
    a0 = incidence_angle
    pol = polarization_type

    asub = np.arcsin(n0 * np.sin(a0) / ns)
    Y0 = n0 * Yfs
    Ys = ns * Yfs

    if pol == "p":
        N0 = Y0 / np.cos(a0)
        Ns = Ys / np.cos(asub)
    elif pol == "s":
        N0 = Y0 * np.cos(a0)
        Ns = Ys * np.cos(asub)
    
    M = np.eye(2, dtype=complex)

    for layer in layers:
        thickness_um, wl_data, nk_data = layer
        nk = np.interp(wl_um, wl_data, nk_data)  
        n_real = nk.real
        k_real = nk.imag

        a = np.arcsin(n0 * np.sin(a0) / n_real)

        delta_r = (2 * np.pi / wl_um) * thickness_um * np.sqrt((n_real**2 - k_real**2) - n0**2 * np.sin(a0)**2 - 2j * n_real * k_real)


        eta_rs = Yfs * np.sqrt((n_real**2 - k_real**2) - n0**2 * np.sin(a0)**2 - 2j * n_real * k_real)

        if pol == "p":
            eta_rp = Yfs**2 * (n_real - 1j * k_real)**2 / eta_rs
        elif pol == "s":
            eta_rp = eta_rs

        M00 = np.cos(delta_r)
        M01 = 1j * np.sin(delta_r) / eta_rp
        M10 = 1j * np.sin(delta_r) * eta_rp
        M11 = M00
        M_layer = np.array([
            [M00, M01],
            [M10, M11]
        ])

        M = np.dot(M, M_layer)

    B = M[0,0] + Ns * M[0,1]
    C = M[1,0] + Ns * M[1,1]

    r = (N0 * B - C) / (N0 * B + C)
    R = np.abs(r)**2
    T = 4 * N0 * Ns.real / np.abs(N0 * B + C)**2
    A = 1 - R - T

    return R, T, A



def download_excel_RTA_angle_p(request):
    angles = request.session.get('angles', [])
    R_list = request.session.get('Rp_angle', [])
    T_list = request.session.get('Tp_angle', [])
    A_list = request.session.get('Ap_angle', [])

    data = {
        'Angle in degree': angles,
        'Reflectance (R)': R_list,
        'Transmittance (T)': T_list,
        'Absorbance (A)': A_list
    }

    df = pd.DataFrame(data)

    # Create a BytesIO buffer to hold the Excel file
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='RTA Data p-polarized')

    # Rewind the buffer
    buffer.seek(0)

    # Return the Excel file as a response
    response = HttpResponse(buffer, content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    response['Content-Disposition'] = 'attachment; filename=RTA_data_angle_p_polarized.xlsx'

    return response

def download_excel_RTA_angle_s(request):
    angles = request.session.get('angles', [])
    R_list = request.session.get('Rs_angle', [])
    T_list = request.session.get('Ts_angle', [])
    A_list = request.session.get('As_angle', [])

    data = {
        'Angle in degree': angles,
        'Reflectance (R)': R_list,
        'Transmittance (T)': T_list,
        'Absorbance (A)': A_list
    }

    df = pd.DataFrame(data)

    # Create a BytesIO buffer to hold the Excel file
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='RTA Data s-polarized')

    # Rewind the buffer
    buffer.seek(0)

    # Return the Excel file as a response
    response = HttpResponse(buffer, content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    response['Content-Disposition'] = 'attachment; filename=RTA_data_angle_s_polarized.xlsx'

    return response

def download_excel_RTA_angle_unpolarized(request):
    angles = request.session.get('angles', [])
    R_list = request.session.get('R_angle', [])
    T_list = request.session.get('T_angle', [])
    A_list = request.session.get('A_angle', [])

    data = {
        'Angle in degree': angles,
        'Reflectance (R)': R_list,
        'Transmittance (T)': T_list,
        'Absorbance (A)': A_list
    }

    df = pd.DataFrame(data)

    # Create a BytesIO buffer to hold the Excel file
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='RTA Data unpolarized-polarized')

    # Rewind the buffer
    buffer.seek(0)

    # Return the Excel file as a response
    response = HttpResponse(buffer, content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    response['Content-Disposition'] = 'attachment; filename=RTA_data_angle_unpolarized.xlsx'

    return response

def download_excel_RTA_thick_p(request):
    thickness_range = request.session.get('thickness_range', [])
    R_list = request.session.get('Rp_thick', [])
    T_list = request.session.get('Tp_thick', [])
    A_list = request.session.get('Ap_thick', [])

    data = {
        'Thickness (nm)': thickness_range,
        'Reflectance (R)': R_list,
        'Transmittance (T)': T_list,
        'Absorbance (A)': A_list
    }

    df = pd.DataFrame(data)

    # Create a BytesIO buffer to hold the Excel file
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='RTA Data p-polarized')

    # Rewind the buffer
    buffer.seek(0)

    # Return the Excel file as a response
    response = HttpResponse(buffer, content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    response['Content-Disposition'] = 'attachment; filename=RTA_data_thick_p_polarized.xlsx'

    return response

def download_excel_RTA_thick_s(request):
    thickness_range = request.session.get('thickness_range', [])
    R_list = request.session.get('Rs_thick', [])
    T_list = request.session.get('Ts_thick', [])
    A_list = request.session.get('As_thick', [])

    data = {
        'Thickness (nm)': thickness_range,
        'Reflectance (R)': R_list,
        'Transmittance (T)': T_list,
        'Absorbance (A)': A_list
    }

    df = pd.DataFrame(data)

    # Create a BytesIO buffer to hold the Excel file
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='RTA Data s-polarized')

    # Rewind the buffer
    buffer.seek(0)

    # Return the Excel file as a response
    response = HttpResponse(buffer, content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    response['Content-Disposition'] = 'attachment; filename=RTA_data_thick_s_polarized.xlsx'

    return response

def download_excel_RTA_thick_unpolarized(request):
    thickness_range = request.session.get('thickness_range', [])
    R_list = request.session.get('R_thick', [])
    T_list = request.session.get('T_thick', [])
    A_list = request.session.get('A_thick', [])

    data = {
        'Thickness (nm)': thickness_range,
        'Reflectance (R)': R_list,
        'Transmittance (T)': T_list,
        'Absorbance (A)': A_list
    }

    df = pd.DataFrame(data)

    # Create a BytesIO buffer to hold the Excel file
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='RTA Data unpolarized-polarized')

    # Rewind the buffer
    buffer.seek(0)

    # Return the Excel file as a response
    response = HttpResponse(buffer, content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    response['Content-Disposition'] = 'attachment; filename=RTA_data_thick_unpolarized.xlsx'

    return response

from .forms import LightSourceForm, MultilayerForm

def led(request):
    if request.method == 'POST':
        light_form = LightSourceForm(request.POST)
        multi_form = MultilayerForm(request.POST)

        if light_form.is_valid() and multi_form.is_valid():
            # Get selected IDs from forms
            selected_led_id = light_form.cleaned_data['led_choice']
            selected_multilayer_id = multi_form.cleaned_data['multilayer_choice']
            
            # Fetch data from the database
            light_source_data = LEDSpectrumData.objects.get(id=selected_led_id)
            multilayer_data = MultilayerFilmData.objects.get(id=selected_multilayer_id)

            # Extract data
            light_data = light_source_data.parsed_data
            wavelengths = [row[0] for row in light_data]
            intensities = [row[1] for row in light_data]
            
            multilayer_data_list = multilayer_data.parsed_data
            multilayer_wavelengths = [row[0] for row in multilayer_data_list]
            reflectance = [row[1] for row in multilayer_data_list]
            transmittance = [row[2] for row in multilayer_data_list]
            absorbance = [row[3] for row in multilayer_data_list]

            common_wavelengths = list(set(multilayer_wavelengths).intersection(wavelengths))
            print(common_wavelengths)
            reflectance_interpolated = np.interp(wavelengths, multilayer_wavelengths, reflectance)
            transmittance_interpolated = np.interp(wavelengths, multilayer_wavelengths, transmittance)
            absorbance_interpolated = np.interp(wavelengths, multilayer_wavelengths, absorbance)

            # Calculate reflected, transmitted, and absorbed light
            reflected_light = [i * r for i, r in zip(intensities, reflectance_interpolated)]
            transmitted_light = [i * t for i, t in zip(intensities, transmittance_interpolated)]
            absorbed_light = [i * a for i, a in zip(intensities, absorbance_interpolated)]
            
            cie_data = pd.read_csv('CIE_cc_1931_2deg.csv', header=None, names=['Wavelength', 'x_cmf', 'y_cmf', 'z_cmf'])

            # Extract CMF values
            wavelengths_cmf = cie_data['Wavelength'].values
            x_cmf = cie_data['x_cmf'].values
            y_cmf = cie_data['y_cmf'].values
            z_cmf = cie_data['z_cmf'].values

            # Interpolate the CMFs to match the wavelengths of the transmitted light
            x_interp = np.interp(wavelengths, wavelengths_cmf, x_cmf)
            y_interp = np.interp(wavelengths, wavelengths_cmf, y_cmf)
            z_interp = np.interp(wavelengths, wavelengths_cmf, z_cmf)

            X_input = np.sum(intensities * x_interp)
            Y_input = np.sum(intensities * y_interp)
            Z_input = np.sum(intensities * z_interp)

            # Calculate XYZ for transmitted light (after multilayer)
            X_transmitted = np.sum(transmitted_light * x_interp)
            Y_transmitted = np.sum(transmitted_light * y_interp)
            Z_transmitted = np.sum(transmitted_light * z_interp)

            x_input, y_input = xyz_to_xy(X_input, Y_input, Z_input)

            # Transmitted light xy coordinates
            x_transmitted, y_transmitted = xyz_to_xy(X_transmitted, Y_transmitted, Z_transmitted)
            cie_x = x_cmf / (x_cmf + y_cmf + z_cmf)
            cie_y = y_cmf / (x_cmf + y_cmf + z_cmf)

            plt.figure()

            # Generate the chromaticity diagram colors
            x_vals = np.linspace(-0.1, 0.8, 300)
            y_vals = np.linspace(-0.1, 0.9, 300)
            X_grid, Y_grid = np.meshgrid(x_vals, y_vals)

            colors = np.zeros((300, 300, 3))
            for i in range(300):
                for j in range(300):
                    colors[j, i, :] = xy_to_rgb(X_grid[j, i], Y_grid[j, i])

            # Create a path from the CIE boundary
            boundary_path = Path(np.column_stack((cie_x, cie_y)))

            # Determine which points are inside the boundary
            points = np.column_stack((X_grid.ravel(), Y_grid.ravel()))
            inside_mask = boundary_path.contains_points(points).reshape(300, 300)

            # Create a white background color array
            white_background = np.ones((300, 300, 3))

            # Apply the mask to the color array
            masked_colors = np.where(inside_mask[:, :, np.newaxis], colors, white_background)

            # Plotting
            plt.figure(figsize=(12, 9))

            # Show the color image with masking applied
            plt.imshow(masked_colors, extent=(-0.1, 0.8, -0.1, 0.9), origin='lower')

            # Plot the CIE boundary
            plt.plot(cie_x, cie_y, 'k-', linewidth=1.5, label='CIE Boundary')
            wavelengths_to_label = [440, 460, 480, 500, 520, 540, 560, 580, 600, 620, 640]
            interp_x = interp1d(wavelengths_cmf, cie_x)
            interp_y = interp1d(wavelengths_cmf, cie_y)

            offset = 0.01  # Offset for placing the text outside the colormap
            for wavelength in wavelengths_to_label:
                x_label = interp_x(wavelength)
                y_label = interp_y(wavelength)
                
                # Add a small dash at the boundary point
                
                # Determine the position for the text outside the colormap
                if x_label < 0.2:
                    x_text = x_label - offset
                    ha = 'right'
                else:
                    x_text = x_label + offset
                    ha = 'left'
                
                if y_label < 0.2:
                    y_text = y_label - offset
                    va = 'top'
                else:
                    y_text = y_label + offset
                    va = 'bottom'
                plt.text(x_text, y_text, f'{int(wavelength)} nm', fontsize=8, ha=ha, va=va)
                plt.plot(x_label, y_label, 'x', color='black')  # Add a dash marker



       

            # Plot input light and transmitted light
            plt.scatter([x_input], [y_input], color='red', label='Input Light')
            plt.scatter([x_transmitted], [y_transmitted], color='blue', label='Transmitted Light')

            plt.title('CIE 1931 Chromaticity Diagram')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.xlim(-0.1, 0.8)
            plt.ylim(-0.1, 0.9)
            plt.legend()
            

            #plt.show()
            # Saving the CIE 1931 chromaticity diagram
            plt.savefig('static/thinfilms/cie_diagram.png')
            plt.close()

          

            # Create the Plotly figure
            fig_cie = go.Figure()


            # Plot the CIE diagram boundary
            fig_cie.add_trace(go.Scatter(x=cie_x, y=cie_y, mode='lines', name='CIE Boundary'))

            # Plot the input light on the CIE diagram
            fig_cie.add_trace(go.Scatter(x=[x_input], y=[y_input], mode='markers', name='Input Light', marker=dict(color='red', size=10)))

            # Plot the transmitted light on the CIE diagram
            fig_cie.add_trace(go.Scatter(x=[x_transmitted], y=[y_transmitted], mode='markers', name='Transmitted Light', marker=dict(color='blue', size=10)))

            fig_cie.update_layout(
                title='CIE 1931 Chromaticity Diagram',
                xaxis_title='x',
                yaxis_title='y',
                xaxis=dict(range=[-0.1, 0.8], showgrid=False),
                yaxis=dict(range=[-0.1, 0.9], showgrid=False),
                showlegend=True,
            )
        
            # Convert Plotly figure to JSON
            graph_cie_json = json.dumps(fig_cie, cls=plotly.utils.PlotlyJSONEncoder)

            


            # Create Plotly graph for Transmitted Light
            fig_transmitted = go.Figure()
            fig_transmitted.add_trace(go.Scatter(x=wavelengths, y=transmitted_light, mode='lines+markers', name='Transmitted Light'))
            fig_transmitted.add_trace(go.Scatter(x=wavelengths, y=reflected_light, mode='lines+markers', name='Reflected Light'))
            fig_transmitted.add_trace(go.Scatter(x=wavelengths, y=absorbed_light, mode='lines+markers', name='Absorbed Light'))
            fig_transmitted.add_trace(go.Scatter(x=wavelengths, y=intensities, mode='lines+markers', name='Normalized Intensity'))
            fig_transmitted.update_layout(
                title='Light Through Multilayer',
                xaxis_title='Wavelength (nm)',
                yaxis_title='Intensity',
            )
            graph_transmitted_json = json.dumps(fig_transmitted, cls=plotly.utils.PlotlyJSONEncoder)



             # Plotly graph
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=wavelengths, y=intensities, mode='lines+markers', name='Normalized Intensity'))
            fig.update_layout(
                title='Light Source Data',
                xaxis_title='Wavelength (nm)',
                yaxis_title='Normalized Intensity',
            )
            # Convert the plotly graph to JSON
            graph_light_json  = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

            
            # Create Plotly graph for Multilayer Data
            fig_multi = go.Figure()
            fig_multi.add_trace(go.Scatter(x=multilayer_wavelengths, y=reflectance, mode='lines+markers', name='Reflectance'))
            fig_multi.add_trace(go.Scatter(x=multilayer_wavelengths, y=transmittance, mode='lines+markers', name='Transmittance'))
            fig_multi.add_trace(go.Scatter(x=multilayer_wavelengths, y=absorbance, mode='lines+markers', name='Absorbance'))
            fig_multi.update_layout(
                title='Multilayer Data',
                xaxis_title='Wavelength (nm)',
                yaxis_title='Percentage (%)',
            )
            graph_multi_json = json.dumps(fig_multi, cls=plotly.utils.PlotlyJSONEncoder)
          

            context = {
                'light_form': light_form,
                'multi_form': multi_form,
                'light_source_data': light_source_data,
                'multilayer_data': multilayer_data,
                'graph_light_json': graph_light_json,
                'graph_multi_json': graph_multi_json,
                'graph_transmitted_json': graph_transmitted_json,
                'graph_cie_json':graph_cie_json,
            }
            
            return render(request, 'thinfilms/led.html', context)
            
    else:
        light_form = LightSourceForm()
        multi_form = MultilayerForm()
        light_source_data = None
        multilayer_data = None

    context = {
        'light_form': light_form,
        'multi_form': multi_form,
        'light_source_data': light_source_data,
        'multilayer_data': multilayer_data,
    }
    return render(request, 'thinfilms/led.html', context)


def xy_to_rgb(x, y, Y=1.0):
    if y == 0:
        return (0.0, 0.0, 0.0)
    
    z = 1.0 - x - y
    X = (x / y) * Y if y != 0 else 0
    Z = (z / y) * Y if y != 0 else 0
    
    M = np.array([[3.2406, -1.5372, -0.4986],
                  [-0.9689, 1.8758, 0.0415],
                  [0.0557, -0.2040, 1.0570]])
    
    rgb = np.dot(M, np.array([X, Y, Z]))
    rgb = np.clip(rgb, 0, 1)
    
    return tuple(rgb)
def create_cie_plot(cie_x, cie_y, x_vals, y_vals):
    # Create mesh grid
    X_grid, Y_grid = np.meshgrid(x_vals, y_vals)
    
    # Convert xy coordinates to RGB
    colors = np.zeros((len(x_vals), len(y_vals), 3))
    for i in range(len(x_vals)):
        for j in range(len(y_vals)):
            colors[i, j, :] = xy_to_rgb(X_grid[i, j], Y_grid[i, j])

    # Create mask for inside the boundary
    points = np.column_stack((cie_x, cie_y))
    tri = Delaunay(points)
    
    def is_inside(x, y):
        return tri.find_simplex(np.array([x, y])) >= 0

    mask = np.array([[is_inside(x, y) for x in x_vals] for y in y_vals])
    
    # Apply mask: white color for out-of-boundary areas
    colors[~mask] = [1.0, 1.0, 1.0]  # White
    
    # Create the Plotly figure
    fig = go.Figure()
    
    # Add the colored background
    fig.add_trace(go.Heatmap(
        z=np.flipud(colors[:, :, 0]),  # Red channel
        colorscale=[[0, 'white'], [1, 'red']],
        showscale=False,
        zmin=0, zmax=1,
        x=np.linspace(0, 0.8, len(x_vals)),
        y=np.linspace(0, 0.9, len(y_vals))
    ))

    fig.add_trace(go.Heatmap(
        z=np.flipud(colors[:, :, 1]),  # Green channel
        colorscale=[[0, 'white'], [1, 'green']],
        showscale=False,
        zmin=0, zmax=1,
        x=np.linspace(0, 0.8, len(x_vals)),
        y=np.linspace(0, 0.9, len(y_vals))
    ))

    fig.add_trace(go.Heatmap(
        z=np.flipud(colors[:, :, 2]),  # Blue channel
        colorscale=[[0, 'white'], [1, 'blue']],
        showscale=False,
        zmin=0, zmax=1,
        x=np.linspace(0, 0.8, len(x_vals)),
        y=np.linspace(0, 0.9, len(y_vals))
    ))

    # Plot the CIE diagram boundary
    fig.add_trace(go.Scatter(x=cie_x, y=cie_y, mode='lines', name='CIE Boundary', line=dict(color='black')))
    
    fig.update_layout(
        title='CIE 1931 Chromaticity Diagram',
        xaxis_title='x',
        yaxis_title='y',
        xaxis=dict(range=[0, 0.8], showgrid=False),
        yaxis=dict(range=[0, 0.9], showgrid=False),
        showlegend=True
    )
    
    return fig

def xyz_to_xy(X, Y, Z):
    x = X / (X + Y + Z)
    y = Y / (X + Y + Z)
    return x, y

def manage_layers(request):
    # Create the formset factory with the desired extra forms
    
    if request.method == 'POST':
        MatrixFormSet = formset_factory(LayerFormSet, extra=1)
        layers=[]
        thickness_list=[]
        matrix_formset = MatrixFormSet(request.POST, request.FILES)
        layerdataform = LayerDataForm(request.POST)
        if matrix_formset.is_valid():

            if 'add_matrix' in request.POST:

                matrix_formset = formset_factory(LayerFormSet, extra=1)(initial=[form.cleaned_data for form in matrix_formset])
            
            elif 'delete_matrix' in request.POST:

                if matrix_formset.total_form_count()>1:

                    matrix_formset = formset_factory(LayerFormSet, extra=-1)(initial=[form.cleaned_data for form in matrix_formset])
            elif 'submitlayers' in request.POST:
                # Process each form in the formset and save data to the database
                layers=[]
                thickness_list=[]
                data_dict = {}
                multilayer_name = []
                for form in matrix_formset:
                    if form.is_valid():
                        data = form.cleaned_data
                        name = data.get('name')
                        thickness = data.get('thickness')
                        thickness_list.append(thickness)
                        #layers.append(LayerData.objects.get(id=name))
                        layer = LayerData.objects.get(id=name)
                        layers.append(layer)
                        multilayer_name.append(f"{layer.name}({thickness} nm)")
                        # Extract parsed data and store in a dictionary
                        parsed_data = layer.parsed_data
                        for wavelength, n, k in parsed_data:
                            if wavelength not in data_dict:
                                data_dict[wavelength] = []
                            data_dict[wavelength].append((n, k))
                common_wavelengths = [float(w) for w, nk in data_dict.items() if len(nk) == len(layers)]
                multilayer_name = " ".join(multilayer_name)
               
                final_data = []
                for wavelength in common_wavelengths:
                    nk_values = data_dict[wavelength]
                    final_data.append((wavelength, nk_values))

                
                form = GraphWavelength(request.POST, wavelengths=common_wavelengths)
                form1 = Wavelength(request.POST, wavelengths=common_wavelengths)
           


                context = {
                    'formset': matrix_formset,
                    'layerdataform': layerdataform,
                    'layers': zip(layers, thickness_list),
                    'final_data': final_data,
                    'books': [layer.name for layer in layers],
                    'first_wavelength': final_data[0][0],
                    'last_wavelength': final_data[-1][0],
                    'multilayer_name':multilayer_name,
                    'form':form,
                    'form1':form1,
                }
                request.session['layer_ids'] = [layer.id for layer in layers]
                request.session['thickness_list'] = thickness_list
                
                # Redirect to the result view
                return redirect('thinfilms:result_calc')
                #return render(request, 'thinfilms/layers.html', context)
    else:
        MatrixFormSet = formset_factory(LayerFormSet, extra=1)
        
        matrix_formset = MatrixFormSet()

        layerdataform = LayerDataForm()
        layers=[]
        thickness_list=[]
    
    context = {'formset': matrix_formset,'layerdataform':layerdataform,'layers':zip(layers,thickness_list)}
    return render(request, 'thinfilms/layers.html', context)
def result_calc(request):
    
    layer_ids = request.session.get('layer_ids')
    layers = [LayerData.objects.get(id=id) for id in layer_ids]
    thickness_list=request.session.get('thickness_list')
    data_dict = {}
    multilayer_name = []
    layers_list = []
    for layer,thickness in zip(layers,thickness_list):
        multilayer_name.append(f"{layer.name}({thickness} nm)")
        wl_data = [data[0] for data in layer.parsed_data]  # Extract wavelengths
        nk_data = [data[1] + 1j* data[2] for data in layer.parsed_data]
        layers_list.append([thickness,wl_data, nk_data])
        # Extract parsed data and store in a dictionary
        parsed_data = layer.parsed_data
        for wavelength, n, k in parsed_data:
            if wavelength not in data_dict:
                data_dict[wavelength] = []
            data_dict[wavelength].append((n, k))
    common_wavelengths = [w for w, nk in data_dict.items() if len(nk) == len(layers)]
    multilayer_name_str = multilayer_name
    multilayer_name = " ".join(multilayer_name)
    print(layer.parsed_data)
    # Build final data for display
    final_data = []
    for wavelength in common_wavelengths:
        nk_values = data_dict[wavelength]
        final_data.append((wavelength, nk_values))

    if request.method == 'POST':
        form = GraphWavelength(request.POST, wavelengths=common_wavelengths)
        form1 = GraphAngle(request.POST, wavelengths=common_wavelengths)
        form3 = GraphThickness(request.POST, wavelengths=common_wavelengths, layers=layers)
        if form.is_valid() and form1.is_valid() and form3.is_valid() :
            incidence_angle = form.cleaned_data.get('incidence_angle', 0)
            min_wavelength = form.cleaned_data.get('min_wavelength')
            max_wavelength = form.cleaned_data.get('max_wavelength')

            chosen_wavelength = form1.cleaned_data.get('chosen_wavelength')
            min_angle = float(form1.cleaned_data.get('min_angle'))
            max_angle = float(form1.cleaned_data.get('max_angle'))
            step_angle = int(form1.cleaned_data.get('step_angle'))


            chosen_layer_thick = int(form3.cleaned_data.get('chosen_layer'))
            incidence_angle_thick = form3.cleaned_data.get('incidence_angle_thick', 0)
            chosen_wavelength_thick = float(form3.cleaned_data.get('chosen_wavelength_thick'))
            min_thick = form3.cleaned_data.get('min_thick')
            max_thick = form3.cleaned_data.get('max_thick')
            step_thick = form3.cleaned_data.get('step_thick')

            # Filter common_wavelengths based on min and max values
            common_wavelengths = [wl for wl in common_wavelengths 
                                    if float(min_wavelength) <= wl <= float(max_wavelength)]
            

    else:
        form = GraphWavelength(wavelengths=common_wavelengths)
        form1 = GraphAngle(wavelengths=common_wavelengths)
        form3 = GraphThickness(wavelengths=common_wavelengths, layers=layers)
        incidence_angle = 0
        chosen_wavelength = min(common_wavelengths)
        min_angle = 0
        max_angle = 90
        step_angle = 1000
        
        chosen_layer_thick = len(layers)-1 
        chosen_wavelength_thick = min(common_wavelengths)
        incidence_angle_thick = 0
        min_thick = 0
        max_thick = 100
        step_thick = 1000

    
    Rp_list = []
    Tp_list = []
    Ap_list = []

    Rs_list = []
    Ts_list = []
    As_list = []

    R_list = []
    T_list = []
    A_list = []
    for wl in common_wavelengths:
        Rp, Tp, Ap = calculate_RT(layers_list, wl, "p", np.radians(incidence_angle)) 
        Rs, Ts, As = calculate_RT(layers_list, wl, "s", np.radians(incidence_angle))
        Rp_list.append(Rp)
        Tp_list.append(Tp)
        Ap_list.append(Ap)

        Rs_list.append(Rs)
        Ts_list.append(Ts)
        As_list.append(As)

        R_list.append((Rp+Rs)/2)
        T_list.append((Tp+Ts)/2)
        A_list.append((Ap+As)/2)
    
    # Convert lists to numpy arrays for further processing
    Rp_array = np.array(Rp_list)
    Tp_array = np.array(Tp_list)
    Ap_array = np.array(Ap_list)
    RTA_p = zip(common_wavelengths, Rp_array, Tp_array, Ap_array)
    
    Rs_array = np.array(Rs_list)
    Ts_array = np.array(Ts_list)
    As_array = np.array(As_list)
    RTA_s = zip(common_wavelengths, Rs_array, Ts_array, As_array)
    
    R_array = np.array(R_list)
    T_array = np.array(T_list)
    A_array = np.array(A_list)
    RTA_unpolarized = zip(common_wavelengths, R_array, T_array, A_array)


    # Generate plots for each polarization
    fig_p = go.Figure()
    fig_p.add_trace(go.Scatter(x=common_wavelengths, y=Rp_array, mode='lines+markers', name='Reflectance'))
    fig_p.add_trace(go.Scatter(x=common_wavelengths, y=Tp_array, mode='lines+markers', name='Transmittance'))
    fig_p.add_trace(go.Scatter(x=common_wavelengths, y=Ap_array, mode='lines+markers', name='Absorbance'))
    fig_p.update_layout(
        title=f'p-polarized at {incidence_angle}° on {multilayer_name} ',
        xaxis_title='Wavelength (nm)',
        yaxis_title='%'
    )
    graph_p_json = json.dumps(fig_p, cls=plotly.utils.PlotlyJSONEncoder)
    
    fig_s = go.Figure()
    fig_s.add_trace(go.Scatter(x=common_wavelengths, y=Rs_array, mode='lines+markers', name='Reflectance'))
    fig_s.add_trace(go.Scatter(x=common_wavelengths, y=Ts_array, mode='lines+markers', name='Transmittance'))
    fig_s.add_trace(go.Scatter(x=common_wavelengths, y=As_array, mode='lines+markers', name='Absorbance'))
    fig_s.update_layout(
        title=f's-polarized at {incidence_angle}° on {multilayer_name}',
        xaxis_title='Wavelength (nm)',
        yaxis_title='%'
    )
    graph_s_json = json.dumps(fig_s, cls=plotly.utils.PlotlyJSONEncoder)
    
    fig_unpolarized = go.Figure()
    fig_unpolarized.add_trace(go.Scatter(x=common_wavelengths, y=R_array, mode='lines+markers', name='Reflectance'))
    fig_unpolarized.add_trace(go.Scatter(x=common_wavelengths, y=T_array, mode='lines+markers', name='Transmittance'))
    fig_unpolarized.add_trace(go.Scatter(x=common_wavelengths, y=A_array, mode='lines+markers', name='Absorbance'))
    fig_unpolarized.update_layout(
        title=f'unpolarized at {incidence_angle}° on {multilayer_name}',
        xaxis_title='Wavelength (nm)',
        yaxis_title='%'
    )
    graph_unpolarized_json = json.dumps(fig_unpolarized, cls=plotly.utils.PlotlyJSONEncoder)

    angles = np.linspace(min_angle, max_angle, step_angle)  # Example angles from 0 to 90 degrees
    Rp_angle = []
    Tp_angle = []
    Ap_angle = []

    Rs_angle = []
    Ts_angle = []
    As_angle = []

    R_angle = []
    T_angle = []
    A_angle = []
    
    for angle in angles:

        Rp, Tp, Ap = calculate_RT(layers_list, float(chosen_wavelength), "p", np.radians(angle)) # calculate_RT_p(layers, wl, 0)
        Rs, Ts, As = calculate_RT(layers_list, float(chosen_wavelength), "s", np.radians(angle))
        Rp_angle.append(Rp)
        Tp_angle.append(Tp)
        Ap_angle.append(Ap)

        Rs_angle.append(Rs)
        Ts_angle.append(Ts)
        As_angle.append(As)

        R_angle.append((Rp+Rs)/2)
        T_angle.append((Tp+Ts)/2)
        A_angle.append((Ap+As)/2)

    RTA_angle_p = zip(angles, Rp_angle, Tp_angle, Ap_angle)
    RTA_angle_s = zip(angles, Rs_angle, Ts_angle, As_angle)
    RTA_angle_unpolarized = zip(angles, R_angle, T_angle, A_angle)
    
    # Plot the data
    fig_angle_unpolarized = go.Figure()
    fig_angle_unpolarized.add_trace(go.Scatter(x=angles, y=np.array(R_angle), mode='lines+markers', name='Reflectance'))
    fig_angle_unpolarized.add_trace(go.Scatter(x=angles, y=np.array(T_angle), mode='lines+markers', name='Transmittance'))
    fig_angle_unpolarized.add_trace(go.Scatter(x=angles, y=np.array(A_angle), mode='lines+markers', name='Absorbance'))
    fig_angle_unpolarized.update_layout(
        title=f'Unpolarized at {chosen_wavelength} nm for {multilayer_name}',
        xaxis_title='Angle (degrees)',
        yaxis_title='%'
    )
    graph_angle_unpolarized_json = json.dumps(fig_angle_unpolarized, cls=plotly.utils.PlotlyJSONEncoder)

    fig_angle_s = go.Figure()
    fig_angle_s.add_trace(go.Scatter(x=angles, y=np.array(Rs_angle), mode='lines+markers', name='Reflectance'))
    fig_angle_s.add_trace(go.Scatter(x=angles, y=np.array(Ts_angle), mode='lines+markers', name='Transmittance'))
    fig_angle_s.add_trace(go.Scatter(x=angles, y=np.array(As_angle), mode='lines+markers', name='Absorbance'))
    fig_angle_s.update_layout(
        title=f'S-polarized at {chosen_wavelength} nm for {multilayer_name}',
        xaxis_title='Angle (degrees)',
        yaxis_title='%'
    )
    graph_angle_s_json = json.dumps(fig_angle_s, cls=plotly.utils.PlotlyJSONEncoder)

    brewster_index = np.argmin(Rp_angle)
    brewster_angle = angles[brewster_index]
    fig_angle_p = go.Figure()
    fig_angle_p.add_trace(go.Scatter(x=angles, y=np.array(Rp_angle), mode='lines+markers', name='Reflectance'))
    fig_angle_p.add_trace(go.Scatter(x=angles, y=np.array(Tp_angle), mode='lines+markers', name='Transmittance'))
    fig_angle_p.add_trace(go.Scatter(x=angles, y=np.array(Ap_angle), mode='lines+markers', name='Absorbance'))
    fig_angle_p.update_layout(
        title=f'P-polarized at {chosen_wavelength} nm for {multilayer_name}',
        xaxis_title='Angle (degrees)',
        yaxis_title='%',
        annotations=[dict(x=brewster_angle, y=Rp_angle[brewster_index], 
                      xref='x', yref='y', 
                      text=f'Brewster Angle: {brewster_angle:.2f}°', 
                      showarrow=True, arrowhead=2)]
    )
    graph_angle_p_json = json.dumps(fig_angle_p, cls=plotly.utils.PlotlyJSONEncoder)

    thickness_range = np.linspace(min_thick, max_thick, step_thick)  
    Rp_thick = []
    Tp_thick = []
    Ap_thick = []

    Rs_thick = []
    Ts_thick = []
    As_thick= []

    R_thick = []
    T_thick = []
    A_thick = []
    multilayer_name_thick = multilayer_name_str
    book = layers[chosen_layer_thick].name
    multilayer_name_thick[chosen_layer_thick] = f"{book}(thickness nm)"
    multilayer_name_thick =  ''.join(multilayer_name_thick)
    layers_thick = layers_list
    for thick in thickness_range:
        layers_thick[chosen_layer_thick][0] = thick 
        Rp, Tp, Ap = calculate_RT(layers_thick, chosen_wavelength_thick, "p", np.radians(incidence_angle_thick)) 
        Rs, Ts, As = calculate_RT(layers_thick, chosen_wavelength_thick, "s", np.radians(incidence_angle_thick))

        Rp_thick.append(Rp)
        Tp_thick.append(Tp)
        Ap_thick.append(Ap)

        Rs_thick.append(Rs)
        Ts_thick.append(Ts)
        As_thick.append(As)

        R_thick.append((Rp + Rs) / 2)
        T_thick.append((Tp + Ts) / 2)
        A_thick.append((Ap + As) / 2)

    RTA_thick_p = zip(thickness_range, Rp_thick, Tp_thick, Ap_thick)
    RTA_thick_s = zip(thickness_range, Rs_thick, Ts_thick, As_thick)
    RTA_thick_unpolarized = zip(thickness_range, R_thick, T_thick, A_thick)

    # Plot the data
    fig_thick_unpolarized = go.Figure()
    fig_thick_unpolarized.add_trace(go.Scatter(x=thickness_range, y=np.array(R_thick), mode='lines+markers', name='Reflectance'))
    fig_thick_unpolarized.add_trace(go.Scatter(x=thickness_range, y=np.array(T_thick), mode='lines+markers', name='Transmittance'))
    fig_thick_unpolarized.add_trace(go.Scatter(x=thickness_range, y=np.array(A_thick), mode='lines+markers', name='Absorbance'))
    fig_thick_unpolarized.update_layout(
        title=f'Unpolarized at {chosen_wavelength_thick} nm for {multilayer_name_thick}',
        xaxis_title='Thickness (nm)',
        yaxis_title='%'
    )
    graph_thick_unpolarized_json = json.dumps(fig_thick_unpolarized, cls=plotly.utils.PlotlyJSONEncoder)

    fig_thick_s = go.Figure()
    fig_thick_s.add_trace(go.Scatter(x=thickness_range, y=np.array(Rs_thick), mode='lines+markers', name='Reflectance'))
    fig_thick_s.add_trace(go.Scatter(x=thickness_range, y=np.array(Ts_thick), mode='lines+markers', name='Transmittance'))
    fig_thick_s.add_trace(go.Scatter(x=thickness_range, y=np.array(As_thick), mode='lines+markers', name='Absorbance'))
    fig_thick_s.update_layout(
        title=f'S-polarized at {chosen_wavelength_thick} nm for {multilayer_name_thick}',
        xaxis_title='Thickness (nm)',
        yaxis_title='%'
    )
    graph_thick_s_json = json.dumps(fig_thick_s, cls=plotly.utils.PlotlyJSONEncoder)

    fig_thick_p = go.Figure()
    fig_thick_p.add_trace(go.Scatter(x=thickness_range, y=np.array(Rp_thick), mode='lines+markers', name='Reflectance'))
    fig_thick_p.add_trace(go.Scatter(x=thickness_range, y=np.array(Tp_thick), mode='lines+markers', name='Transmittance'))
    fig_thick_p.add_trace(go.Scatter(x=thickness_range, y=np.array(Ap_thick), mode='lines+markers', name='Absorbance'))
    fig_thick_p.update_layout(
        title=f'P-polarized at {chosen_wavelength_thick} nm for {multilayer_name_thick}',
        xaxis_title='Thickness (nm)',
        yaxis_title='%'
    )
    graph_thick_p_json = json.dumps(fig_thick_p, cls=plotly.utils.PlotlyJSONEncoder)

    request.session['common_wavelengths'] = common_wavelengths

    request.session['Rp_list'] = Rp_list
    request.session['Tp_list'] = Tp_list
    request.session['Ap_list'] = Ap_list

    request.session['Rs_list'] = Rs_list
    request.session['Ts_list'] = Ts_list
    request.session['As_list'] = As_list

    request.session['R_list'] = R_list
    request.session['T_list'] = T_list
    request.session['A_list'] = A_list
   
    request.session['angles'] = angles.tolist()
    request.session['Rp_angle'] = Rp_angle
    request.session['Tp_angle'] = Tp_angle
    request.session['Ap_angle'] = Ap_angle

    request.session['Rs_angle'] = Rs_angle
    request.session['Ts_angle'] = Ts_angle
    request.session['As_angle'] = As_angle

    request.session['R_angle'] = R_angle
    request.session['T_angle'] = T_angle
    request.session['A_angle'] = A_angle

    request.session['thickness_range'] = thickness_range.tolist()
    request.session['Rp_thick'] = Rp_thick
    request.session['Tp_thick'] = Tp_thick
    request.session['Ap_thick'] = Ap_thick

    request.session['Rs_thick'] = Rs_thick
    request.session['Ts_thick'] = Ts_thick
    request.session['As_thick'] = As_thick

    request.session['R_thick'] = R_thick
    request.session['T_thick'] = T_thick
    request.session['A_thick'] = A_thick

    context = {
        'layers': zip(layers, thickness_list),
        'final_data': final_data,
        'books': [layer.name for layer in layers],
        'first_wavelength': final_data[0][0],
        'last_wavelength': final_data[-1][0],
        'multilayer_name':multilayer_name,
        'form':form,
        'form1':form1,
        'form3':form3,
        'graph_p_json': graph_p_json,
        'graph_s_json': graph_s_json,
        'graph_unpolarized_json': graph_unpolarized_json,
        'RTA_p': RTA_p,
        'RTA_s': RTA_s,
        'RTA_unpolarized': RTA_unpolarized,
        
        'RTA_angle_p': RTA_angle_p,
        'RTA_angle_s': RTA_angle_s,
        'RTA_angle_unpolarized': RTA_angle_unpolarized,
      
        'graph_angle_unpolarized_json':graph_angle_unpolarized_json,
        'graph_angle_s_json':graph_angle_s_json,
        'graph_angle_p_json':graph_angle_p_json,

        'graph_thick_unpolarized_json': graph_thick_unpolarized_json,
        'graph_thick_s_json': graph_thick_s_json,
        'graph_thick_p_json': graph_thick_p_json,
        'RTA_thick_p': RTA_thick_p,
        'RTA_thick_s': RTA_thick_s,
        'RTA_thick_unpolarized': RTA_thick_unpolarized,
        'min_angle':min(angles),
        'max_angle':max(angles),
        'angles':angles,
        'wavelength_angle': chosen_wavelength,
        'multilayer_name_thick':multilayer_name_thick,


    }
    #context = {'layers': layers}

    return render(request, 'thinfilms/result_calc.html', context)


def data(request):
    if request.method == 'POST':
        name = request.POST.get('name')
        description = request.POST.get('description')
        data_file = request.FILES.get('data_file')
        data_type = request.POST.get('data_type')  # New field to identify the type of data

        if name and description and data_file and data_type:
            if data_type == 'layer':
                LayerData.objects.create(name=name, description=description, filedata=data_file)
            elif data_type == 'led':
                LEDSpectrumData.objects.create(name=name, description=description, filedata=data_file)
            elif data_type == 'multilayer':
                MultilayerFilmData.objects.create(name=name, description=description, filedata=data_file)

        elif 'delete' in request.POST:
            data_id = request.POST.get('id')
            data_type = request.POST.get('data_type')
            if data_type == 'layer':
                LayerData.objects.filter(id=data_id).delete()
            elif data_type == 'led':
                LEDSpectrumData.objects.filter(id=data_id).delete()
            elif data_type == 'multilayer':
                MultilayerFilmData.objects.filter(id=data_id).delete()

        return redirect('thinfilms:data')  # Redirect to avoid resubmission

    # Fetch data based on the selected data type
    selected_data_type = request.GET.get('data_type', 'layer')
    if selected_data_type == 'layer':
        data_entries = LayerData.objects.all()
    elif selected_data_type == 'led':
        data_entries = LEDSpectrumData.objects.all()
    elif selected_data_type == 'multilayer':
        data_entries = MultilayerFilmData.objects.all()
    else:
        data_entries = []

    context = {
        'data_entries': data_entries,
        'selected_data_type': selected_data_type
    }
    return render(request, 'thinfilms/data.html', context)

def download_file(request, layer_id):
    try:
        layer = LayerData.objects.get(id=layer_id)
        file_path = layer.filedata.path
        if not os.path.exists(file_path):
            raise Http404("File does not exist")
        with open(file_path, 'rb') as file:
            response = HttpResponse(file.read(), content_type='application/octet-stream')
            response['Content-Disposition'] = f'attachment; filename="{os.path.basename(file_path)}"'
            return response
    except LayerData.DoesNotExist:
        raise Http404("Layer does not exist")
def base(request):


    return render(request, 'thinfilms/base.html')

