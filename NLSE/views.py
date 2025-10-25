from django.shortcuts import render

# Create your views here.
import numpy as np
from scipy.fftpack import fft, ifft, fftshift, ifftshift, fftfreq

import matplotlib.pyplot as plt
from matplotlib import cm

global pi; pi=np.pi
import os
from django.conf import settings
import time
import shutil
from shutil import copyfile
from datetime import datetime
import os
import ast
from django.http import HttpResponse
from .forms import SimulationParameters, PulseForm,FiberParametersForm


import plotly.graph_objs as go
import plotly.offline as opy
import json
import plotly
import plotly.io as pio

from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.legend import LineCollection
from matplotlib.colors import ListedColormap, LinearSegmentedColormap


from io import BytesIO
from django.http import HttpResponse
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt
import shutil


import plotly.graph_objs as go
from plotly.subplots import make_subplots


def home(request):
    
    if request.method == "POST":
        form = SimulationParameters(request.POST)
        form1 = PulseForm(request.POST)
        form3 = FiberParametersForm(request.POST)
        if form.is_valid() and form1.is_valid() and form3.is_valid():
            N = form.cleaned_data.get('number_of_points')
            time_resolution_1 = form.cleaned_data.get('time_resolution_1')
            time_resolution_2 = form.cleaned_data.get('time_resolution_2')
            duration_k = form.cleaned_data.get('duration_k')
            amplitude = form1.cleaned_data.get('amplitude')
            offset = 0
            chirp = form1.cleaned_data.get('chirp')
            order = form1.cleaned_data.get('order')
            testCarrierFreq = form1.cleaned_data.get('testCarrierFreq')

            pulse_type = form1.cleaned_data.get('pulse_type')

            addnoise_bool = form1.cleaned_data.get('add_noise')
            addnoise_amplitude = form1.cleaned_data.get('noise_amplitude')

            dt = time_resolution_1 * (10 ** time_resolution_2)
            duration = duration_k*dt
            t = np.linspace(0, N * dt, N)
            t = t - np.mean(t)

            f=getFreqRangeFromTime(t)

            if pulse_type == "gaussian":
                testPulse=GaussianPulse(t, amplitude, duration, offset, chirp, order,testCarrierFreq)
            elif pulse_type=="sech":
                testPulse=sechPulse(t, amplitude, duration, offset, chirp,testCarrierFreq)
            elif pulse_type=="sinc":
                testPulse=sincPulse(t, amplitude, duration, offset, chirp, testCarrierFreq)
            elif pulse_type=="cw":
                testPulse=amplitude*np.ones_like(t)*(1+0j)

            if addnoise_bool == True:
                testPulse = testPulse + addNoise(t,amplitude/addnoise_amplitude)

           
            length_bool = form3.cleaned_data.get('length_bool')
            amplitude_bool = form3.cleaned_data.get('amplitude_bool')

            

            nsteps = form3.cleaned_data.get('nsteps')

            gamma_a = form3.cleaned_data.get('gamma_a')
            gamma_b = form3.cleaned_data.get('gamma_b')

            gamma = gamma_a * 10 ** gamma_b

            beta2_a = form3.cleaned_data.get('beta2_a')
            beta2_b = form3.cleaned_data.get('beta2_b')

            beta2 = beta2_a * 10 ** beta2_b
            beta2= beta2*1e-30

            alpha_dB_per_m_a = form3.cleaned_data.get('alpha_dB_per_m_a')
            alpha_dB_per_m_b = form3.cleaned_data.get('alpha_dB_per_m_b')

            alpha_dB_per_m = alpha_dB_per_m_a * 10 ** alpha_dB_per_m_b
            alpha_Np_per_m = alpha_dB_per_m*np.log(10)/10.0 #Loss coeff is usually specified in dB/km, but Nepers/km is more useful for calculations

            amplitude = form1.cleaned_data.get('amplitude')
            
            offset = 0
            chirp = form1.cleaned_data.get('chirp')
            order = form1.cleaned_data.get('order')
            testCarrierFreq = form1.cleaned_data.get('testCarrierFreq')

            
            

            if amplitude_bool==True:
                amplitude_charac = form3.cleaned_data.get('amplitude_charac')
                A_char = np.sqrt(np.abs(beta2)/gamma/(duration)**2)
                amplitude=A_char*amplitude_charac
            else:
                amplitude = form1.cleaned_data.get('amplitude')

            if length_bool==True:
                length_charac = form3.cleaned_data.get('length_charac')
                z_char= pi/2*duration**2/np.abs(beta2) #Characteristic length
                Length    = length_charac* z_char
            else:
                Length_a = form3.cleaned_data.get('Length_a')
                Length_b = form3.cleaned_data.get('Length_b')
                Length = Length_a * 10 ** Length_b


            if pulse_type == "gaussian":
                testPulse=GaussianPulse(t, amplitude, duration, offset, chirp, order,testCarrierFreq)
            elif pulse_type=="sech":
                testPulse=sechPulse(t, amplitude, duration, offset, chirp,testCarrierFreq)
            elif pulse_type=="sinc":
                testPulse=sincPulse(t, amplitude, duration, offset, chirp, testCarrierFreq)
            elif pulse_type=="cw":
                testPulse=amplitude*np.ones_like(t)*(1+0j)


            if addnoise_bool == True:
                testPulse = testPulse + addNoise(t,amplitude/addnoise_amplitude)

           
            pulseMatrix = np.zeros((nsteps+1,N) )*(1+0j)
            spectrumMatrix = np.copy(pulseMatrix)

            pulse = testPulse
            dz= Length/nsteps

            pulseMatrix[0,:]=pulse
            spectrumMatrix[0,:] = getSpectrumFromPulse(t, pulse)

            #Pre-calculate effect of dispersion and loss as it's the same everywhere
            disp_and_loss=np.exp((1j*beta2/2*(2*pi*f)**2-alpha_Np_per_m/2)*dz )
            
            #Precalculate constants for nonlinearity
            nonlinearity=1j*gamma*dz

            for n in range(nsteps):   
                pulse*=np.exp(nonlinearity*getPower(pulse)) #Apply nonlinearity
                spectrum = getSpectrumFromPulse(t, pulse)*disp_and_loss #Go to spectral domain and apply disp and loss
                pulse=getPulseFromSpectrum(f, spectrum) #Return to time domain 
                
                #Store results and repeat
                pulseMatrix[n+1,:]=pulse
                spectrumMatrix[n+1,:]=spectrum

            #return pulseMatrix, spectrumMatrix
            
           
            
            #if "plot_all" in request.POST:
            frames_lines = []
            frames_spectrum = []
            frames_chirp = []
            for frame in range(nsteps):
                i=frame 
                
                frame_data_lines = [
                        go.Scatter(x=t.tolist(), y=getPower(pulseMatrix[i,:]).tolist(), mode='lines', name=str(frame)),
                    ]
                
                frames_lines.append(go.Frame(data=frame_data_lines, name=str(frame)))

                frame_data_spectrum = [
                        go.Scatter(x=f.tolist(), y=getPower(spectrumMatrix[i,:]).tolist(), mode='lines', name=str(frame)),
                    ]
                
                frames_spectrum.append(go.Frame(data=frame_data_spectrum, name=str(frame)))


                frame_data_chirp = [
                        go.Scatter(x=t.tolist(), y=getChirp(t,pulseMatrix[i,:]).tolist(), mode='lines', name=str(frame)),
                    ]
                
                frames_chirp.append(go.Frame(data=frame_data_chirp, name=str(frame)))

            

            screen_distances = np.linspace(0, Length, nsteps)
            slider_steps = [
                    {
                        "args": [[str(i)], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
                        "label": f"{screen_distance:.2f} m",  # Convert to cm
                        "method": "animate"
                    } for i, screen_distance in enumerate(screen_distances)
                ]
            max_power = np.max(getPower(pulseMatrix))
            fig_lines = go.Figure(
                    data=frames_lines[0].data,
                    layout=go.Layout(
                        title="Pulse Evolution",
                        xaxis={
                            "title": "Time [s]",
                            "range": [-duration*1.5,duration*1.5],  # Fixed y-axis range based on the maximum power
                        },
                        yaxis={
                            "title": "Power [W]",
                            "range": [0, max_power*1.1],  # Fixed y-axis range based on the maximum power
                        },
                        
                        width=500,
                        height=500,
                        sliders=[{
                            "currentvalue": {
                                "prefix": "Distance: ",
                                "font": {"size": 12}
                            },
                            "steps": slider_steps,
                            "transition": {"duration": 300},
                            "x": 0.0,
                            "y": -0.1,  # Position the slider below the x-axis
                            "len": 1
                        }]
                    ),
                    frames=frames_lines
                )
            anim_lines_plot_data = pio.to_json(fig_lines)

            max_power_spectrum = np.max(getPower(spectrumMatrix))
            fig_spectrum = go.Figure(
                    data=frames_spectrum[0].data,
                    layout=go.Layout(
                        title="Spectrum Evolution",
                        xaxis={
                            "title": "Freq. [Hz]",
                            "range": [-(1/duration)*2.5,(1/duration)*2.5],  # Fixed y-axis range based on the maximum power
                        },
                        yaxis={
                            "title": "PSD [W/Hz]",
                            "range": [0, max_power_spectrum*1.1],  # Fixed y-axis range based on the maximum power
                        },
                        width=500,
                        height=500,
                        sliders=[{
                            "currentvalue": {
                                "prefix": "Distance: ",
                                "font": {"size": 12}
                            },
                            "steps": slider_steps,
                            "transition": {"duration": 300},
                            "x": 0.0,
                            "y": -0.1,  # Position the slider below the x-axis
                            "len": 1
                        }]
                    ),
                    frames=frames_spectrum
                )
            anim_lines_plot_data_spectrum = pio.to_json(fig_spectrum)


            max_power_chirp = np.max([getChirp(t,pulseMatrix[i,:]) for i in range(nsteps)])
            fig_chirp = go.Figure(
                    data=frames_chirp[0].data,
                    layout=go.Layout(
                        title="Chirp Evolution",
                        xaxis={
                            "title": "Time [s]",
                                # Fixed y-axis range based on the maximum power
                        },
                        yaxis={
                            "title": "Chirp [Hz]",
                            "range": [-max_power_chirp*1.1, max_power_chirp*1.1],  # Fixed y-axis range based on the maximum power
                        },
                        width=500,
                        height=500,
                        sliders=[{
                            "currentvalue": {
                                "prefix": "Distance: ",
                                "font": {"size": 12}
                            },
                            "steps": slider_steps,
                            "transition": {"duration": 300},
                            "x": 0.0,
                            "y": -0.1,  # Position the slider below the x-axis
                            "len": 1
                        }]
                    ),
                    frames=frames_chirp
                )
            anim_lines_plot_data_chirp = pio.to_json(fig_chirp)
            
            """ points = np.array( [t*1e12 ,  getPower(pulseMatrix[nsteps-1,:])   ]  ).T.reshape(-1,1,2)
            segments = np.concatenate([points[0:-1],points[1:]],axis=1)

            colors = ["red" ,"gray", "blue"]
            cmap1 = LinearSegmentedColormap.from_list("mycmap", colors)

            norm = plt.Normalize(-20,20)
            lc=LineCollection(segments,cmap=cmap1,norm=norm)
            lc.set_array( getChirp(t,pulseMatrix[nsteps-1,:])/1e9 )

            fig, ax = plt.subplots(dpi=150)
            line = ax.add_collection(lc)
            fig.colorbar(line,ax=ax, label = 'Chirp [GHz]')

            def init():
                ax.set_title(f'{pulse_type} pulse evolution')

                ax.set_xlim([-duration*2.5*1e12,duration*2.5*1e12])
                ax.set_ylim([0,1.05*np.max( np.abs(pulseMatrix) )**2])
                
                ax.set_xlabel('Time [ps]')
                ax.set_ylabel('Power [W]')
                #Function for updating the plot in the .gif
            def update(i):
                ax.clear() #Clear figure
                init()     #Reset axes

                #Make collection of points from pulse power
                points = np.array( [t*1e12 ,  getPower(pulseMatrix[i,:])   ]  ).T.reshape(-1,1,2)
                
                #Make collection of lines from points
                segments = np.concatenate([points[0:-1],points[1:]],axis=1)
                lc=LineCollection(segments,cmap=cmap1,norm=norm)

                #Activate norm function based on local chirp
                lc.set_array( getChirp(t,pulseMatrix[i,:])/1e9 )
                
                #Plot line
                line = ax.add_collection(lc)

            #Make animation
            ani = FuncAnimation(fig,update,range(nsteps),init_func=init)
            
            #Save animation as .gif
            framerate=30 # Framerate of 30 will look smooth
            writer = PillowWriter(fps=framerate)
            ani.save('static/NLSE/pulse_animation.gif',writer=writer)
            plt.clf()
            shutil.copy('static/NLSE/pulse_animation.gif', 'staticfiles/NLSE/pulse_animation.gif')
                """
            #context = {"form":form, "form1":form1,"form3":form3,'animation':True}

            #gif_file_path = 'static/NLSE/pulse_animation.gif'
            
            context = {"form":form, "form1":form1,"form3":form3,
                        'anim_lines_plot_data':anim_lines_plot_data,
                        'anim_lines_plot_data_spectrum':anim_lines_plot_data_spectrum,
                        'anim_lines_plot_data_chirp':anim_lines_plot_data_chirp,
                        
                        }



            
            if "downloadgif" in request.POST:
                gif_file_path = 'static/NLSE/pulse_animation.gif'

                with open(gif_file_path, 'rb') as gif_file:
                    response = HttpResponse(gif_file.read(), content_type='image/gif')
                    response['Content-Disposition'] = 'attachment; filename="pulse_animation.gif"'
                    return response
            
        
    else:
        form = SimulationParameters()
        form1 = PulseForm()
        form3 = FiberParametersForm()
        context = {"form":form, "form1":form1,"form3":form3}



    if request.headers.get("HX-Request") == "true":
        return render(request, "NLSE/partials/home_results.html",context)
    else:
        return render(request, "NLSE/home.html", context)

#################################################### PULSES #################
import plotly.graph_objects as go
import numpy as np
import plotly.io as pio

def func_pulseEvolutionAnimation(pulseMatrix, fiber, sim):
    """
    Creates a Plotly animation showing pulse evolution along the fiber using a slider.
    """
    t = sim.t * 1e12  # Time in picoseconds
    num_frames = pulseMatrix.shape[0]  # Total number of frames based on pulseMatrix
    frame_distances = np.linspace(0, fiber.Length, num_frames)  # Distance for each frame

    # Create frames for the animation
    frames = []
    for i in range(num_frames):
        pulse_data = np.abs(pulseMatrix[i, :]) ** 2  # Calculate power for current frame
        frames.append(
            go.Frame(
                data=[go.Scatter(
                    x=t.tolist(),
                    y=pulse_data.tolist(),
                    mode="lines",
                    line=dict(color='blue')
                )],
                name=f"Frame {i}"  # Frame name for reference
            )
        )

    # Initial trace (first frame data)
    initial_data = go.Scatter(
        x=t.tolist(),
        y=(np.abs(pulseMatrix[0, :]) ** 2).tolist(),
        mode="lines",
        line=dict(color='blue'),
        name="Pulse Power"
    )

    # Layout with a slider
    layout = go.Layout(
        title="Pulse Evolution Along the Fiber",
        xaxis={"title": "Time [ps]", "range": [np.min(t), np.max(t)]},
        yaxis={"title": "Power [W]", "range": [0, 1.05 * np.max(np.abs(pulseMatrix) ** 2)]},
        sliders=[
            {
                "steps": [
                    {
                        "args": [[f"Frame {i}"],  # Target the frame by its name
                                 {"frame": {"duration": 100, "redraw": True},
                                  "mode": "immediate"}],
                        "label": f"{frame_distances[i]:.2f} m",
                        "method": "animate"
                    } for i in range(num_frames)
                ],
                "active": 0,  # Start at the first frame
                "x": 0.1,
                "y": -0.2,
                "len": 0.9,
                "currentvalue": {
                    "font": {"size": 16},
                    "prefix": "Distance: ",
                    "visible": True,
                    "xanchor": "right"
                },
                "transition": {"duration": 100, "easing": "linear"}
            }
        ],
        transition={"duration": 100}  # Smooth frame transitions
    )

    # Create the figure
    fig = go.Figure(data=[initial_data], layout=layout, frames=frames)

    # Convert the figure to JSON
    animation_json = pio.to_json(fig)
    return animation_json

def GaussianPulse(time, amplitude, duration, offset, chirp, order, carrier_freq):
    
    Carrier_freq = np.exp(-1j * 2 * pi * carrier_freq * time)
    gaussian_pulse = (amplitude * np.exp(-(1 + 1j * chirp) / 2 * ((time - offset) / (duration)) ** (2 * order))* Carrier_freq)
    return gaussian_pulse


def addNoise(time,noiseAmplitude):
  randomAmplitudes=np.random.normal(loc=0.0, scale=noiseAmplitude, size=len(time))
  randomPhases = np.random.uniform(-pi,pi, len(time))
  return randomAmplitudes*np.exp(1j*randomPhases)


def sincPulse(time, amplitude, duration, offset, chirp, carrier_freq):
   

    carrier_freq = np.exp(-1j * 2 * pi * carrier_freq * time)
    chirp_factor = np.exp(-(1j * chirp) / 2 * ((time - offset) / (duration)) ** 2)
    sinc_pulse = (amplitude* np.sinc((time - offset) / (duration))* chirp_factor* carrier_freq)
    return sinc_pulse

def sechPulse(time,amplitude,duration,offset,chirp,carrier_freq):
    carrier_freq = np.exp(-1j * 2 * pi * carrier_freq * time)
    chirp_factor = np.exp(
        -(1j * chirp) / 2 * ((time - offset) / (duration)) ** 2
    )
    sech_pulse = (
        amplitude
        / np.cosh((time - offset) / duration)
        * chirp_factor
        * carrier_freq
    )
    return sech_pulse


#################################################### FUNCTIONS #################
def getFreqRangeFromTime(time):
    return fftshift(fftfreq(len(time), d=time[1]-time[0]))


def getPhase(pulse):
    phi=np.unwrap(np.angle(pulse)) 
    phi=phi-phi[int(len(phi)/2)]   
    return phi    


def getChirp(time,pulse):
    phi=getPhase(pulse)
    dphi=np.diff(phi ,prepend = phi[0] - (phi[1]  - phi[0]  ),axis=0) 
    dt  =np.diff(time,prepend = time[0]- (time[1] - time[0] ),axis=0) 

    return -1.0/(2*pi)*dphi/dt #Chirp = - 1/(2pi) * d(phi)/dt

def getPower(amplitude):
    return np.abs(amplitude)**2  

#Function gets the energy of a pulse pulse or spectrum by integrating the power
def getEnergy(time_or_frequency,amplitude):
    return np.trapz(getPower(amplitude),time_or_frequency)




def getSpectrumFromPulse(time,pulse_amplitude):
    pulseEnergy=getEnergy(time,pulse_amplitude) #Get pulse energy
    f=getFreqRangeFromTime(time) 
    dt=time[1]-time[0]
    
    spectrum_amplitude=fftshift(fft(pulse_amplitude))*dt #Take FFT and do shift
    
    return spectrum_amplitude


#Equivalent function for getting time base from frequency range
def getTimeFromFrequency(frequency):  
    return fftshift(fftfreq(len(frequency), d=frequency[1]-frequency[0]))


#Equivalent function for getting pulse from spectrum
def getPulseFromSpectrum(frequency,spectrum_amplitude):
    
    spectrumEnergy=getEnergy(frequency, spectrum_amplitude)
    
    time = getTimeFromFrequency(frequency)
    dt = time[1]-time[0]
     
    pulse = ifft(ifftshift(spectrum_amplitude))/dt
    
    
    return pulse

#################################################### SIMULATION  #################
class SIM_config:
    def __init__(self,N,dt):
        self.number_of_points=N
        self.time_step=dt
        t=np.linspace(0,N*dt,N)
        self.t=t-np.mean(t)
        self.tmin=self.t[0]
        self.tmax=self.t[-1]
        
        self.f=getFreqRangeFromTime(self.t)
        self.fmin=self.f[0]
        self.fmax=self.f[-1]
        self.freq_step=self.f[1]-self.f[0]


class Fiber_config:
  def __init__(self,nsteps,L,gamma,beta2,alpha_dB_per_m):
      self.nsteps=nsteps
      self.ntraces = self.nsteps+1 #Note: If we want to do 100 steps, we will get 101 calculated pulses (zeroth at the input + 100 computed ones)
      self.Length=L
      self.dz=L/nsteps
      self.zlocs=np.linspace(0,L,self.ntraces) #Locations of each calculated pulse
      self.gamma=gamma
      self.beta2=beta2
      self.alpha_dB_per_m=alpha_dB_per_m
      self.alpha_Np_per_m = self.alpha_dB_per_m*np.log(10)/10.0 #Loss coeff is usually specified in dB/km, but Nepers/km is more useful for calculations

def SSFM(fiber:Fiber_config,sim:SIM_config, pulse):
    
    #Initialize arrays to store pulse and spectrum throughout fiber
    pulseMatrix = np.zeros((fiber.nsteps+1,sim.number_of_points ) )*(1+0j)
    spectrumMatrix = np.copy(pulseMatrix)
    pulseMatrix[0,:]=pulse
    spectrumMatrix[0,:] = getSpectrumFromPulse(sim.t, pulse)

    #Pre-calculate effect of dispersion and loss as it's the same everywhere
    disp_and_loss=np.exp((1j*fiber.beta2/2*(2*pi*sim.f)**2-fiber.alpha_Np_per_m/2)*fiber.dz )
    
    #Precalculate constants for nonlinearity
    nonlinearity=1j*fiber.gamma*fiber.dz

    for n in range(fiber.nsteps):   
        pulse*=np.exp(nonlinearity*getPower(pulse)) #Apply nonlinearity
        spectrum = getSpectrumFromPulse(sim.t, pulse)*disp_and_loss #Go to spectral domain and apply disp and loss
        pulse=getPulseFromSpectrum(sim.f, spectrum) #Return to time domain 
        
        #Store results and repeat
        pulseMatrix[n+1,:]=pulse
        spectrumMatrix[n+1,:]=spectrum

    return pulseMatrix, spectrumMatrix

#################################################### PLOTING FUNCTIONS #################

def func_plotFirstAndLastPulse_fig(testPulse0,testPulseFinal, fiber, sim):
 
    t = sim.t * 1e12
    initial_pulse = getPower(testPulse0)
    final_pulse = getPower(testPulseFinal)

    trace_initial = {
        "x": t.tolist(),
        "y": initial_pulse.tolist(),
        "mode": "lines",
        "name": "Initial Pulse",
    }

    trace_final = {
        "x": t.tolist(),
        "y": final_pulse.tolist(),
        "mode": "lines",
        "name": "Final Pulse",
    }
    duration=2**7*sim.time_step
    layout = {
        "title": "Initial pulse and final pulse",
        "xaxis": { "title": 'Time [ps]',"range": [-duration*5*1e12,duration*5*1e12] },
        "yaxis": { "title": 'Power [W]'},
        "height": 700,
        
    }
    #np.copy(SSFM(fiber,sim_config,GaussianPulse(t, 1, duration, 0, 0, 1))[0][0,:])
    data = [trace_initial, trace_final]
    plot_data = {"data": data, "layout": layout}

    # Serialize the plot data to JSON
    plot_json = json.dumps(plot_data)

    return plot_json 

def func_plotFirstAndLastChirp_fig(testPulse0,testPulseFinal, fiber, sim):
    t = sim.t*1e12
    initial_pulse = getChirp(t/1e12,testPulse0)/1e9
    final_pulse = getChirp(t/1e12,testPulseFinal)/1e9

    trace_initial = {
        "x": t.tolist(),
        "y": initial_pulse.tolist(),
        "mode": "lines",
        "name": "Initial Chirp",
    }

    trace_final = {
        "x": t.tolist(),
        "y": final_pulse.tolist(),
        "mode": "lines",
        "name": "Final Chirp",
    }
    duration=2**7*sim.time_step
    layout = {
        "title": "Initial Chirp and final Chirp",
        "xaxis": { "title": 'Time [ps]',"range": [-duration*5*1e12,duration*5*1e12] },
        "yaxis": { "title": 'Chirp [GHz]',"range": [-40,40]},
        "height": 700,
        
    }
    #np.copy(SSFM(fiber,sim_config,GaussianPulse(t, 1, duration, 0, 0, 1))[0][0,:])
    data = [trace_initial, trace_final]
    plot_data = {"data": data, "layout": layout}

    # Serialize the plot data to JSON
    plot_json = json.dumps(plot_data)

    return plot_json 
  
def func_plotFirstAndLastSpectrum_fig(testSpectrum0,testSpectrumFinal,fiber:Fiber_config,sim):
    #f=sim.f[int(sim.number_of_points/2-nrange):int(sim.number_of_points/2+nrange)]/1e9    
    #initial_pulse = getPower(matrix[0, int(sim.number_of_points/2-nrange):int(sim.number_of_points/2+nrange)])*1e9
    #final_pulse = getPower(matrix[-1, int(sim.number_of_points/2-nrange):int(sim.number_of_points/2+nrange)])*1e9
    f=sim.f/1e9    
    initial_pulse = getPower(testSpectrum0)*1e9
    final_pulse = getPower(testSpectrumFinal)*1e9

    
    trace_initial = {
        "x": f.tolist(),
        "y": initial_pulse.tolist(),
        "mode": "lines",
        "name": "Initial spectrum",
    }

    trace_final = {
        "x": f.tolist(),
        "y": final_pulse.tolist(),
        "mode": "lines",
        "name": "Final spectrum",
    }
    layout = {
        "title": "Initial spectrum and final spectrum",
        "xaxis": { "title": 'Freq. [GHz]'},
        "yaxis": { "title": 'PSD [W/GHz]'},
        "height": 700,
        
    }
    data = [trace_initial, trace_final]
    plot_data = {"data": data, "layout": layout}

    # Serialize the plot data to JSON
    plot_json = json.dumps(plot_data)

    return plot_json 
    #return json.dumps(data)



def func_plotSpectrumMatrix2D_fig(matrix,fiber,sim,  dB_cutoff):

    f = sim.f/1e9 
    z = fiber.zlocs 
    F, Z = np.meshgrid(f, z)
    Pf=getPower(matrix  )/np.max(getPower(matrix))
    Pf[Pf<1e-100]=1e-100
    Pf = 10*np.log10(Pf)
    Pf[Pf<dB_cutoff]=dB_cutoff
    data = [go.Contour(x=f, y=z, z=Pf, colorscale='Viridis')]
    
    layout = go.Layout(
        title="Spectrum Evolution (dB scale) 2D",
        xaxis=dict(title='Freq. [GHz]'),
        yaxis=dict(title='Distance [m]'),
        height=700,
    )
    
    fig = go.Figure(data=data, layout=layout)
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def func_plotPulseChirp2D_fig(matrix, fiber, sim_config):
    t = sim_config.t[int(sim_config.number_of_points/2-nrange):int(sim_config.number_of_points/2+nrange)] * 1e12
    z = fiber.zlocs
    T, Z = np.meshgrid(t, z)
    Cmatrix = np.ones((len(z), len(t))) * 1.0

    for i in range(fiber.ntraces):
        Cmatrix[i, :] = getChirp(t/1e12, matrix[i, int(sim_config.number_of_points/2-nrange):int(sim_config.number_of_points/2+nrange)]) / 1e9

    Cmatrix[Cmatrix < -20] = -20
    Cmatrix[Cmatrix > 20] = 20

    data = [{
        "type": "contour",
        "x": t.tolist(),
        "y": z.tolist(),
        "z": Cmatrix.tolist(),
        "colorbar": {"title": "Chirp [GHz]"},
    }]

    layout = {
        "title": "Pulse Chirp Evolution 2D",
        "xaxis": {"title": "Time [ps]"},
        "yaxis": {"title": "Distance [m]"},
        "height":700,
        
    }

    return json.dumps({"data": data, "layout": layout})

def func_plotPulseMatrix2D_fig(matrix, fiber, sim,  dB_cutoff):
    t = sim.t * 1e12
    z = fiber.zlocs
    T, Z = np.meshgrid(t, z)
    P = getPower(matrix) / np.max(getPower(matrix))
    P[P < 1e-100] = 1e-100
    P = 10 * np.log10(P)
    P[P < dB_cutoff] = dB_cutoff

    data = [go.Contour(x=t, y=z, z=P, colorscale='Viridis')]
    
    layout = go.Layout(
        title="Pulse Evolution (dB scale) 2D",
        xaxis=dict(title='Time [ps]'),
        yaxis=dict(title='Distance [m]'),
        height=700,
    )
    colorbar_title = "Power (dB)"
    contour = data[0]
    contour['colorbar'] = dict(title=colorbar_title)
    fig = go.Figure(data=data, layout=layout)
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)



def func_plotPulseMatrix3D_fig(matrix, fiber, sim, dB_cutoff):
    #t = sim.t[int(sim.number_of_points/2 - nrange):int(sim.number_of_points/2 + nrange)] * 1e12
    t = sim.t* 1e12
    z = fiber.zlocs
    T_surf, Z_surf = np.meshgrid(t, z)
    #P_surf = getPower(matrix[:, int(sim.number_of_points/2 - nrange):int(sim.number_of_points/2 + nrange)]) / np.max(getPower(matrix[:, int(sim.number_of_points/2 - nrange):int(sim.number_of_points/2 + nrange)]))
    P_surf = getPower(matrix)
    P_surf[P_surf < 1e-100] = 1e-100
    P_surf = 10 * np.log10(P_surf)
    P_surf[P_surf < dB_cutoff] = dB_cutoff

    data = [go.Surface(z=P_surf, x=t, y=z)]
    layout = go.Layout(
        title="Pulse Evolution (dB scale) 3D",
        scene=dict(
            
            xaxis=dict(title='Time [ps]'),
            yaxis=dict(title='Distance [m]'),
            zaxis=dict(title='Power (dB)'),
        ),
        height=700,
    )

    fig = go.Figure(data=data, layout=layout)
    plot_div = opy.plot(fig, auto_open=False, output_type='div')

    return plot_div

def func_plotSpectrumMatrix3D_fig(matrix,fiber,sim, dB_cutoff):
    #Plot pulse evolution in 3D
    fig, ax = plt.subplots(1,1, figsize=(10,7),subplot_kw={"projection": "3d"})
    plt.title("Spectrum Evolution (dB scale)")

    f = sim.f/1e9 
    z = fiber.zlocs 
    F_surf, Z_surf = np.meshgrid(f, z)
    P_surf=getPower(matrix )/np.max(getPower(matrix))
    P_surf[P_surf<1e-100]=1e-100
    P_surf = 10*np.log10(P_surf)
    P_surf[P_surf<dB_cutoff]=dB_cutoff
    # Plot the surface.
    data = [go.Surface(z=P_surf, x=f, y=z)]
    layout = go.Layout(
        title="Spectrum Evolution (dB scale) 3D",
        scene=dict(
            
            xaxis=dict(title='Freq. [GHz]'),
            yaxis=dict(title='Distance [m]'),
            #zaxis=dict(title='Power (dB)'),
        ),
        height=700
    )

    fig = go.Figure(data=data, layout=layout)
    plot_div = opy.plot(fig, auto_open=False, output_type='div')

    return plot_div
    


"""       
elif "plot2d" in request.POST:
                dB_cutoff = -30
                z = np.linspace(0, Length, nsteps)
                T, Z = np.meshgrid(t, z)
                P = getPower(pulseMatrix) / np.max(getPower(pulseMatrix))
                P[P < 1e-100] = 1e-100
                P = 10 * np.log10(P)
                P[P < dB_cutoff] = dB_cutoff

                data = [go.Contour(x=t, y=z, z=P, colorscale='Viridis')]
                
                layout = go.Layout(
                    title="Pulse Evolution (dB scale) 2D",
                    xaxis=dict(title='Time [ps]'),
                    yaxis=dict(title='Distance [m]'),
                    height=700,
                )
                colorbar_title = "Power (dB)"
                contour = data[0]
                contour['colorbar'] = dict(title=colorbar_title)
                fig = go.Figure(data=data, layout=layout)
                plotPulseMatrix2D_fig= json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

                #t = sim_config.t[int(sim_config.number_of_points/2-nrange):int(sim_config.number_of_points/2+nrange)] * 1e12
                #z = fiber.zlocs
                T, Z = np.meshgrid(t, z)
                Cmatrix = np.ones((len(z), len(t))) * 1.0

                for i in range(nsteps):
                    Cmatrix[i, :] = getChirp(t/1e12, pulseMatrix[i, :]) / 1e9

                Cmatrix[Cmatrix < -20] = -20
                Cmatrix[Cmatrix > 20] = 20

                data = [{
                    "type": "contour",
                    "x": t.tolist(),
                    "y": z.tolist(),
                    "z": Cmatrix.tolist(),
                    "colorbar": {"title": "Chirp [GHz]"},
                }]

                layout = {
                    "title": "Pulse Chirp Evolution 2D",
                    "xaxis": {"title": "Time [ps]"},
                    "yaxis": {"title": "Distance [m]"},
                    "height":700,
                    
                }

                plotPulseChirp2D_fig = json.dumps({"data": data, "layout": layout})

                context = {"form":form, "form1":form1,"form3":form3,
                     
                           "plotPulseMatrix2D_fig":plotPulseMatrix2D_fig,
                           'plotPulseChirp2D_fig':plotPulseChirp2D_fig,
                           }


            elif "plotFirstAndLastSpectrum" in request.POST:
                sim_config=SIM_config(N,dt)
                fiber=Fiber_config(nsteps, Length, gamma, beta2, alpha_dB_per_m)
                pulseMatrix, spectrumMatrix = SSFM(fiber,sim_config,testPulse) 
                plotFirstAndLastSpectrum_fig = func_plotFirstAndLastSpectrum_fig(testSpectrum0,testSpectrumFinal,fiber,sim_config)
                context = {'plotFirstAndLastSpectrum_fig':plotFirstAndLastSpectrum_fig,"form":form, "form1":form1,"form3":form3}
                
            elif "plotFirstAndLastChirp" in request.POST:
                plotFirstAndLastChirp_fig = func_plotFirstAndLastChirp_fig(testPulse0,testPulseFinal,fiber,sim_config)
                context = {"form":form, "form1":form1,"form3":form3,"plotFirstAndLastChirp_fig":plotFirstAndLastChirp_fig}

            elif "plotPulseMatrix2D" in request.POST:
                sim_config=SIM_config(N,dt)
                fiber=Fiber_config(nsteps, Length, gamma, beta2, alpha_dB_per_m)
                pulseMatrix, spectrumMatrix = SSFM(fiber,sim_config,testPulse) 
                plotPulseMatrix2D_fig = func_plotPulseMatrix2D_fig(pulseMatrix, fiber, sim_config, -30)
                context = {"form":form, "form1":form1,"form3":form3,"plotPulseMatrix2D_fig":plotPulseMatrix2D_fig}

            elif "plotPulseChirp2D" in request.POST:
                plotPulseChirp2D_fig = func_plotPulseChirp2D_fig(pulseMatrix,fiber,sim_config)
                context = {'plotPulseChirp2D_fig':plotPulseChirp2D_fig,"form":form, "form1":form1,"form3":form3}

            elif "plotSpectrumMatrix2D" in request.POST:
                plotSpectrumMatrix2D_fig = func_plotSpectrumMatrix2D_fig(spectrumMatrix,fiber,sim_config, -30)
                context = {'plotSpectrumMatrix2D_fig':plotSpectrumMatrix2D_fig,"form":form, "form1":form1,"form3":form3}

            elif "plotPulseMatrix3D" in request.POST:
                plotPulseMatrix3D_fig = func_plotPulseMatrix3D_fig(pulseMatrix, fiber, sim_config, -30)
                context = {"form":form, "form1":form1,"form3":form3,"plotPulseMatrix3D_fig":plotPulseMatrix3D_fig}
            elif "plotSpectrumMatrix3D" in request.POST:
                plotSpectrumMatrix3D_fig = func_plotSpectrumMatrix3D_fig(spectrumMatrix, fiber, sim_config,  -30)
                context = {"form":form, "form1":form1,"form3":form3,"plotSpectrumMatrix3D_fig":plotSpectrumMatrix3D_fig}
            
if "pulse" in request.POST  :
                psd = getPower(testPulse)
                layout = go.Layout(
                    title="Initial Pulse",
                    xaxis={"title":"Time [ps]","range": [-duration*2*1e12,duration*2*1e12]},
                    yaxis={"title":"Power [W]"},
                    height=600,
                )
                trace = go.Scatter(x=t* 1e12, y=psd, mode="lines", name="testPulse")
                fig = go.Figure(data=[trace], layout=layout)
                pulse = pio.to_json(fig)

                context = {"form":form, "form1":form1,"form3":form3,"pulse":pulse}

            elif "chirp_fig" in request.POST  :
                chirp_fig = getChirp(t,testPulse)
                layout = go.Layout(
                    title="Chirp of Initial Pulse ",
                    xaxis={"title":"Time [ps]","range": [-duration*5*1e12,duration*5*1e12]},
                    yaxis={"title":"Chirp [Ghz]"},
                    height=600,
                )
                trace = go.Scatter(x=t*1e12, y=chirp_fig/1e9, mode="lines", name="Chirp")
                fig = go.Figure(data=[trace], layout=layout)
                chirp_figure = pio.to_json(fig)

                context = {"form":form, "form1":form1,"form3":form3,"chirp_figure":chirp_figure}

            elif "spectrum" in request.POST  :
                f=getFreqRangeFromTime(t)
                testSpectrum=getSpectrumFromPulse(t,testPulse)
                
                sim_config=SIM_config(N,dt)
                psd = getPower(testSpectrum)
                layout = go.Layout(
                    title="Spectrum of Initial Pulse",
                    xaxis={"title":"Freq. [THz]","range": [-1/duration*2/1e12,1/duration*2/1e12]},
                    yaxis={"title":"PSD [W/THz]"},
                    height=600,
                )
                trace = go.Scatter(x=f/1e12, y=psd, mode="lines", name="Spectrum of testPulse")
                fig = go.Figure(data=[trace], layout=layout)
                plotly_json = pio.to_json(fig)

                context = {"form":form, "form1":form1,"form3":form3,"plotly_json":plotly_json}
             """