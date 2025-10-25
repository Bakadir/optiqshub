from django.shortcuts import render
import numpy as np, os, json

def laser_analysis_view(request):
    """
    Charge les données (.txt) pour le jeu choisi (AA, B ou C)
    et affiche les graphes Plotly.
    """
    base = os.path.join('static', 'dm')

    # Choisir le jeu via paramètre GET ?set=AA/B/C
    dataset = request.GET.get('set', 'AA')

    y1 = np.loadtxt(os.path.join(base, f'y1_{dataset}.txt'))
    y2 = np.loadtxt(os.path.join(base, f'y2_{dataset}.txt'))
    y3 = np.loadtxt(os.path.join(base, f'y3_{dataset}.txt'))
    y4 = np.loadtxt(os.path.join(base, f'y4_{dataset}.txt'))

    dt = 20e-12
    Nrt = 5217
    alpha = 2 / 3

    # Champ complexe reconstitué
    S_c = alpha * (y1 + y2 * np.exp(-1j*2*np.pi/3) + y3 * np.exp(1j*2*np.pi/3))
    phi = np.unwrap(np.angle(S_c))
    f_inst = np.gradient(phi, dt) / (2*np.pi)
    E = np.sqrt(np.maximum(y4,0)) * np.exp(1j*phi)

    # Roundtrips
    K = len(E)//Nrt
    E_rt = E[:K*Nrt].reshape(K, Nrt)
    I_rt = y4[:K*Nrt].reshape(K, Nrt)

    # g1
    g1 = []
    for m in range(K):
        num=dena=denb=0
        for k in range(K-m):
            num += np.mean(np.conjugate(E_rt[k])*E_rt[k+m])
            dena += np.mean(np.abs(E_rt[k])**2)
            denb += np.mean(np.abs(E_rt[k+m])**2)
        num /= max(K-m,1)
        dena /= max(K-m,1)
        denb /= max(K-m,1)
        g1.append(num/np.sqrt(dena*denb))
    g1=np.array(g1)

    # Spectre
    Nfft=2048
    spec=np.abs(np.fft.fftshift(np.fft.fft(E_rt,n=Nfft,axis=1),axes=1))**2
    spec/=spec.max()
    freqs=np.fft.fftshift(np.fft.fftfreq(Nfft,dt))

    # g2
    g2=[]
    for m in range(K):
        num=dena=denb=0
        for k in range(K-m):
            num += np.mean(I_rt[k]*I_rt[k+m])
            dena += np.mean(I_rt[k])
            denb += np.mean(I_rt[k+m])
        num/=max(K-m,1)
        dena/=max(K-m,1)
        denb/=max(K-m,1)
        g2.append(num/(dena*denb))
    g2=np.array(g2)

    g2_0 = [np.mean(I_rt[k]**2)/np.mean(I_rt[k])**2 for k in range(K)]

    # Contexte
    context = {
        "dataset": dataset,
        "m_vals": json.dumps(list(range(K))),
        "g1_abs": json.dumps(np.abs(g1).tolist()),
        "g1_real": json.dumps(np.real(g1).tolist()),
        "g2_vals": json.dumps(g2.tolist()),
        "g2_0_vals": json.dumps(g2_0),
        "spectrum": json.dumps(spec.tolist()),
        "freq": json.dumps((freqs/1e9).tolist()),
        "roundtrips": json.dumps(list(range(K))),
    }
    return render(request, "dm/laser_analysis.html", context)
