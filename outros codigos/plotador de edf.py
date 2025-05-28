#%%
import mne
import numpy as np

edf = mne.io.read_raw_edf(r'c:\Users\LaBios - BCI\Documents\eeg-motor-movementimagery-dataset-1.0.0\files\S001\S001R04.edf')

canais = ['C3..', 'C4..', 'Fp1.', 'Fp2.', 'F7..', 'F3..', 'F4..', 'F8..','T7..', 'T8..', 'P7..', 'P3..', 'P4..', 'P8..', 'O1..', 'O2..']
# %%
edf.pick(canais)

atividade = 'T0'

def pegar_acao(edf,acao):
    # acao = T0, T1 ou T2
    indice = mne.events_from_annotations(edf)[1][acao]
    data = mne.Epochs(edf,mne.events_from_annotations(edf)[0],indice).get_data()
    #junta tudo
    try:
        data = np.concatenate(data,axis=-1)
    except:
        raise ValueError(f'Dado: {data}')
    return data

#print(pegar_acao(edf,atividade))

edf.plot()

from matplotlib.pyplot import show
show()