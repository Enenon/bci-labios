#%%
import mne
import numpy as np
import os

canais = ['C3..', 'C4..', 'Fp1.', 'Fp2.', 'F7..', 'F3..', 'F4..', 'F8..','T7..', 'T8..', 'P7..', 'P3..', 'P4..', 'P8..', 'O1..', 'O2..']
def pegar_acao(edf,acao):
    # acao = T0, T1 ou T2
    var = mne.events_from_annotations(edf)
    indice = mne.events_from_annotations(edf)[1][acao]
    data = mne.Epochs(raw=edf,events=mne.events_from_annotations(edf)[0],event_id=indice,tmin=0.0,tmax=4,baseline=(None,None)).get_data()
    #junta tudo
    try:
        data = np.concatenate(data,axis=-1)
    except:
        raise ValueError(f'Dado: {data}')
    return data

dataset_dir = r'c:\Users\LaBios - BCI\Documents\eeg-motor-movementimagery-dataset-1.0.0\files'

acoes = ['T0','T1','T2']

T0l = []
T1l = []
T2l = []

for ind in os.listdir(dataset_dir):
    if not os.path.isdir(os.path.join(dataset_dir,ind)):
        continue
    for arq_dir in os.listdir(os.path.join(dataset_dir,ind)):
        if not True in [arq_dir.endswith(i) for i in ['R04.edf','R08.edf','R12.edf']]:
            continue
        data_dir = os.path.join(dataset_dir,ind,arq_dir)
        edf = mne.io.read_raw_edf(data_dir)
        edf.pick(canais)
        for task in acoes:
            data = pegar_acao(edf,task)
            match task:
                case 'T0': T0l.append(data)
                case 'T1': T1l.append(data)
                case 'T2': T2l.append(data)

saida = r'../bci'

T0l = np.concatenate(T0l,axis=-1)
T1l = np.concatenate(T1l,axis=-1)
T2l = np.concatenate(T2l,axis=-1)


#%%
np.savez(saida+r'\T0l.npz',T0l)
np.savez(saida+r'\T1l.npz',T1l)
np.savez(saida+r'\T2l.npz',T2l)

print('Fim!')