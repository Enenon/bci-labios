#%%
import keyboard
import numpy as np
from random import choice
#from tensorflow import lite
from keras.models import load_model
#from tensorflow.keras.models import load_model
import time
from pylsl import StreamInlet, resolve_stream, local_clock
from time import sleep
from sys import exit

class Sistema():
    def __init__(self):
        pass

    t0 = time.time()
    dt = 0.3

tipo_dado = {'rede':0,'bruto':1}[['rede','bruto'][1]]

min, max = -1238.0123, 1268.5555

norm_dinamica = False # dita se vai utilizar os maximos e minimos de cada matriz ou um maximo e minimo global pra normalizar

model = load_model(r'C:\Users\Enenon\Documents\Projetos Programacao\IC\data\modelo\melhor_modelo_fold_5_acc_0.8880 mudado.h5')

limiar = 0.4 # limiar tem que estar entre 0 e 0.5

task = 'T2' # é a tarefa que está sendo testada

model.summary()


def predict(model,input):
    #return model(np.array([np.transpose(input)]), training=False) # olha o armengue q tive que fazer pra funcionar
    global max, min
    if norm_dinamica:
        max = np.array(input).max()
        min = np.array(input).min()
    #print(max,round(min,3))
    m = (np.array([input])-min)/(max-min) # isso aqui é a normalização da matriz
    print(m.max(),m.min())
    return model(m, training=False)
    return model.predict(np.array([np.transpose(input)]))
 
def predict2(model,matriz):
    return choice(['Esquerda','Direita','Parado'])

def corrPearson(matriz):
    matriz = np.transpose(matriz)
    if len(matriz[0]) > len(matriz[1]): print('Aviso! Talvez a matriz para correlação precise ser transposta!')
    m = np.zeros((len(matriz),len(matriz)))
    meanM = [np.mean(matriz[i,:]) for i in range(len(matriz))]
    for x in range(len(m)):
        for y in range(len(m)):
            xm, ym = matriz[x,:], matriz[y,:]
            xx = xm - meanM[x]
            yy = ym - meanM[y]
            m[x,y] = sum((xx)*(yy))/(np.sqrt(sum(xx*xx)) * np.sqrt(sum(yy*yy)))
    return m

epochsize = model.input_shape[1] # tá 10 pontos, tem que converter pra segundos pra ficar mais claro

print('Modelo carregado!')
print('Input shape:',model.input_shape)
print('Output shape:',model.output_shape)

if tipo_dado == 0 and model.input_shape[1] != model.input_shape[2]: raise ValueError(f'Erro! O seu modelo não foi feito para trabalhar com redes. Shape do modelo:{model.input_shape}, shape esperado: (None, {model.input_shape[2]}, {model.input_shape[2]})')

print("procurando por uma stream EEG...")
streams = resolve_stream('type', 'EEG')

inlet = StreamInlet(streams[0])

sleep(1)



sistema = Sistema()

start = time.time()
totalNumSamples = 0
validSamples = 0
numChunks = 0
print( "Testando Sampling Rates..." )
data = []
totdata = []

np.set_printoptions(precision=4, suppress=True)

print('Pronto!')


print('Intervalo de tempo:',Sistema().dt)
print(f'Aguarde {round(model.input_shape[1]/160,3)} segundos após começar o streaming.')

if False:
    while len(data) < 721:
        chunk, timestamp = inlet.pull_chunk()
        if chunk:
            for sample in chunk:
                if len(data) < 721: data.append(sample)
    print(np.array(data).shape)
    
    print(predict(model,data))

    print('Agora, iniciando a aquisição online...')




#%%
data = []
tgraf = []
redes = np.zeros((model.input_shape[1],model.input_shape[2]))
primeiro = True
l = []
while not keyboard.is_pressed('Esc'):
    # get chunks of samples
    chunk, timestamp = inlet.pull_chunk()
    if chunk:
        #print("\nNew chunk!")
        numChunks += 1
        # print( len(chunk) )
        totalNumSamples += len(chunk)
        # print(chunk)
        i = 0
        for ind, sample in enumerate(chunk): #não entendo direito o porque disso, mas parece que as séries temporais vêm do openbci como pacotes
            data.append(sample)
            if keyboard.is_pressed('g'):
                try:
                    print(f'Frequência: {1/(timestamp[ind]-tempo_a)}hz')
                except: pass
            if len(data) >= epochsize:
                if time.time() - sistema.t0 > sistema.dt:
                    sistema.t0 = time.time()
                    if len(data) == epochsize:
                        if primeiro: 
                            l.append(data)
                            l = np.array(l)
                            primeiro = False
                        if tipo_dado == 1: pred = predict(model,data).numpy()[0][0]
                        else:
                            rede_data = corrPearson(data)
                            pred = predict(model, rede_data).numpy()[0][0]
                            redes = redes + rede_data
                        if pred < limiar: let = 'E'
                        elif pred > 1 - limiar: let = 'D'
                        else: let = 'P'
                        tgraf.append(let)
                        print('Avaliação:', let, round(pred,3),len(data))
                #totdata += data
                data.pop(0)
            tempo_a = timestamp[ind]


#%%
print(len(data)+1)
#print(len(totdata))
print( "Number of Chunks and Samples == {} , {}".format(numChunks, totalNumSamples) )

if len(tgraf) == 0: exit('Não rodou o EEG')

import matplotlib.pyplot as plt

print('E:',round(tgraf.count('E')/len(tgraf),3))
print('D:',round(tgraf.count('D')/len(tgraf),3))
print('P:',round(tgraf.count('P')/len(tgraf),3))

task2ativi = {'T0':'Parado','T1':'Esquerda','T2':'Direita'}

plt.plot(tgraf,c='red')
plt.plot(tgraf,'.')
plt.show()

Dlist = [i for i in range(len(tgraf)) if tgraf[i] == 'D']
Elist = [i for i in range(len(tgraf)) if tgraf[i] == 'E']
Plist = [i for i in range(len(tgraf)) if tgraf[i] == 'P']

lintam = 80

dot = 's'

plt.plot(Dlist,['Previsto' for i in Dlist],dot,c='red',linewidth=lintam,label='Direita')
plt.plot(Elist,['Previsto' for i in Elist],dot,c='blue',linewidth=lintam,label='Esquerda')
plt.plot(Plist,['Previsto' for i in Plist],dot,c='green',linewidth=lintam,label='Parado')


corest = {'T0':'green','T1':'blue','T2':'red'}

if task: plt.plot([i for i in range(len(tgraf))],['Esperado' for i in tgraf],dot,c=corest[task],linewidth=lintam)

leg = plt.legend(['Direita','Esquerda','Parado'])

# tem acesso às linhas dentro da caixa da legenda
leg_lines = leg.get_lines()

# muda as propriedades de todas as linhas
plt.setp(leg_lines, linewidth=4)

plt.title(f'Limiar:{limiar}-{1-limiar}',loc='right')

plt.show()


#testLSLSamplingRate()
# %%
