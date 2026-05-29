import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from keras.models import load_model
from keras.optimizers import Adam
import numpy as np

local_modelo = r'c:\Users\Enenon\Documents\GitHub\aaaaa.h5'
local_dados = r'c:\Users\Enenon\Documents\GitHub\teste\output_0'

modelo = load_model(local_modelo)
modelo.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

lista_dados = os.listdir(local_dados)
saidas = []
for arquivo in lista_dados:
    caminho_arquivo = os.path.join(local_dados, arquivo)
    dados = np.loadtxt(caminho_arquivo)
    dados = np.expand_dims(dados, axis=0)
    predicao = modelo(dados)[0].numpy()
    saidas.append(predicao)
print(*[str(i[0]) + '\n' for i in saidas])