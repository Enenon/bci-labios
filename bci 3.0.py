#%%
import numpy as np
import keyboard  # para detectar ESC
import time
from pylsl import StreamInlet, resolve_stream
from time import sleep
import matplotlib.pyplot as plt
from keras.models import load_model
from tensorflow.keras.optimizers import Adam

# Carrega o modelo (entrada esperada: batch x 721 x 16)
model = load_model(r"C:\Users\Enenon\Documents\GitHub\bci-labios\modelos\modelo rede c.h5")
model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# ---------- CONFIGURAÇÕES ----------

tipo_dado = 2 # 0 = dado bruto, 1 = rede, 2 = rede com caracteristicas

fases = ['T1','T2']  # ← 'T1' ou 'T2'
time_fase1 = 76.5
time_fase2 = 81.0
time_fase3 = 67.5
limiar = 0.499
# Sequência de verdadeiros rótulos para Fase 3 (preencher manualmente)
truth_sequence3 = ['T1', 'T0', 'T2', 'T1', 'T0', 'T2', 'T1', 'T0', 'T2', 'T1', 'T0', 'T2', 'T1', 'T0', 'T2']  # exemplo: ['T1','T2','T0', ...]
cm_predicoes = np.array([[0 for j in range(3)] for i in range(3)])

# Classe de controle de atualização de gráfico
class Sistema:
    def __init__(self):
        self.dt = 4.506  # intervalo de plotagem (s)
sistema = Sistema()

# Auxiliares
label2num = {'T1': 1, 'T2': 2, 'T0': 0}

def normalize_sample(input_sample):
    inp = np.array(input_sample)
    local_max, local_min = inp.max(), inp.min()
    return (inp - local_min) / (local_max - local_min + 1e-8)

def predict(model, input_sample):
    arr = normalize_sample(input_sample)
    arr = np.expand_dims(arr, axis=0)
    return model(arr, training=False)

from numpy import zeros

def S(matriz):
    som = [sum(i) for i in matriz]
    return som

def C(matriz):
    cs = zeros(len(matriz))
    for i in range(len(matriz)):
        x = 0
        y = 0
        z = 0
        for j in range(len(matriz)):
            y = y + matriz[i][j]
            z = z + matriz[i][j]**2
            for k in range(len(matriz)):
                x = x + matriz[i][j]*matriz[j][k]*matriz[k][i]
        cs[i] = x/(y**2 + z)
    return cs

def I(matriz):
    iss = [[1/i if i != 0 else 0 for i in j] for j in matriz]
    ii = [sum(j)/(len(iss)-1) for j in iss]
    return ii

def corrPearson(matriz):
    matriz = np.transpose(matriz)
    m = np.zeros((len(matriz),len(matriz)))
    meanM = [np.mean(matriz[i,:]) for i in range(len(matriz))]
    for x in range(len(m)):
        for y in range(len(m)):
            xm, ym = matriz[x,:], matriz[y,:]
            xx = xm - meanM[x]
            yy = ym - meanM[y]
            m[x,y] = round(sum((xx)*(yy))/(np.sqrt(sum(xx*xx)) * np.sqrt(sum(yy*yy))),3)
    return m

def randomThreshold(corr,matriz):
    smatriz = matriz.copy() # por causa da falta desse .copy() eu tava tendo meio mundo de dor de cabeça
    for i in range(len(smatriz[0])):
        smatriz[:,i] = np.random.permutation(smatriz[:,i])
    return corr(smatriz)

def rede(corr,threshold,matriz,intervalos=100):
    REAl = []
    for n in range(int(len(matriz)/intervalos)):
        m = matriz[n*intervalos:(n+1)*intervalos]
        rnd = threshold(corr,m)
        # tirando os 1 da diagonal
        for i in range(len(rnd)): rnd[i,i] = 0
        maxim = rnd.max()
        rmatriz = np.array([[i if i > maxim else 0 for i in j] for j in corr(matriz)])
        REAl.append(rmatriz)
    REDEM = sum(np.array(REAl))/int(len(matriz)/intervalos)
    return REDEM

def redep(matriz): # essa função cria a REA da serie
    return rede(corrPearson,randomThreshold,matriz)

def redetot(matriz): #  essa função cria a REA junto com as características strength, cloyster coefficient e charateristic path length
    rmatriz = redep(matriz)
    nm = np.concatenate((rmatriz,np.array([S(rmatriz)]),np.array([C(rmatriz)]),np.array([I(rmatriz)])),axis=0)
    return nm

def brut(matriz):
    return matriz

conversao = [brut,redep,redetot][tipo_dado]


# Inicializa stream
if tipo_dado == 0:
    epochsize = model.input_shape[1]
else:
    epochsize = 721

print(f"Aguardando stream EEG... input shape: {model.input_shape}")
streams = resolve_stream('type', 'EEG')
inlet = StreamInlet(streams[0])
sleep(1)
print("Stream EEG encontrada!")

# Fases e métricas
tfase = 1
time_start = None
phase1_total = phase1_correct = 0
phase2_total = phase2_correct = 0
phase3_total = phase3_correct = 0
phases_total = [0 for i in range(3)]
phases_correct = [0 for i in range(3)]
idx3 = 0

# Para plotagem dinâmica
pred_values = []
pred_colors = []

#plt.ion()
fig, ax = plt.subplots()
last_plot = time.time()

print('Pressione ESC para encerrar.')
current_data = []
started = False
while not keyboard.is_pressed('Esc'):
    # Inicia ao detectar dados não-zero
    if not started:
        chunk, _ = inlet.pull_chunk()
        if chunk and np.any(np.array(chunk) != 0):
            started = True
            time_start = time.time()
            print("Início da Fase 1: dados válidos detectados.")
        else:
            continue

    # Atualiza fase
    elapsed = time.time() - time_start
    if tfase == 1 and elapsed > time_fase1:
        tfase = 2; time_start = time.time(); print("-> Fase 2 iniciada")
    elif tfase == 2 and elapsed > time_fase2:
        tfase = 3; time_start = time.time(); print("-> Fase 3 iniciada")
    elif tfase == 3 and elapsed > time_fase3:
        print("-> Todas as fases concluídas.")
        break

    # Puxa chunk de amostras
    chunk, _ = inlet.pull_chunk()
    if not chunk:
        continue

    for sample in chunk:
        # janela deslizante
        current_data.append(sample)
        if len(current_data) > epochsize:
            current_data.pop(0)
        if len(current_data) < epochsize:
            continue

        # Predição
        pred = predict(model, conversao(np.array(current_data))).numpy()[0][0]
        if pred < limiar:
            label = 'T1'
        elif pred > 1 - limiar:
            label = 'T2'
        else:
            label = 'T0'

        # Avalia o certo
        if tfase in (1,2):
            expected = fases[tfase-1]
            phases_total[tfase-1] += 1
            correct = label==expected
            cm_predicoes[label2num[expected] ,int(label[1])] += 1
            if correct:phases_correct[tfase-1] += 1
        else:
            expected = truth_sequence3[idx3] if idx3 < len(truth_sequence3) else None
            idx3 += 1
            phases_total[2] += 1
            correct = (label == expected)
            cm_predicoes[label2num[expected] ,int(label[1])] += 1
            if correct: phases_correct[2] += 1

        # Salva para plot
        pred_values.append(label2num[label])
        pred_colors.append('g' if correct else 'r')

        print(f"[Fase {tfase}] Prev: {label} | Exp: {expected} | Prob: {pred:.3f} | Acerto: {correct}")

        # Transfer Learning se acerto nas fases 1 ou 2
        if tfase in [1,2] and correct:
            bx = np.array([normalize_sample(current_data)])
            by = np.array([label2num[label]]).reshape(-1,1)
            #print("=== Iniciando transfer learning (1 época) ===")
            #model.fit(bx, by, epochs=1, verbose=1)
            #print("=== Transfer learning concluído ===")

        current_data = []

        # Plot dinâmico a cada sistema.dt
        if time.time() - last_plot >= sistema.dt:
            last_plot = time.time()
            ax.cla()
            # plota últimos 100 pontos com cores
            for i, (y, c) in enumerate(zip(pred_values[-100:], pred_colors[-100:])):
                ax.scatter(i, y, c=c, marker='o')
            ax.set_title(f"Live: Fase {tfase} | Label Fase1: {fases[0]}")
            ax.set_ylabel('Label numérico')
            #plt.draw(); plt.pause(0.001)

print("\n=== Acurácias finais ===")
if phase1_total:
    print(f"Fase 1: {phase1_correct}/{phase1_total} = {phase1_correct/phase1_total*100:.2f}%")
if phase2_total:
    print(f"Fase 2: {phase2_correct}/{phase2_total} = {phase2_correct/phase2_total*100:.2f}%")
if phase3_total:
    print(f"Fase 3: {phase3_correct}/{phase3_total} = {phase3_correct/phase3_total*100:.2f}%")

plt.ioff()

#%% plotar matriz confusão

from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

#Make predictions
#Convert prediction probabilities into integers
y_preds = []
for i in range(len(cm_predicoes)):
    for j in range(len(cm_predicoes[i])):
        for k in range(cm_predicoes[i,j]):
            y_preds.append(f'T{j}')
#y_true = [fases[0] for i in range(phases_total[0])] + [fases[1] for i in range(phases_total[1])] + truth_sequence3[:sum(cm_predicoes[2])]
y_true = []
for i in range(3):
    y_true += [f'T{i}' for j in range(sum(cm_predicoes[i]))]

#Confusion matrix
cm=confusion_matrix(y_true,y_preds)
#
label_names = ["T0","T1","T2"]
disp=ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=label_names)
fig, ax = plt.subplots(figsize=(5,5))
disp.plot(ax=ax)
plt.show()

# %%
