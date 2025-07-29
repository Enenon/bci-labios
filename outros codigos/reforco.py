import os
import gc
import fnmatch
import numpy as np
import tensorflow as tf
import mne
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score

def processar_arquivos(diretorio_raiz, lista_epocas, tarefas, pessoas_escolhidas, limite_por_evento=None):
    canais_desejados = ['Fp1.', 'F7..', 'F3..', 'T7..', 'C3..', 'P7..', 'P3..', 'O1..',
                        'Fp2.', 'F4..', 'F8..', 'C4..', 'T8..', 'P4..', 'P8..', 'O2..']

    epocas_por_evento = {evento: [] for evento in lista_epocas}
    total_arquivos_eventos = {evento: 0 for evento in lista_epocas}

    pastas = [p for p in os.listdir(diretorio_raiz) if os.path.isdir(os.path.join(diretorio_raiz, p))]

    for pasta in pastas:
        try:
            numero = int(pasta[1:4])
            if numero in pessoas_escolhidas:
                caminho_pasta = os.path.join(diretorio_raiz, pasta)
                arquivos = []
                for tarefa in tarefas:
                    arquivos.extend(fnmatch.filter(os.listdir(caminho_pasta), tarefa))

                raws = []
                for arq in arquivos:
                    raw = mne.io.read_raw_edf(os.path.join(caminho_pasta, arq), preload=True)
                    raw.pick_channels(canais_desejados)
                    raws.append(raw)

                if raws:
                    raw_concatenado = mne.concatenate_raws(raws)
                    events, event_id = mne.events_from_annotations(raw_concatenado)
                    for evento in lista_epocas:
                        if evento in event_id:
                            epochs = mne.Epochs(raw_concatenado, events, event_id={evento: event_id[evento]},
                                                tmin=-0.5, tmax=4, baseline=(-0.5, 0))
                            epocas = epochs.get_data().astype(np.float32)
                            epocas_por_evento[evento].extend(epocas)
                            total_arquivos_eventos[evento] += len(epocas)
        except:
            continue

    if limite_por_evento is None:
        limite_por_evento = int(np.mean([q for q in total_arquivos_eventos.values() if q > 0]))

    for evento in epocas_por_evento:
        if len(epocas_por_evento[evento]) > limite_por_evento:
            np.random.shuffle(epocas_por_evento[evento])
            epocas_por_evento[evento] = epocas_por_evento[evento][:limite_por_evento]

    return epocas_por_evento

def separar_dados(epocas_por_evento):
    arrays = {}
    for evento, epocas in epocas_por_evento.items():
        arrays[evento] = np.array(epocas)

    data = np.concatenate([a for a in arrays.values() if a.size > 0])
    labels = []
    for i, (evento, array) in enumerate(arrays.items()):
        labels.extend([i] * len(array))

    x = np.nan_to_num(data)
    y = np.array(labels)
    x_min, x_max = np.min(x, axis=0), np.max(x, axis=0)
    x = (x - x_min) / (x_max - x_min + 1e-8)
    x = x.transpose(0, 2, 1)  # (amostras, tempo, canais)

    return x, y

def treinar_individual(modelo, x, y, epochs=5, batch_size=8):
    modelo.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    modelo.fit(x, y, epochs=epochs, batch_size=batch_size, verbose=0)
    return modelo

def avaliar(modelo, x, y):
    y_pred = modelo.predict(x, verbose=0)
    y_pred_classes = (y_pred > 0.5).astype(int)
    return accuracy_score(y, y_pred_classes)

# Caminhos
modelo_pre_treinado = load_model(r"c:\Users\Enenon\Documents\GitHub\bci-labios\modelos\melhor_modelo_fold_6_acc_0.9000.h5")
diretorio = r"c:\Users\Enenon\Downloads\eeg-motor-movementimagery-dataset-1.0.0\files"
eventos = ["T1", "T2"]
tarefas = ["*R04.edf", "*R08.edf", "*R12.edf"]

resultados = {}

for pessoa in range(85, 87):
    print(f"\n--- Pessoa S{pessoa:03d} ---")
    try:
        epocas = processar_arquivos(diretorio, eventos, tarefas, [pessoa], limite_por_evento=50)
        x, y = separar_dados(epocas)

        if x.shape[0] == 0 or len(np.unique(y)) < 2:
            print("Dados insuficientes.")
            resultados[pessoa] = None
            continue

        modelo_treinado = tf.keras.models.clone_model(modelo_pre_treinado)
        modelo_treinado.set_weights(modelo_pre_treinado.get_weights())

        modelo_treinado = treinar_individual(modelo_treinado, x, y, epochs=5)
        acc = avaliar(modelo_treinado, x, y)

        resultados[pessoa] = acc
        print(f"Acurácia após fine-tuning: {acc * 100:.2f}%")
        modelo_treinado.save(f'pessoa{pessoa}ac{round(acc,2)}.h5')
    except Exception as e:
        print(f"Erro em S{pessoa:03d}: {str(e)}")
        resultados[pessoa] = None

# Salvar resultados
if False:
    with open("resultados_transfer_learning.csv", "w") as f:
        f.write("Pessoa,Acuracia\n")
        for p, acc in resultados.items():
            linha = f"S{p:03d},{acc:.4f}\n" if acc is not None else f"S{p:03d},N/A\n"
            f.write(linha)

print("\nFinalizado! Resultados salvos em resultados_transfer_learning.csv")