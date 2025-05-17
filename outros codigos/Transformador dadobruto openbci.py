import mne
import pandas as pd
import numpy as np
import os

def processar_edf_para_openbci(diretorio_edf, filtragem, canais, algarismos_significativos=5, tarefa=""):
    """
    Processa um arquivo EDF para o formato CSV compatível com OpenBCI.

    Parâmetros:
    - diretorio_edf: Caminho do arquivo EDF.
    - filtragem: Tupla (low, high) para aplicação de filtro passa-banda.
    - canais: Lista de canais a serem selecionados. Se vazio, pega todos os eletrodos do EDF.
    - algarismos_significativos: Número de casas decimais para arredondamento.
    - tarefa: String "T1", "T2" ou "" (vazio) para selecionar tipos específicos de épocas.

    Retorna:
    - Caminho do arquivo CSV gerado.
    """

    # Carregar arquivo EDF
    raw = mne.io.read_raw_edf(diretorio_edf, preload=True)

    # Aplicar filtro personalizado
    if filtragem:
        raw.filter(filtragem[0], filtragem[1])

    # Selecionar canais
    if not canais:  
        canais = raw.ch_names
    else:
        raw.pick(canais)

    # Selecionar épocas caso tenha tarefa definida
    if tarefa:
        anotacoes = raw.annotations
        eventos, event_dict = mne.events_from_annotations(raw)
        if tarefa in event_dict:
            epochs = mne.Epochs(raw, eventos, event_id=event_dict[tarefa], tmin=0.5, tmax=4, baseline=None, preload=True)
            dados = epochs.get_data()  # Dados no formato (n_epochs, n_channels, n_times)
            dados = np.concatenate(dados, axis=-1)  # Concatena épocas ao longo do tempo
        else:
            raise ValueError(f"Tarefa '{tarefa}' não encontrada nas anotações.")
    else:
        dados, _ = raw.get_data(), raw.times  # Dados no formato (n_channels, n_times)

    # Converter para microV e arredondar
    dados_volts = dados * 1e6  # Converter para microV
    dados_arredondados = np.round(dados_volts, algarismos_significativos)

    # Criar DataFrame
    df = pd.DataFrame(dados_arredondados.T, 
                     columns=[f'EXG Channel {i}' for i in range(len(raw.ch_names))])

    # Adicionar colunas complementares zeradas
    estrutura_colunas = [
        'Sample Index',
        *[f'EXG Channel {i}' for i in range(len(raw.ch_names))],
        'Accel Channel 0', 'Accel Channel 1', 'Accel Channel 2',
        *['Other']*7,
        'Analog Channel 0', 'Analog Channel 1', 'Analog Channel 2',
        'Timestamp', 'Other', 'Timestamp (Formatted)'
    ]

    df.insert(0, 'Sample Index', range(1, len(df)+1))
    for col in estrutura_colunas[len(raw.ch_names)+1:]:
        df[col] = 0

    df = df[estrutura_colunas]  # Ordenar colunas

    # Gerar nome do arquivo
    nome_base = os.path.splitext(os.path.basename(diretorio_edf))[0]
    epoca_suffix = f"_{tarefa}" if tarefa else ""
    caminho_saida = os.path.join(
        os.path.dirname(diretorio_edf),
        f"{nome_base}{epoca_suffix}_csv_openbci.csv"
    )

    if True:
        caminho_saida = r'c:\Users\Enenon\Documents\OpenBCI_GUI\dado.csv'

    # Salvar CSV formatado
    df.to_csv(caminho_saida, index=False, float_format=f'%.{algarismos_significativos}g')

    # Adicionar cabeçalho personalizado
    with open(caminho_saida, 'r+') as f:
        conteudo = f.read()
        f.seek(0, 0)
        f.write(
            f"%OpenBCI Raw EXG Data\n"
            f"%Number of channels = {len(raw.ch_names)}\n"
            f"%Sample Rate = {int(raw.info['sfreq'])} Hz\n"
            "%Board = OpenBCI_GUI$BoardCytonSerialDaisy\n"
        )
        f.write(conteudo)

    print(f"Arquivo gerado: {caminho_saida}")
    return caminho_saida

# Exemplo de uso:
arquivo_edf = r"f:\eeg-motor-movementimagery-dataset-1.0.0\files\S022\S022R04.edf"
canais_desejados = ['C3..', 'C4..', 'Fp1.', 'Fp2.', 'F7..', 'F3..', 'F4..', 'F8..','T7..', 'T8..', 'P7..', 'P3..', 'P4..', 'P8..', 'O1..', 'O2..']
filtragem = (0.5, 50)
algarismos_significativos = 4
tarefa = "T2"  

processar_edf_para_openbci(arquivo_edf, filtragem, canais_desejados, algarismos_significativos, tarefa)