import os

### LEIA ABAIXO ###

# com o objetivo de padronizar as matrizes, resolvi dar o trabalho de criar os índices antes de começar a leitura das matrizes.
# depois irei comparar com o processo de fazer os índices toda vez e ver se houve diferença
# portanto, note onde está a variável 'neur' agora
# isso significa q existe agora apenas um 'neur' universal, ao invés de 'neur' estar sendo criado toda vez q rodar o sepRea()

neur = {}
neur_feito = False

def pegDados(entrada):
    with open(entrada, 'r') as texto:
        texto = texto.read()
        dados = [i.split('\t') for i in texto.split('\n')][1:-1]
    return dados

def sepRea(caminho_entrada, caminho_saida):
    dados = pegDados(caminho_entrada)
    count = 0
    global neur_feito
    global neur
    if neur_feito == False: # essa parte faz com q neur seja feito apenas na primeira rodagem
        for i in dados:
            if i[0] not in neur:
                neur[i[0]] = count
                count += 1
            if i[1] not in neur:
                neur[i[1]] = count
                count += 1
        neur_feito = True
    count = len(neur)
    matriz = [[0 for i in range(count)] for i in range(count)]

    print(f"Processando: {caminho_entrada}")
    for dado in dados:
        matriz[neur[dado[0]]][neur[dado[1]]] = int(dado[2])
        matriz[neur[dado[1]]][neur[dado[0]]] = int(dado[2])

    # Lembrando que é uma matriz simétrica

    # Salve o resultado na pasta de saída
    with open(caminho_saida, 'w') as arquivo_saida:
        for linha in matriz:
            arquivo_saida.write('\t'.join(map(str, linha)) + '\n')

def processar_pasta(pasta_entrada, pasta_saida):

    for arquivo in os.listdir(pasta_entrada):
        if arquivo.endswith(".txt"):
            caminho_arquivo_entrada = os.path.join(pasta_entrada, arquivo)
            caminho_arquivo_saida = os.path.join(pasta_saida, arquivo)

            sepRea(caminho_arquivo_entrada, caminho_arquivo_saida)

# Especificando as pastas de entrada e saída para os três conjuntos de dados
dir_entrada = r"C:\Users\Enenon\Documents\Projetos Programacao\IC\asc sample\dataset mu"
dir_saida = r"C:\Users\Enenon\Documents\Projetos Programacao\IC\asc sample\dataset mu saida"

# Processando todos os arquivos
for T in os.listdir(dir_entrada):
    print('{}:'.format(T))
    processar_pasta(dir_entrada+'\\'+T, dir_saida+'\\'+T)
# Processar todos os arquivos na pasta de entrada

# Por fim:

print('Arquivos processados!')

print('Índices usados:')
print(neur)