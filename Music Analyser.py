import librosa
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import math
import librosa.display
import json


# TODO:
#   - resultado esta legal, precisa de algumas melhorias na hora de pegar as notas
#   - vai ser um microservico
#   - dependendo um gap de 0.5 entre um segundo e outro?

# Carregar o arquivo de áudio
audio_file = r"C:\Users\NOTE155\Desktop\Spleeter\Musics\Desmeon - Hellcat [NCS Release]\drums.wav"
y, sr = librosa.load(audio_file)

# Normalizar o áudio
y_norm = librosa.util.normalize(y)

n_fft = 2048
hop_length = n_fft // 4
# Aplicar a Transformada de Fourier de curto tempo (STFT) no áudio normalizado
D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)

# Converter para decibéis (escala logarítmica)
D_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

# Binarizar as frequências
limiar = -25  # Limiar para considerar frequências relevantes
relevantes = D_db > limiar

# Aplicar K-means para agrupar as frequências relevantes
X = np.argwhere(relevantes)
kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
labels = kmeans.labels_

# Imprimir as frequências relevantes
frequencias_relevantes = np.argwhere(relevantes)
frequencias_relevantes = frequencias_relevantes[frequencias_relevantes[:, 0] != 0]

# Defina o caminho para o arquivo de saída
output_file = r"C:\Users\NOTE155\Desktop\Spleeter\spectro.txt"


def freq_to_note(frequency):
    A4_freq = 440  # Frequência da nota A4 em Hz
    A4_index = 49   # Índice MIDI da nota A4
    semitone_ratio = 2 ** (1/12)

    # Calcula a distância em semitons da frequência para A4
    semitone_distance = 12 * math.log2(frequency / A4_freq)

    # Calcula o índice MIDI da nota correspondente
    note_index = round(semitone_distance) + A4_index

    # Mapeia o índice MIDI para a nota musical correspondente
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    note = notes[note_index % 12]

    opcoes = {
        'C': 1,
        'C#': 1,
        'D': 2,
        'D#': 2,
        'E': 3,
        'F': 3,
        'F#': 4,
        'G': 4,
        'G#': 5,
        'A': 5,
        'A#': 6,
        'B': 6
    }

    # Calcula a oitava da nota
    octave = note_index // 12 - 1

    # return f"{note}{octave}"
    return f"{opcoes[note]}"


def escrever_arquivo(path, conjunto, flag=True):
    if (flag):
        with open(path, "w") as f:
            for dados in conjunto:
                f.write("tamanho: {}\n".format(len(dados)))
                for info in dados:
                    f.write("Tempo: {:.6f}s, Frequência: {:.2f}Hz, Nota: {}, Amplitude: {:.2f}\n".format(
                        info['tempo'], info['frequencia'], info['nota'], info['amplitude']))
                f.write("\n")
    else:
        with open(path, "w") as f:
            json_data = ""
            for info in conjunto:
                for nota in info:
                    json_data = json_data + \
                        f"{nota['tempo']:.6f}|{nota['nota']};"

            f.write(json_data)


def pegar_notas_maior_amplitude(info_conj, n):
    retorno = []
    # Ordenar as informações pelo valor da amplitude em ordem decrescente
    info_conj_ordenado = sorted(
        info_conj, key=lambda x: x['amplitude'], reverse=True)

    contador = 1
    index = 0
    gap = 0.1

    retorno.append(info_conj_ordenado[0])
    for j in range(1, len(info_conj_ordenado)):
        if (info_conj_ordenado[j]['tempo'] == info_conj_ordenado[0]['tempo']):
            retorno.append(info_conj_ordenado[j])
            info_conj_ordenado.remove(info_conj_ordenado[j])
            break

    for i in range(1, len(info_conj_ordenado)):
        flag = True
        for j in range(0, len(retorno)):
            if not (abs(info_conj_ordenado[i]['tempo'] - info_conj_ordenado[j]['tempo']) >= gap):
                flag = False
                break

        if flag:
            retorno.append(info_conj_ordenado[i])
            subvetor = info_conj_ordenado[i:]
            for j in range(1, len(subvetor)):
                if (subvetor[j]['tempo'] == info_conj_ordenado[i]['tempo']):
                    retorno.append(subvetor[j])
            contador = contador + 1

        if (contador == n):
            break

    retorno_ordenado = sorted(retorno, key=lambda x: x['tempo'])

    return retorno_ordenado


amplitudes = []
informacoes = []
ddos_separados_por_tempo_exato = []
dados_separados_por_1s = []
dados_maior = []  # Lista vazia para armazenar os dados
dados_maior_por_sec = []
# Escreva as frequências relevantes, suas notas e amplitudes correspondentes no arquivo
for freq in frequencias_relevantes:
    amplitude = np.abs(D[freq[0], freq[1]])
    amplitudes.append(amplitude)

media_amplitudes = np.mean(amplitudes)
for freq in frequencias_relevantes:
    # Calcula o tempo em segundos
    tempo = librosa.frames_to_time(freq[1], sr=sr, hop_length=hop_length)
    # Calcula a frequência em Hz
    frequencia = freq[0] * sr / D.shape[0]
    # Converte a frequência em nota
    nota = freq_to_note(frequencia)
    # Obtém a amplitude da nota na STFT
    amplitude = np.abs(D[freq[0], freq[1]])
    amplitudes.append(amplitude)

    if amplitude >= media_amplitudes:
        # Cria um dicionário com as informações
        info = {
            'tempo': tempo,
            'frequencia': frequencia,
            'nota': nota,
            'amplitude': amplitude
        }

        informacoes.append(info)

tempo = None
informacoes_ordenadas = sorted(informacoes, key=lambda x: x['tempo'])
dados = []
conjunto = []
# Pegando toda as ocorrencias de um mesmo tempo exato
for info in informacoes_ordenadas:
    if tempo != None and tempo != info['tempo']:
        dados.append(conjunto)
        conjunto = []
    conjunto.append(info)
    tempo = info['tempo']

escrever_arquivo(
    r"C:\\Users\\NOTE155\\Desktop\\Spleeter\\tempoexato.txt", dados)

info = None
contador = 0
# Pegando a(s) maior amplitude de cada tempo exato
for conjunto in dados:
    conjunto_ordenado = sorted(
        conjunto, key=lambda x: (x['amplitude']), reverse=True)
    dados_maior.append(conjunto_ordenado[0])

    if (len(conjunto) > 10):
        # Se o comprimento do conjunto for maior que 10, pegue as duas maiores amplitudes
        for i in conjunto_ordenado:
            if conjunto_ordenado[0]['nota'] != i['nota']:
                dados_maior.append(i)
                info = i
                contador += 1
                break
print("Quantidade de ocorrencias de notas duplas no mesmo exato tempo: " + str(contador))

dados = []
conjunto = []
tempo = None
# Separando toda as ocorrencias de um mesmo range de segundo
for info in dados_maior:
    if tempo != None and tempo != int(info['tempo']):
        dados.append(conjunto)
        conjunto = []
    conjunto.append(info)
    tempo = int(info['tempo'])
dados.append(conjunto)

escrever_arquivo(
    r"C:\\Users\\NOTE155\\Desktop\\Spleeter\\intervalo_1s.txt", dados)


bpm = 120
num_notas = math.floor(bpm/60)
for info_conj in dados:
    if (len(info_conj) >= 40):
        notas_maior_amplitude = pegar_notas_maior_amplitude(
            info_conj, num_notas+3)
    elif (len(info_conj) >= 26):
        notas_maior_amplitude = pegar_notas_maior_amplitude(
            info_conj, num_notas+1)
    else:
        notas_maior_amplitude = pegar_notas_maior_amplitude(
            info_conj, num_notas)
    dados_maior_por_sec.append(notas_maior_amplitude)


escrever_arquivo(
    r"C:\\Users\\NOTE155\\Desktop\\Spleeter\\maior_amplitude.txt", dados_maior_por_sec)
escrever_arquivo(
    r"C:\\Users\\NOTE155\\Desktop\\Spleeter\\resultado.txt", dados_maior_por_sec, False)

# Supondo que 'info' seja uma lista de dicionários como mostrado anteriormente
# Extrai as informações
tempos = []
amplitudes = []
for info in dados_maior_por_sec:
    for nota in info:
        tempos.append(nota['tempo'])
        amplitudes.append(nota['amplitude'])

# Criar figura e eixos para os subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Plotar o primeiro gráfico de amplitude ao longo do tempo
ax1.plot(tempos, amplitudes, marker='o', linestyle='', markersize=5)
ax1.set_xlabel('Tempo (s)')
ax1.set_ylabel('Amplitude')
ax1.set_title('Amplitude ao longo do tempo')

# Plotar o segundo gráfico de espectrograma
img = librosa.display.specshow(
    D_db, sr=sr, x_axis='time', y_axis='log', ax=ax2)
plt.colorbar(img, format='%+2.0f dB', ax=ax2)
ax2.set_title('Espectrograma')

# Ajustar layout para evitar sobreposição de rótulos
plt.tight_layout()

# Exibir os subplots
plt.show()
