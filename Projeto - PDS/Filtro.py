import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

#Carrega o arquivo
samplerate, data = wavfile.read('Áudio - PDS.wav')

#Carrega o arquivo em dois canais (audio estereo)
print(f"numero de canais = {data.shape[1]}")

#Tempo total = numero de amostras / fs
length = data.shape[0] / samplerate
print(f"duracao = {length}s")
print(f"freq amostragem = {samplerate} Hz")

#Plota as figuras ao longo do tempo
#Interpola para determinar eixo do tempo
time = np.linspace(0., length, data.shape[0])

nsampples=data.shape[0]

#Plota os canais esquerdo e direito
plt.figure(1)
plt.plot(time[0:nsampples], data[0:nsampples, 0], label="Canal esquerdo")
plt.legend()
plt.xlabel("Tempo [s]")
plt.ylabel("Amplitude")
plt.show()

plt.figure(2)
plt.plot(time[0:nsampples], data[0:nsampples, 1], label="Canal direito")
plt.legend()
plt.xlabel("Tempo [s]")
plt.ylabel("Amplitude")
plt.show()

# Especifica o valor de N (número de amostras)
N = 300  # Ajuste esse valor conforme necessário

# Pega as primeiras N amostras do sinal (Só estou pegando o canal esquerdo)
signal_chunk = data[:N,0] 

# Calcula a FFT das primeiras N amostras
fft_result = np.fft.fft(signal_chunk)

# Calcula as frequências correspondentes à FFT
frequencies = np.fft.fftfreq(N, d=1/samplerate)

# Encontra o valor absoluto (magnitude) da FFT
magnitude = np.abs(fft_result)

# Plotagem do espectro de frequência
plt.plot(frequencies, magnitude)
plt.xlabel('Frequência (Hz)')
plt.ylabel('Magnitude')
plt.title('Espectro de Frequência para as primeiras N amostras')
plt.show()

