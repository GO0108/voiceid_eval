import os
import numpy as np
import librosa
import scipy.signal
from scipy.signal import butter, filtfilt, wiener, medfilt, sosfilt
from scipy.io import wavfile
import pyloudnorm as pyln
import torchaudio
import shutil
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Try to import noisereduce - it's optional
try:
    import noisereduce as nr
    NOISEREDUCE_AVAILABLE = True
except ImportError:
    NOISEREDUCE_AVAILABLE = False
    print("Warning: noisereduce not available. Install with: pip install noisereduce")

class AudioPreprocessor:
    def __init__(self, sample_rate=16000):
        """
        Inicializa o preprocessador de áudio.
        
        Args:
            sample_rate (int): Taxa de amostragem alvo
        """
        self.sample_rate = sample_rate
        self.meter = pyln.Meter(sample_rate)  # Medidor de loudness
        
    def load_audio(self, file_path):
        """
        Carrega arquivo de áudio e converte para a taxa de amostragem alvo.
        
        Args:
            file_path (str): Caminho para o arquivo de áudio
            
        Returns:
            tuple: (audio_data, sample_rate)
        """
        try:
            # audio, sr = librosa.load(file_path, sr=self.sample_rate, mono=True)
            audio, sr = torchaudio.load(file_path, normalize=True)
            audio = audio.mean(dim=0).numpy().squeeze()
            return audio, sr
        except Exception as e:
            print(f"Erro ao carregar {file_path}: {e}")
            return None, None
    
    def bandpass_filter(self, audio, low_freq=80, high_freq=8000):
        """
        Aplica filtro passa-banda para focar na faixa de frequência da voz.
        
        Args:
            audio (np.array): Sinal de áudio
            low_freq (int): Frequência de corte inferior (Hz)
            high_freq (int): Frequência de corte superior (Hz)
            
        Returns:
            np.array: Áudio filtrado
        """
        nyquist = self.sample_rate / 2
        low = low_freq / nyquist
        high = high_freq / nyquist
        
        # Filtro Butterworth de 4ª ordem
        b, a = butter(4, [low, high], btype='band')
        filtered_audio = filtfilt(b, a, audio)
        
        return filtered_audio
    
    def median_filter_1d(self, audio, kernel_size=5):
        """
        Aplica filtro mediano 1D no sinal de áudio (implementação da sua solicitação).
        
        Args:
            audio (np.array): Sinal de áudio
            kernel_size (int): Tamanho do kernel (deve ser ímpar)
            
        Returns:
            np.array: Áudio filtrado
        """
        return medfilt(audio, kernel_size=kernel_size)
    
    def butterworth_lowpass_sos(self, audio, cutoff_freq=2000, order=10):
        """
        Aplica filtro passa-baixa Butterworth usando SOS (implementação da sua solicitação).
        
        Args:
            audio (np.array): Sinal de áudio
            cutoff_freq (int): Frequência de corte (Hz)
            order (int): Ordem do filtro
            
        Returns:
            np.array: Áudio filtrado
        """
        sos = butter(order, cutoff_freq, btype='lowpass', fs=self.sample_rate, output='sos')
        return sosfilt(sos, audio)
    
    def peak_based_trimming(self, audio, percentile_threshold=99, padding_seconds=0.1):
        """
        Remove marcadores de início/fim baseado em picos de amplitude (implementação da sua solicitação).
        
        Args:
            audio (np.array): Sinal de áudio
            percentile_threshold (float): Percentil para definir limiar de amplitude
            padding_seconds (float): Padding em segundos após o primeiro marcador
            
        Returns:
            np.array: Áudio trimado
        """
        # Define limiar baseado no percentil
        amplitude_threshold = np.percentile(np.abs(audio), percentile_threshold)
        
        # Encontra índices dos picos
        peak_indices = np.where(np.abs(audio) > amplitude_threshold)[0]
        
        if len(peak_indices) < 2:
            # Se não há picos suficientes, retorna o áudio original
            return audio
        
        # Calcula gaps entre picos consecutivos
        gaps = np.diff(peak_indices)
        
        # Encontra o maior gap (presumivelmente entre marcadores)
        split_point_index = np.argmax(gaps)
        start_marker_end = peak_indices[split_point_index]
        
        # Adiciona padding
        padding_samples = int(padding_seconds * self.sample_rate)
        start_sample = start_marker_end + padding_samples
        
        # Retorna áudio a partir do ponto calculado
        if start_sample < len(audio):
            return audio[start_sample:]
        else:
            return audio
    
    def noisereduce_filter(self, audio, stationary=True, prop_decrease=1.0):
        """
        Aplica redução de ruído usando biblioteca noisereduce (implementação da sua solicitação).
        
        Args:
            audio (np.array): Sinal de áudio
            stationary (bool): Se True, assume ruído estacionário
            prop_decrease (float): Proporção de redução do ruído (0.0-1.0)
            
        Returns:
            np.array: Áudio com ruído reduzido
        """
        if not NOISEREDUCE_AVAILABLE:
            print("Warning: noisereduce não disponível, retornando áudio original")
            return audio
            
        try:
            return nr.reduce_noise(y=audio, sr=self.sample_rate, 
                                 stationary=stationary, 
                                 prop_decrease=prop_decrease)
        except Exception as e:
            print(f"Erro no noisereduce: {e}")
            return audio
    
    def high_frequency_emphasis(self, audio, pre_emphasis=0.97):
        """
        Aplica pré-ênfase para realçar frequências altas.
        
        Args:
            audio (np.array): Sinal de áudio
            pre_emphasis (float): Coeficiente de pré-ênfase
            
        Returns:
            np.array: Áudio com pré-ênfase aplicada
        """
        return np.append(audio[0], audio[1:] - pre_emphasis * audio[:-1])
    
    def dynamic_range_compression(self, audio, threshold=0.5, ratio=4.0, attack_time=0.003, release_time=0.1):
        """
        Aplica compressão de faixa dinâmica.
        
        Args:
            audio (np.array): Sinal de áudio
            threshold (float): Limiar de compressão (0-1)
            ratio (float): Razão de compressão
            attack_time (float): Tempo de ataque em segundos
            release_time (float): Tempo de liberação em segundos
            
        Returns:
            np.array: Áudio comprimido
        """
        # Parâmetros
        attack_samples = int(attack_time * self.sample_rate)
        release_samples = int(release_time * self.sample_rate)
        
        # Envelope detector
        envelope = np.abs(audio)
        
        # Smooth envelope
        for i in range(1, len(envelope)):
            if envelope[i] > envelope[i-1]:
                # Attack
                alpha = np.exp(-1.0 / attack_samples)
            else:
                # Release
                alpha = np.exp(-1.0 / release_samples)
            envelope[i] = alpha * envelope[i-1] + (1 - alpha) * envelope[i]
        
        # Compute gain reduction
        gain = np.ones_like(envelope)
        over_threshold = envelope > threshold
        gain[over_threshold] = threshold + (envelope[over_threshold] - threshold) / ratio
        gain[over_threshold] = gain[over_threshold] / envelope[over_threshold]
        
        return audio * gain
    
    def adaptive_gain_control(self, audio, target_rms=0.1, window_size=1024):
        """
        Controle adaptativo de ganho baseado em RMS.
        
        Args:
            audio (np.array): Sinal de áudio
            target_rms (float): RMS alvo
            window_size (int): Tamanho da janela para cálculo do RMS
            
        Returns:
            np.array: Áudio com ganho adaptativo
        """
        output = np.zeros_like(audio)
        
        for i in range(0, len(audio), window_size):
            end_idx = min(i + window_size, len(audio))
            window = audio[i:end_idx]
            
            # Calcula RMS da janela
            current_rms = np.sqrt(np.mean(window**2))
            
            # Calcula ganho necessário
            if current_rms > 0:
                gain = target_rms / current_rms
                # Limita o ganho para evitar amplificação excessiva
                gain = min(gain, 10.0)
            else:
                gain = 1.0
            
            output[i:end_idx] = window * gain
        
        return output
    
    def spectral_gating(self, audio, gate_freq=300, gate_strength=0.3):
        """
        Aplica spectral gating para atenuar frequências específicas.
        
        Args:
            audio (np.array): Sinal de áudio
            gate_freq (float): Frequência central do gate (Hz)
            gate_strength (float): Força da atenuação (0-1)
            
        Returns:
            np.array: Áudio com spectral gating
        """
        # STFT
        stft = librosa.stft(audio, n_fft=2048, hop_length=512)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        # Calcula frequências
        freqs = librosa.fft_frequencies(sr=self.sample_rate, n_fft=2048)
        
        # Encontra índice da frequência de gate
        gate_idx = np.argmin(np.abs(freqs - gate_freq))
        
        # Aplica atenuação na frequência específica
        gate_width = 10  # largura do gate em bins
        start_idx = max(0, gate_idx - gate_width//2)
        end_idx = min(len(freqs), gate_idx + gate_width//2)
        
        magnitude[start_idx:end_idx, :] *= (1 - gate_strength)
        
        # Reconstrói o sinal
        enhanced_stft = magnitude * np.exp(1j * phase)
        return librosa.istft(enhanced_stft, hop_length=512)
    
    def harmonic_percussive_separation(self, audio, use_harmonic=True):
        """
        Separa componentes harmônicos e percussivos, retornando um deles.
        
        Args:
            audio (np.array): Sinal de áudio
            use_harmonic (bool): Se True, retorna parte harmônica; se False, percussiva
            
        Returns:
            np.array: Componente harmônico ou percussivo
        """
        # Separa componentes
        harmonic, percussive = librosa.effects.hpss(audio)
        
        return harmonic if use_harmonic else percussive
    
    def time_stretching(self, audio, stretch_factor=1.0):
        """
        Aplica time-stretching (mudança de velocidade sem alterar pitch).
        
        Args:
            audio (np.array): Sinal de áudio
            stretch_factor (float): Fator de estiramento (>1 = mais lento, <1 = mais rápido)
            
        Returns:
            np.array: Áudio com time-stretching
        """
        if stretch_factor == 1.0:
            return audio
            
        return librosa.effects.time_stretch(audio, rate=1.0/stretch_factor)
    
    def normalize_loudness(self, audio, target_lufs=-23.0):
        """
        Normaliza o áudio usando padrão LUFS.
        
        Args:
            audio (np.array): Sinal de áudio
            target_lufs (float): LUFS alvo (-23 para broadcast, -16 para podcast)
            
        Returns:
            np.array: Áudio normalizado
        """
        try:
            # Mede loudness atual
            loudness = self.meter.integrated_loudness(audio)
            
            # Se o loudness for muito baixo (silêncio), retorna o áudio original
            if loudness < -70.0:
                return audio
            
            # Normaliza para o LUFS alvo
            normalized_audio = pyln.normalize.loudness(audio, loudness, target_lufs)
            
            # Previne clipping
            if np.max(np.abs(normalized_audio)) > 0.95:
                normalized_audio = normalized_audio / np.max(np.abs(normalized_audio)) * 0.95
                
            return normalized_audio
        except Exception as e:
            print(f"Erro na normalização de loudness: {e}")
            return audio
    
    def simple_vad(self, audio, frame_length=2048, hop_length=512, energy_threshold=0.01):
        """
        Detecção de Atividade de Voz simples baseada em energia.
        
        Args:
            audio (np.array): Sinal de áudio
            frame_length (int): Tamanho do frame
            hop_length (int): Passo do hop
            energy_threshold (float): Limiar de energia
            
        Returns:
            np.array: Áudio com segmentos de silêncio removidos
        """
        # Calcula energia de curto prazo
        energy = librosa.feature.rms(y=audio, frame_length=frame_length, 
                                   hop_length=hop_length)[0]
        
        # Normaliza energia
        energy = energy / np.max(energy)
        
        # Identifica frames com atividade de voz
        voice_frames = energy > energy_threshold
        
        # Converte índices de frames para índices de amostras
        voice_samples = np.zeros(len(audio), dtype=bool)
        for i, is_voice in enumerate(voice_frames):
            start_sample = i * hop_length
            end_sample = min(start_sample + frame_length, len(audio))
            voice_samples[start_sample:end_sample] = is_voice
        
        # Retorna apenas os segmentos com voz
        if np.any(voice_samples):
            return audio[voice_samples]
        else:
            return audio  # Se não detectar voz, retorna o áudio original
    
    def spectral_subtraction(self, audio, alpha=2.0, beta=0.1):
        """
        Implementa subtração espectral para redução de ruído.
        
        Args:
            audio (np.array): Sinal de áudio
            alpha (float): Fator de sobre-subtração
            beta (float): Flooring espectral (0.1 = 10% do espectro original)
            
        Returns:
            np.array: Áudio com ruído reduzido
        """
        # Parâmetros STFT
        n_fft = 2048
        hop_length = 512
        
        # Calcula STFT
        stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        # Estima o ruído dos primeiros e últimos 10% do áudio
        noise_frames = int(0.1 * magnitude.shape[1])
        noise_spectrum = np.mean(np.concatenate([
            magnitude[:, :noise_frames],
            magnitude[:, -noise_frames:]
        ], axis=1), axis=1, keepdims=True)
        
        # Subtração espectral com over-subtraction
        enhanced_magnitude = magnitude - alpha * noise_spectrum
        
        # Aplicar flooring espectral
        enhanced_magnitude = np.maximum(enhanced_magnitude, 
                                      beta * magnitude)
        
        # Reconstroi o sinal
        enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
        enhanced_audio = librosa.istft(enhanced_stft, hop_length=hop_length)
        
        return enhanced_audio
    
    def wiener_filter(self, audio, noise_estimation_ratio=0.1):
        """
        Implementa filtro de Wiener para redução de ruído.
        
        Args:
            audio (np.array): Sinal de áudio
            noise_estimation_ratio (float): Proporção do áudio usada para estimar ruído
            
        Returns:
            np.array: Áudio filtrado
        """
        # Parâmetros STFT
        n_fft = 2048
        hop_length = 512
        
        # Calcula STFT
        stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        # Estima o espectro de potência do ruído
        noise_frames = int(noise_estimation_ratio * magnitude.shape[1])
        noise_power = np.mean(np.concatenate([
            magnitude[:, :noise_frames] ** 2,
            magnitude[:, -noise_frames:] ** 2
        ], axis=1), axis=1, keepdims=True)
        
        # Estima o espectro de potência do sinal
        signal_power = magnitude ** 2
        
        # Calcula o ganho do filtro de Wiener
        # H(w) = P_signal / (P_signal + P_noise)
        wiener_gain = signal_power / (signal_power + noise_power)
        
        # Aplica o filtro
        enhanced_magnitude = magnitude * wiener_gain
        
        # Reconstroi o sinal
        enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
        enhanced_audio = librosa.istft(enhanced_stft, hop_length=hop_length)
        
        return enhanced_audio
    
    def lowpass_filter(self, audio, cutoff_freq=3500):
        """
        Aplica filtro passa-baixa para limitar frequências altas.
        
        Args:
            audio (np.array): Sinal de áudio
            cutoff_freq (int): Frequência de corte (Hz)
            
        Returns:
            np.array: Áudio filtrado
        """
        nyquist = self.sample_rate / 2
        normalized_cutoff = cutoff_freq / nyquist
        
        # Filtro Butterworth de 6ª ordem para corte mais acentuado
        b, a = butter(6, normalized_cutoff, btype='low')
        filtered_audio = filtfilt(b, a, audio)
        
        return filtered_audio
    
    def advanced_denoiser(self, audio, method='median_filter', **kwargs):
        """
        Implementa técnicas avançadas de denoising.
        
        Args:
            audio (np.array): Sinal de áudio
            method (str): Método de denoising ('median_filter', 'adaptive_wiener', 'rnnoise_like')
            **kwargs: Parâmetros específicos do método
            
        Returns:
            np.array: Áudio com ruído reduzido
        """
        if method == 'median_filter':
            return self._median_filter_denoiser(audio, **kwargs)
        elif method == 'adaptive_wiener':
            return self._adaptive_wiener_denoiser(audio, **kwargs)
        elif method == 'rnnoise_like':
            return self._rnnoise_like_denoiser(audio, **kwargs)
        else:
            return audio
    
    def _median_filter_denoiser(self, audio, kernel_size=5, n_fft=2048, hop_length=512):
        """
        Denoiser baseado em filtro mediano no domínio espectral.
        """
        # STFT
        stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        # Aplica filtro mediano na magnitude
        from scipy.ndimage import median_filter
        filtered_magnitude = median_filter(magnitude, size=(1, kernel_size))
        
        # Reconstrói o sinal
        enhanced_stft = filtered_magnitude * np.exp(1j * phase)
        enhanced_audio = librosa.istft(enhanced_stft, hop_length=hop_length)
        
        return enhanced_audio
    
    def _adaptive_wiener_denoiser(self, audio, alpha=0.98, n_fft=2048, hop_length=512):
        """
        Filtro de Wiener adaptativo com estimativa temporal de ruído.
        """
        # STFT
        stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        # Estimativa adaptativa de ruído
        noise_power = np.zeros_like(magnitude)
        noise_power[:, 0] = magnitude[:, 0] ** 2  # Inicialização
        
        for t in range(1, magnitude.shape[1]):
            # Smoothing exponencial da estimativa de ruído
            current_power = magnitude[:, t] ** 2
            noise_power[:, t] = alpha * noise_power[:, t-1] + (1-alpha) * current_power
        
        # Filtro de Wiener adaptativo
        signal_power = magnitude ** 2
        wiener_gain = signal_power / (signal_power + noise_power)
        
        # Aplica suavização do ganho para evitar artefatos
        wiener_gain = np.maximum(wiener_gain, 0.1)  # Mínimo 10% do sinal
        
        enhanced_magnitude = magnitude * wiener_gain
        
        # Reconstrói
        enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
        enhanced_audio = librosa.istft(enhanced_stft, hop_length=hop_length)
        
        return enhanced_audio
    
    def _rnnoise_like_denoiser(self, audio, frame_size=480, overlap=240):
        """
        Implementação simplificada inspirada no RNNoise.
        Usa análise de características espectrais para detecção de ruído.
        """
        # Divide em frames com overlap
        frames = []
        for i in range(0, len(audio) - frame_size, overlap):
            frames.append(audio[i:i + frame_size])
        
        enhanced_frames = []
        
        for frame in frames:
            # FFT do frame
            fft_frame = np.fft.fft(frame)
            magnitude = np.abs(fft_frame)
            phase = np.angle(fft_frame)
            
            # Características espectrais para detecção de voz
            spectral_centroid = np.sum(magnitude * np.arange(len(magnitude))) / np.sum(magnitude)
            spectral_rolloff = len(magnitude) * 0.85  # 85% da energia
            spectral_flux = np.sum(np.diff(magnitude) ** 2)
            
            # Heurística simples: se as características indicam voz, preserva mais
            if spectral_centroid > 50 and spectral_flux > 0.1:
                # Frame com voz - menor atenuação
                gain = np.ones_like(magnitude)
                gain[magnitude < np.percentile(magnitude, 20)] = 0.3
            else:
                # Frame com ruído - maior atenuação
                gain = np.ones_like(magnitude)
                gain[magnitude < np.percentile(magnitude, 30)] = 0.1
            
            # Aplica ganho
            enhanced_magnitude = magnitude * gain
            enhanced_fft = enhanced_magnitude * np.exp(1j * phase)
            enhanced_frame = np.real(np.fft.ifft(enhanced_fft))
            enhanced_frames.append(enhanced_frame)
        
        # Reconstrói o áudio usando overlap-add
        enhanced_audio = np.zeros(len(audio))
        for i, frame in enumerate(enhanced_frames):
            start_idx = i * overlap
            end_idx = start_idx + frame_size
            if end_idx <= len(enhanced_audio):
                enhanced_audio[start_idx:end_idx] += frame
        
        return enhanced_audio
    
    def peak_clipping(self, audio, threshold=0.95, method='hard'):
        """
        Aplica clipping de valores de pico.
        
        Args:
            audio (np.array): Sinal de áudio
            threshold (float): Limiar de clipping (0-1)
            method (str): Tipo de clipping ('hard', 'soft', 'cubic')
            
        Returns:
            np.array: Áudio com clipping aplicado
        """
        if method == 'hard':
            # Clipping hard tradicional
            clipped_audio = np.clip(audio, -threshold, threshold)
        
        elif method == 'soft':
            # Soft clipping com função tanh
            clipped_audio = np.tanh(audio / threshold) * threshold
        
        elif method == 'cubic':
            # Clipping cúbico - mais suave
            def cubic_clip(x, thresh):
                if abs(x) <= thresh:
                    return x
                else:
                    sign = np.sign(x)
                    x_norm = abs(x) / thresh
                    return sign * thresh * (2 - x_norm**2) if x_norm < 2 else sign * thresh
            
            clipped_audio = np.vectorize(lambda x: cubic_clip(x, threshold))(audio)
        
        else:
            clipped_audio = audio
        
        return clipped_audio
    
    def silence_clipping(self, audio, threshold_db=-40, min_silence_duration=0.1):
        """
        Remove ou atenua segmentos muito silenciosos.
        
        Args:
            audio (np.array): Sinal de áudio
            threshold_db (float): Limiar de silêncio em dB
            min_silence_duration (float): Duração mínima de silêncio para processar (segundos)
            
        Returns:
            np.array: Áudio processado
        """
        # Converte threshold para amplitude linear
        threshold_linear = 10 ** (threshold_db / 20)
        
        # Calcula envelope de amplitude com janela móvel
        window_size = int(0.05 * self.sample_rate)  # 50ms window
        envelope = np.convolve(np.abs(audio), np.ones(window_size)/window_size, mode='same')
        
        # Identifica regiões silenciosas
        silence_mask = envelope < threshold_linear
        
        # Aplica filtro de duração mínima
        min_samples = int(min_silence_duration * self.sample_rate)
        
        # Encontra início e fim de regiões silenciosas
        silence_regions = []
        in_silence = False
        start_silence = 0
        
        for i, is_silent in enumerate(silence_mask):
            if is_silent and not in_silence:
                start_silence = i
                in_silence = True
            elif not is_silent and in_silence:
                if i - start_silence >= min_samples:
                    silence_regions.append((start_silence, i))
                in_silence = False
        
        # Processa última região se necessário
        if in_silence and len(silence_mask) - start_silence >= min_samples:
            silence_regions.append((start_silence, len(silence_mask)))
        
        # Aplica processamento às regiões silenciosas
        processed_audio = audio.copy()
        for start, end in silence_regions:
            # Atenua região silenciosa em vez de remover completamente
            processed_audio[start:end] *= 0.1  # Reduz para 10% do volume original
        
        return processed_audio
    
    def save_audio(self, audio, output_path):
        """
        Salva o áudio processado.
        
        Args:
            audio (np.array): Dados de áudio
            output_path (str): Caminho de saída
        """
        # Normaliza para evitar clipping
        try:
            if np.max(np.abs(audio)) > 0:
                audio = audio / np.max(np.abs(audio)) * 0.95
        except Exception as e:
            pass
        # Converte para int16
        audio_int16 = (audio * 32767).astype(np.int16)
        
        # Cria diretório se não existir
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Salva arquivo
        wavfile.write(output_path, self.sample_rate, audio_int16)

def process_dataset(input_path, output_base_path):
    """
    Processa todo o dataset aplicando diferentes técnicas de pré-processamento.
    
    Args:
        input_path (str): Caminho para o dataset original
        output_base_path (str): Caminho base para os datasets processados
    """
    processor = AudioPreprocessor()
    
    # Define os caminhos de saída para cada tipo de processamento
    output_paths = {
        'originais': os.path.join(output_base_path, 'originais'),
        'linha_de_base': os.path.join(output_base_path, 'linha_de_base'),
        'denoised_ss': os.path.join(output_base_path, 'denoised_ss'),
        'denoised_wf': os.path.join(output_base_path, 'denoised_wf'),
        'lowpass_3500': os.path.join(output_base_path, 'lowpass_3500'),
        'median_filter_1d': os.path.join(output_base_path, 'median_filter_1d'),
        'butterworth_sos': os.path.join(output_base_path, 'butterworth_sos'),
        'peak_trimming': os.path.join(output_base_path, 'peak_trimming'),
        'noisereduce': os.path.join(output_base_path, 'noisereduce'),
        'pre_emphasis': os.path.join(output_base_path, 'pre_emphasis'),
        'dynamic_compression': os.path.join(output_base_path, 'dynamic_compression'),
        'adaptive_gain': os.path.join(output_base_path, 'adaptive_gain'),
        'spectral_gating': os.path.join(output_base_path, 'spectral_gating'),
        'harmonic_separation': os.path.join(output_base_path, 'harmonic_separation'),
        'advanced_denoiser_median': os.path.join(output_base_path, 'advanced_denoiser_median'),
        'advanced_denoiser_adaptive': os.path.join(output_base_path, 'advanced_denoiser_adaptive'),
        'advanced_denoiser_rnnoise': os.path.join(output_base_path, 'advanced_denoiser_rnnoise'),
        'peak_clipping_hard': os.path.join(output_base_path, 'peak_clipping_hard'),
        'peak_clipping_soft': os.path.join(output_base_path, 'peak_clipping_soft'),
        'silence_clipped': os.path.join(output_base_path, 'silence_clipped'),
        'combined_aggressive': os.path.join(output_base_path, 'combined_aggressive'),
        'combined_conservative': os.path.join(output_base_path, 'combined_conservative'),
        'combined_new_techniques': os.path.join(output_base_path, 'combined_new_techniques')
    }
    
    # Cria os diretórios de saída
    for path in output_paths.values():
        os.makedirs(path, exist_ok=True)
    
    # Processa todos os arquivos .wav no diretório
    audio_files = []
    for root, dirs, files in os.walk(input_path):
        for file in files:
            if file.endswith('.wav'):
                audio_files.append(os.path.join(root, file))
    
    print(f"Encontrados {len(audio_files)} arquivos de áudio para processar...")
    
    for i, file_path in enumerate(audio_files):
        print(f"Processando {i+1}/{len(audio_files)}: {os.path.basename(file_path)}")
        
        # Carrega o áudio
        audio, sr = processor.load_audio(file_path)
        if audio is None:
            continue
        
        # Nome do arquivo
        filename = os.path.basename(file_path)
        
        # 1. ORIGINAIS: Copia o arquivo original (apenas resampling)
        original_path = os.path.join(output_paths['originais'], filename)
        processor.save_audio(audio, original_path)
        
        # 2. LINHA DE BASE: Aplicar condicionamento fundamental
        # a) Filtro passa-banda
        audio_filtered = processor.bandpass_filter(audio, low_freq=80, high_freq=3500)
        
        # b) Normalização de loudness
        audio_normalized = processor.normalize_loudness(audio_filtered, target_lufs=-23.0)
        
        # c) Detecção de Atividade de Voz (VAD)
        audio_vad = processor.simple_vad(audio_normalized)
        
        # Salva linha de base
        baseline_path = os.path.join(output_paths['linha_de_base'], filename)
        processor.save_audio(audio_vad, baseline_path)
        
        # 3. MEDIAN FILTER 1D: Aplicação do filtro mediano 1D (sua solicitação)
        try:
            audio_median_1d = processor.median_filter_1d(audio_vad, kernel_size=5)
            median_1d_path = os.path.join(output_paths['median_filter_1d'], filename)
            processor.save_audio(audio_median_1d, median_1d_path)
        except Exception as e:
            print(f"Erro no filtro mediano 1D para {filename}: {e}")
            shutil.copy2(baseline_path, os.path.join(output_paths['median_filter_1d'], filename))
        
        # 4. BUTTERWORTH SOS: Filtro Butterworth usando SOS (sua solicitação)
        try:
            audio_butter_sos = processor.butterworth_lowpass_sos(audio_vad, cutoff_freq=2000, order=10)
            butter_sos_path = os.path.join(output_paths['butterworth_sos'], filename)
            processor.save_audio(audio_butter_sos, butter_sos_path)
        except Exception as e:
            print(f"Erro no Butterworth SOS para {filename}: {e}")
            shutil.copy2(baseline_path, os.path.join(output_paths['butterworth_sos'], filename))
        
        # 5. PEAK TRIMMING: Remoção de marcadores baseada em picos (sua solicitação)
        try:
            audio_peak_trim = processor.peak_based_trimming(audio_vad, percentile_threshold=99, padding_seconds=0.1)
            peak_trim_path = os.path.join(output_paths['peak_trimming'], filename)
            processor.save_audio(audio_peak_trim, peak_trim_path)
        except Exception as e:
            print(f"Erro no peak trimming para {filename}: {e}")
            shutil.copy2(baseline_path, os.path.join(output_paths['peak_trimming'], filename))
        
        # 6. NOISEREDUCE: Biblioteca noisereduce (sua solicitação)
        try:
            audio_nr = processor.noisereduce_filter(audio_vad, stationary=True, prop_decrease=1.0)
            nr_path = os.path.join(output_paths['noisereduce'], filename)
            processor.save_audio(audio_nr, nr_path)
        except Exception as e:
            print(f"Erro no noisereduce para {filename}: {e}")
            shutil.copy2(baseline_path, os.path.join(output_paths['noisereduce'], filename))
        
        # 7. PRE-EMPHASIS: Realce de frequências altas
        try:
            audio_pre_emph = processor.high_frequency_emphasis(audio_vad, pre_emphasis=0.97)
            pre_emph_path = os.path.join(output_paths['pre_emphasis'], filename)
            processor.save_audio(audio_pre_emph, pre_emph_path)
        except Exception as e:
            print(f"Erro na pré-ênfase para {filename}: {e}")
            shutil.copy2(baseline_path, os.path.join(output_paths['pre_emphasis'], filename))
        
        # 8. DYNAMIC COMPRESSION: Compressão de faixa dinâmica
        try:
            audio_compressed = processor.dynamic_range_compression(audio_vad, threshold=0.5, ratio=4.0)
            compressed_path = os.path.join(output_paths['dynamic_compression'], filename)
            processor.save_audio(audio_compressed, compressed_path)
        except Exception as e:
            print(f"Erro na compressão dinâmica para {filename}: {e}")
            shutil.copy2(baseline_path, os.path.join(output_paths['dynamic_compression'], filename))
        
        # 9. ADAPTIVE GAIN: Controle adaptativo de ganho
        try:
            audio_adaptive_gain = processor.adaptive_gain_control(audio_vad, target_rms=0.1)
            adaptive_gain_path = os.path.join(output_paths['adaptive_gain'], filename)
            processor.save_audio(audio_adaptive_gain, adaptive_gain_path)
        except Exception as e:
            print(f"Erro no controle adaptativo de ganho para {filename}: {e}")
            shutil.copy2(baseline_path, os.path.join(output_paths['adaptive_gain'], filename))
        
        # 10. SPECTRAL GATING: Atenuação de frequências específicas
        try:
            audio_gated = processor.spectral_gating(audio_vad, gate_freq=300, gate_strength=0.3)
            gated_path = os.path.join(output_paths['spectral_gating'], filename)
            processor.save_audio(audio_gated, gated_path)
        except Exception as e:
            print(f"Erro no spectral gating para {filename}: {e}")
            shutil.copy2(baseline_path, os.path.join(output_paths['spectral_gating'], filename))
        
        # 11. HARMONIC SEPARATION: Separação harmônico/percussivo
        try:
            audio_harmonic = processor.harmonic_percussive_separation(audio_vad, use_harmonic=True)
            harmonic_path = os.path.join(output_paths['harmonic_separation'], filename)
            processor.save_audio(audio_harmonic, harmonic_path)
        except Exception as e:
            print(f"Erro na separação harmônica para {filename}: {e}")
            shutil.copy2(baseline_path, os.path.join(output_paths['harmonic_separation'], filename))
        
        # 12. DENOISED_SS: Subtração Espectral aplicada à linha de base
        try:
            audio_ss = processor.spectral_subtraction(audio_vad, alpha=2.0, beta=0.1)
            ss_path = os.path.join(output_paths['denoised_ss'], filename)
            processor.save_audio(audio_ss, ss_path)
        except Exception as e:
            print(f"Erro na subtração espectral para {filename}: {e}")
            shutil.copy2(baseline_path, os.path.join(output_paths['denoised_ss'], filename))
        
        # 13. DENOISED_WF: Filtro de Wiener aplicado à linha de base
        try:
            audio_wf = processor.wiener_filter(audio_vad, noise_estimation_ratio=0.1)
            wf_path = os.path.join(output_paths['denoised_wf'], filename)
            processor.save_audio(audio_wf, wf_path)
        except Exception as e:
            print(f"Erro no filtro de Wiener para {filename}: {e}")
            shutil.copy2(baseline_path, os.path.join(output_paths['denoised_wf'], filename))
        
        # 14. LOWPASS_3500: Filtro passa-baixa a 3500Hz aplicado à linha de base
        try:
            audio_lp = processor.lowpass_filter(audio_vad, cutoff_freq=3500)
            lp_path = os.path.join(output_paths['lowpass_3500'], filename)
            processor.save_audio(audio_lp, lp_path)
        except Exception as e:
            print(f"Erro no filtro passa-baixa para {filename}: {e}")
            shutil.copy2(baseline_path, os.path.join(output_paths['lowpass_3500'], filename))
        
        # 15. ADVANCED DENOISERS: Diferentes técnicas de denoising avançado
        # 15a. Median Filter Denoiser
        try:
            audio_median = processor.advanced_denoiser(audio_vad, method='median_filter', kernel_size=5)
            median_path = os.path.join(output_paths['advanced_denoiser_median'], filename)
            processor.save_audio(audio_median, median_path)
        except Exception as e:
            print(f"Erro no denoiser median para {filename}: {e}")
            shutil.copy2(baseline_path, os.path.join(output_paths['advanced_denoiser_median'], filename))
        
        # 15b. Adaptive Wiener Denoiser
        try:
            audio_adaptive = processor.advanced_denoiser(audio_vad, method='adaptive_wiener', alpha=0.95)
            adaptive_path = os.path.join(output_paths['advanced_denoiser_adaptive'], filename)
            processor.save_audio(audio_adaptive, adaptive_path)
        except Exception as e:
            print(f"Erro no denoiser adaptativo para {filename}: {e}")
            shutil.copy2(baseline_path, os.path.join(output_paths['advanced_denoiser_adaptive'], filename))
        
        # 15c. RNNoise-like Denoiser
        try:
            audio_rnnoise = processor.advanced_denoiser(audio_vad, method='rnnoise_like')
            rnnoise_path = os.path.join(output_paths['advanced_denoiser_rnnoise'], filename)
            processor.save_audio(audio_rnnoise, rnnoise_path)
        except Exception as e:
            print(f"Erro no denoiser RNNoise-like para {filename}: {e}")
            shutil.copy2(baseline_path, os.path.join(output_paths['advanced_denoiser_rnnoise'], filename))
        
        # 16. PEAK CLIPPING: Diferentes tipos de clipping
        # 16a. Hard clipping
        try:
            audio_hard_clip = processor.peak_clipping(audio_vad, threshold=0.9, method='hard')
            hard_clip_path = os.path.join(output_paths['peak_clipping_hard'], filename)
            processor.save_audio(audio_hard_clip, hard_clip_path)
        except Exception as e:
            print(f"Erro no hard clipping para {filename}: {e}")
            shutil.copy2(baseline_path, os.path.join(output_paths['peak_clipping_hard'], filename))
        
        # 16b. Soft clipping
        try:
            audio_soft_clip = processor.peak_clipping(audio_vad, threshold=0.8, method='soft')
            soft_clip_path = os.path.join(output_paths['peak_clipping_soft'], filename)
            processor.save_audio(audio_soft_clip, soft_clip_path)
        except Exception as e:
            print(f"Erro no soft clipping para {filename}: {e}")
            shutil.copy2(baseline_path, os.path.join(output_paths['peak_clipping_soft'], filename))
        
        # 17. SILENCE CLIPPING: Remoção/atenuação de silêncios
        try:
            audio_silence_clip = processor.silence_clipping(audio_vad, threshold_db=-45, min_silence_duration=0.2)
            silence_clip_path = os.path.join(output_paths['silence_clipped'], filename)
            processor.save_audio(audio_silence_clip, silence_clip_path)
        except Exception as e:
            print(f"Erro no silence clipping para {filename}: {e}")
            shutil.copy2(baseline_path, os.path.join(output_paths['silence_clipped'], filename))
        
        # 18. COMBINAÇÕES: Abordagens combinadas
        # 18a. Aggressive: Múltiplas técnicas agressivas
        try:
            # Pipeline agressivo: lowpass + denoising + hard clipping + silence clipping
            audio_combined_aggr = processor.lowpass_filter(audio_vad, cutoff_freq=3500)
            audio_combined_aggr = processor.advanced_denoiser(audio_combined_aggr, method='adaptive_wiener', alpha=0.98)
            audio_combined_aggr = processor.peak_clipping(audio_combined_aggr, threshold=0.85, method='hard')
            audio_combined_aggr = processor.silence_clipping(audio_combined_aggr, threshold_db=-50, min_silence_duration=0.15)
            
            combined_aggr_path = os.path.join(output_paths['combined_aggressive'], filename)
            processor.save_audio(audio_combined_aggr, combined_aggr_path)
        except Exception as e:
            print(f"Erro na combinação agressiva para {filename}: {e}")
            shutil.copy2(baseline_path, os.path.join(output_paths['combined_aggressive'], filename))
        
        # 18b. Conservative: Combinação mais conservadora
        try:
            # Pipeline conservador: denoising suave + soft clipping suave
            audio_combined_cons = processor.advanced_denoiser(audio_vad, method='median_filter', kernel_size=3)
            audio_combined_cons = processor.peak_clipping(audio_combined_cons, threshold=0.95, method='soft')
            
            combined_cons_path = os.path.join(output_paths['combined_conservative'], filename)
            processor.save_audio(audio_combined_cons, combined_cons_path)
        except Exception as e:
            print(f"Erro na combinação conservadora para {filename}: {e}")
            shutil.copy2(baseline_path, os.path.join(output_paths['combined_conservative'], filename))
        
        # 18c. NEW TECHNIQUES: Combinação das novas técnicas solicitadas
        try:
            # Pipeline com as novas técnicas: peak trimming + median filter 1D + butterworth SOS + noisereduce
            audio_new_tech = processor.peak_based_trimming(audio_vad, percentile_threshold=99)
            if NOISEREDUCE_AVAILABLE:
                audio_new_tech = processor.noisereduce_filter(audio_new_tech, stationary=True, prop_decrease=0.8)
            audio_new_tech = processor.median_filter_1d(audio_new_tech, kernel_size=5)
            audio_new_tech = processor.butterworth_lowpass_sos(audio_new_tech, cutoff_freq=2500, order=8)
            audio_new_tech = processor.high_frequency_emphasis(audio_new_tech, pre_emphasis=0.95)
            
            new_tech_path = os.path.join(output_paths['combined_new_techniques'], filename)
            processor.save_audio(audio_new_tech, new_tech_path)
        except Exception as e:
            print(f"Erro na combinação de novas técnicas para {filename}: {e}")
            shutil.copy2(baseline_path, os.path.join(output_paths['combined_new_techniques'], filename))
    
    # Copia o arquivo pairs.txt para todos os diretórios de saída
    pairs_file = os.path.join(input_path, 'pairs.txt')
    if os.path.exists(pairs_file):
        for output_path in output_paths.values():
            shutil.copy2(pairs_file, os.path.join(output_path, 'pairs.txt'))
    
    print("\nProcessamento concluído!")
    print("Datasets criados:")
    for name, path in output_paths.items():
        file_count = len([f for f in os.listdir(path) if f.endswith('.wav')])
        print(f"  - {name}: {file_count} arquivos em {path}")

def create_processing_summary(output_base_path):
    """
    Cria um arquivo de resumo explicando cada tipo de processamento.
    
    Args:
        output_base_path (str): Caminho base dos datasets processados
    """
    summary = """# Resumo dos Datasets Processados - Versão Expandida

## Técnicas Básicas

### 1. Originais
- **Descrição**: Arquivos de áudio originais com apenas resampling para 16kHz
- **Processamento**: Nenhum além da padronização da taxa de amostragem
- **Objetivo**: Servir como controle experimental

### 2. Linha de Base
- **Descrição**: Condicionamento fundamental do sinal
- **Processamento aplicado**:
  1. Filtro passa-banda (80Hz - 8kHz) - Remove ruídos de baixa/alta frequência
  2. Normalização de loudness (-23 LUFS) - Padroniza volume percebido
  3. Detecção de Atividade de Voz (VAD) - Remove segmentos de silêncio
- **Objetivo**: Estabelecer padrão de boas práticas antes da redução agressiva de ruído

## Técnicas Específicas Solicitadas

### 3. Median Filter 1D
- **Descrição**: Aplicação de filtro mediano unidimensional ao sinal temporal
- **Processamento**: `scipy.signal.medfilt(y, kernel_size=5)`
- **Características**: Remove impulsos e ruído impulsivo no domínio temporal
- **Objetivo**: Testar remoção de ruído impulsivo preservando bordas do sinal

### 4. Butterworth SOS
- **Descrição**: Filtro passa-baixa Butterworth usando Second-Order Sections
- **Processamento**: `butter(10, 2000, btype='lowpass', fs=sr, output='sos')` + `sosfilt()`
- **Características**: Implementação numericamente mais estável do filtro Butterworth
- **Objetivo**: Comparar com implementação tradicional do filtro passa-baixa

### 5. Peak Trimming
- **Descrição**: Remoção de marcadores baseada em detecção de picos
- **Processamento**: 
  - Detecta picos acima do percentil 99%
  - Remove início até primeiro gap grande entre picos + padding
- **Características**: Remove automaticamente marcadores de sincronização
- **Objetivo**: Melhorar qualidade removendo artefatos de sincronização

### 6. NoiseReduce
- **Descrição**: Biblioteca noisereduce para redução de ruído
- **Processamento**: `noisereduce.reduce_noise(y=y, sr=sr)`
- **Características**: Algoritmo avançado de redução de ruído baseado em subtração espectral
- **Objetivo**: Comparar com implementações próprias de denoising

## Técnicas Avançadas Adicionadas

### 7. Pre-Emphasis
- **Descrição**: Realce de frequências altas usando pré-ênfase
- **Processamento**: Filtro FIR de primeira ordem (coef. 0.97)
- **Características**: Melhora SNR em frequências altas da voz
- **Objetivo**: Preparar sinal para melhor reconhecimento de características vocais

### 8. Dynamic Compression
- **Descrição**: Compressão de faixa dinâmica adaptativa
- **Processamento**: Compressor com ataque/liberação configuráveis
- **Características**: Reduz variações de amplitude preservando detalhes
- **Objetivo**: Normalizar variações de volume entre locutores

### 9. Adaptive Gain
- **Descrição**: Controle adaptativo de ganho baseado em RMS
- **Processamento**: Ajuste de ganho por janelas baseado em RMS alvo
- **Características**: Mantém nível de energia consistente
- **Objetivo**: Compensar variações de nível entre segmentos

### 10. Spectral Gating
- **Descrição**: Atenuação seletiva de frequências específicas
- **Processamento**: Redução de energia em banda específica (300Hz)
- **Características**: Remove componentes de ruído em frequência específica
- **Objetivo**: Atenuar ruído de linha elétrica ou ventiladores

### 11. Harmonic Separation
- **Descrição**: Separação de componentes harmônicos vs. percussivos
- **Processamento**: HPSS (Harmonic-Percussive Source Separation)
- **Características**: Preserva apenas componentes tonais (harmônicos)
- **Objetivo**: Remover ruídos transientes preservando características vocais

## Técnicas de Denoising Existentes

### 12. Denoised_SS (Subtração Espectral)
- **Descrição**: Linha de base + Subtração Espectral
- **Processamento adicional**: 
  - Subtração espectral com fator de sobre-subtração = 2.0
  - Flooring espectral = 10% do espectro original
- **Características**: Maior redução de ruído, mas pode introduzir "ruído musical"

### 13. Denoised_WF (Filtro de Wiener)
- **Descrição**: Linha de base + Filtro de Wiener
- **Processamento adicional**: Filtro de Wiener com estimativa de ruído baseada em 10% do áudio
- **Características**: Redução de ruído mais suave, chiado residual mais consistente

### 14-16. Advanced Denoisers
- **Median Filter Denoiser**: Filtro mediano no domínio espectral
- **Adaptive Wiener**: Wiener com estimativa temporal de ruído
- **RNNoise-like**: Heurísticas de detecção voz/ruído

### 17-18. Peak Clipping & Silence Clipping
- **Peak Clipping**: Hard/soft clipping de amplitude
- **Silence Clipping**: Atenuação de segmentos silenciosos

## Combinações Avançadas

### 19. Combined Aggressive
- **Pipeline**: Lowpass + Adaptive Wiener + Hard Clipping + Silence Clipping
- **Objetivo**: Máxima redução de ruído e artefatos

### 20. Combined Conservative
- **Pipeline**: Median Filter + Soft Clipping suave
- **Objetivo**: Processamento suave preservando qualidade

### 21. Combined New Techniques (NOVO)
- **Pipeline**: Peak Trimming + NoiseReduce + Median Filter 1D + Butterworth SOS + Pre-emphasis
- **Objetivo**: Integrar todas as novas técnicas solicitadas em pipeline otimizado

## Hipóteses de Teste

1. **Filtros temporais vs. espectrais**: Median filter 1D vs. Median filter espectral
2. **Implementação de filtros**: Butterworth tradicional vs. SOS
3. **Denoising**: NoiseReduce vs. implementações próprias
4. **Preprocessing automático**: Peak trimming vs. processamento manual
5. **Combinações**: Técnicas individuais vs. pipelines combinados

## Métricas de Avaliação Recomendadas
- Equal Error Rate (EER)
- Detection Error Tradeoff (DET)  
- Minimum Detection Cost Function (minDCF)
- Análise de SNR antes/depois
- Testes perceptuais de qualidade

## Estrutura para Comparação
1. **Baseline**: Originais vs. Linha de Base vs. técnicas individuais
2. **Denoising**: Comparação entre todas as técnicas de redução de ruído
3. **Preprocessing**: Impacto de peak trimming e outras técnicas de preparação
4. **Pipelines**: Efetividade de combinações vs. técnicas isoladas
"""
    
    summary_path = os.path.join(output_base_path, 'RESUMO_PROCESSAMENTO_EXPANDIDO.md')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary)
    
    print(f"Resumo expandido salvo em: {summary_path}")

if __name__ == "__main__":
    # Configurações
    input_dataset_path = "toy_dataset/toy_dataset_tv/Download"
    output_base_path = "toy_dataset/processed_tv_enhanced"
    
    # Verifica se o diretório de entrada existe
    if not os.path.exists(input_dataset_path):
        print(f"Erro: Diretório {input_dataset_path} não encontrado!")
        print("Verifique se o caminho está correto.")
        exit(1)
    
    print("=== Pipeline de Pré-processamento EXPANDIDO para Reconhecimento de Locutor ===")
    print(f"Diretório de entrada: {input_dataset_path}")
    print(f"Diretório de saída: {output_base_path}")
    print(f"NoiseReduce disponível: {NOISEREDUCE_AVAILABLE}")
    print()
    
    # Executa o processamento
    try:
        process_dataset(input_dataset_path, output_base_path)
        create_processing_summary(output_base_path)
        
        print(f"\n✅ Processamento concluído com sucesso!")
        print(f"📁 Datasets processados salvos em: {output_base_path}")
        print(f"📄 Consulte o arquivo RESUMO_PROCESSAMENTO_EXPANDIDO.md para detalhes")
        
        if not NOISEREDUCE_AVAILABLE:
            print(f"⚠️  NoiseReduce não disponível. Para instalar: pip install noisereduce")
        
    except Exception as e:
        print(f"❌ Erro durante o processamento: {e}")
        import traceback
        traceback.print_exc()