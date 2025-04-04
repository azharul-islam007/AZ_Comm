# physical_channel.py
import numpy as np
import torch
from scipy import signal
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PhysicalChannelLayer:
    """Base class for physical channel layers"""

    def __init__(self,
                 snr_db=20.0,
                 channel_type='awgn',
                 modulation='qam',
                 modulation_order=16,
                 coding_rate=0.75,
                 coding_type='repetition',
                 use_channel_coding=True,
                 importance_weighting=True,
                 enable_adaptive_modulation=True,
                 enable_unequal_error_protection=True,
                 ofdm_carriers=64,
                 fading_param=1.0):
        """Initialize the physical channel layer with parameters"""
        # Store parameters
        self.snr_db = snr_db
        self.channel_type = channel_type
        self.modulation = modulation
        self.modulation_order = modulation_order
        self.coding_rate = coding_rate
        self.coding_type = coding_type
        self.use_channel_coding = use_channel_coding
        self.importance_weighting = importance_weighting
        self.enable_adaptive_modulation = enable_adaptive_modulation
        self.enable_unequal_error_protection = enable_unequal_error_protection
        self.ofdm_carriers = ofdm_carriers
        self.fading_param = fading_param

        # Set thresholds for adaptive modulation
        self.snr_threshold_low = 15.0
        self.snr_threshold_high = 25.0

        # Channel statistics
        self.channel_stats = {
            'error_rate': 0.0,
            'estimated_snr': snr_db,
            'transmission_count': 0
        }

        # Initialize other components as needed
        self._init_channel_components()

    def _init_channel_components(self):
        """Initialize channel-specific components"""
        # Add your implementation here
        pass

    def transmit(self, embedding, importance_weights=None, debug=False):
        """
        Transmit embedding through the physical channel

        Args:
            embedding: Input embedding to transmit
            importance_weights: Optional weights for unequal error protection
            debug: Whether to return debug information

        Returns:
            Received embedding after transmission through the channel
        """
        # Convert torch tensor to numpy if needed
        original_tensor = False
        original_device = None
        if isinstance(embedding, torch.Tensor):
            original_tensor = True
            original_device = embedding.device
            embedding = embedding.detach().cpu().numpy()

        # Save original shape for reconstruction
        original_shape = embedding.shape

        # Ensure embedding is 2D (batch_size, features)
        if len(original_shape) == 1:
            # Single embedding, add batch dimension
            embedding = embedding.reshape(1, -1)

        # Check expected dimension (physical channel expects 460)
        expected_dim = 460

        # Handle dimension mismatch
        current_dim = embedding.shape[1]
        if current_dim != expected_dim:
            # Create a properly sized array
            adjusted_embedding = np.zeros((embedding.shape[0], expected_dim))
            # Copy as much data as possible
            min_dim = min(current_dim, expected_dim)
            adjusted_embedding[:, :min_dim] = embedding[:, :min_dim]
            embedding = adjusted_embedding

        # Flatten for processing
        flattened = embedding.flatten()

        # Apply importance weighting if provided
        weighted = flattened
        actual_weights = np.ones_like(flattened)

        if self.importance_weighting and importance_weights is not None:
            if isinstance(importance_weights, torch.Tensor):
                importance_weights = importance_weights.detach().cpu().numpy()

            # Reshape weights if needed
            if importance_weights.shape != flattened.shape:
                # Try to adapt the weights to match embedding shape
                if len(importance_weights.shape) == 1 and len(flattened.shape) == 2:
                    # Broadcast 1D weights to match 2D embeddings
                    importance_weights = np.tile(importance_weights, (flattened.shape[0], 1))
                elif len(importance_weights.shape) == 2 and len(flattened.shape) == 1:
                    # Use first row of 2D weights for 1D embedding
                    importance_weights = importance_weights[0]
                else:
                    # Fallback to uniform weights
                    importance_weights = np.ones_like(flattened)

            # Apply weighting
            weighted = flattened * importance_weights
            actual_weights = importance_weights

        # Convert vector to bits
        bits = self._vector_to_bits(weighted)

        # Apply channel coding
        encoded_bits = self._apply_channel_coding(bits)

        # Store original bits for error rate estimation
        original_bits = encoded_bits.copy()

        # Map bits to symbols, apply OFDM, and transmit through channel
        symbols = self._bits_to_symbols(encoded_bits)
        signal = self._apply_ofdm_modulation(symbols)
        received_signal = self._apply_channel_effects(signal)
        received_symbols = self._apply_ofdm_demodulation(received_signal)
        received_bits = self._symbols_to_bits(received_symbols)

        # Apply channel decoding
        decoded_bits = self._decode_channel_coding(received_bits)

        # Convert bits back to vector
        received_vector = self._bits_to_vector(decoded_bits, original_shape)

        # If importance weighting was applied, unapply it
        if self.importance_weighting and np.any(actual_weights != 1.0):
            # Avoid division by zero
            safe_weights = np.where(actual_weights > 1e-10, actual_weights, 1.0)
            received_vector = received_vector / safe_weights

        # Convert back to torch tensor if input was a tensor
        if original_tensor:
            received_vector = torch.tensor(received_vector, device=original_device)

        # Return the received embedding and debug info if requested
        if debug:
            debug_info = {
                'bits': bits,
                'encoded_bits': encoded_bits,
                'symbols': symbols,
                'received_symbols': received_symbols,
                'decoded_bits': decoded_bits,
                'error_rate': self._estimate_error_rate(original_bits, received_bits),
                'estimated_snr': self.channel_stats.get('estimated_snr', self.snr_db)
            }
            return received_vector, debug_info
        else:
            return received_vector

    def get_channel_info(self):
        """Return information about the physical channel"""
        return {
            'channel_type': self.channel_type,
            'modulation': f"{self.modulation}-{self.modulation_order}",
            'snr_db': self.snr_db,
            'coding_rate': self.coding_rate,
            'coding_type': self.coding_type,
            'use_channel_coding': self.use_channel_coding,
            'importance_weighting': self.importance_weighting,
            'enable_adaptive_modulation': self.enable_adaptive_modulation,
            'enable_unequal_error_protection': self.enable_unequal_error_protection,
            'error_rate': self.channel_stats.get('error_rate', 0.0)
        }


class SemanticAwarePhysicalChannel:
    """
    Advanced physical channel with semantic-aware protection strategies
    """

    def __init__(self,
                 snr_db=20.0,
                 channel_type='time_varying',
                 coding_scheme='polar',
                 adaptive_modulation=True,
                 semantic_protection=True):

        self.snr_db = snr_db
        self.channel_type = channel_type
        self.coding_scheme = coding_scheme
        self.adaptive_modulation = adaptive_modulation
        self.semantic_protection = semantic_protection

        # Channel models
        self.channel_models = {
            'awgn': self._apply_awgn,
            'rayleigh': self._apply_rayleigh,
            'rician': self._apply_rician,
            'frequency_selective': self._apply_frequency_selective,
            'time_varying': self._apply_time_varying
        }

        # Coding schemes
        self.coding_schemes = {
            'repetition': self._apply_repetition_code,
            'hamming': self._apply_hamming_code,
            'ldpc': self._apply_ldpc_code,
            'polar': self._apply_polar_code,
            'rateless': self._apply_rateless_code
        }

        # Modulation schemes with bits per symbol
        self.modulation_schemes = {
            'bpsk': 1,
            'qpsk': 2,
            'qam16': 4,
            'qam64': 6,
            'qam256': 8
        }

        # Default modulation
        self.current_modulation = 'qam16'

        # Initialize channel state
        self._init_channel_state()

    def _init_channel_state(self):
        """Initialize channel state for time-varying channels"""
        # For time-varying channel
        self.doppler_frequency = 0.01  # Normalized Doppler (0.01 = moderate mobility)
        self.coherence_time = int(0.423 / self.doppler_frequency)  # in symbols

        # For frequency-selective channel
        self.delay_spread = 5  # in symbols
        self.coherence_bandwidth = 1.0 / self.delay_spread  # normalized

        # Channel coefficient memory
        self.channel_coefs = None

        # Channel statistics
        self.channel_stats = {
            'ber': [],
            'snr_estimates': [],
            'channel_capacity': [],
            'estimated_snr': self.snr_db,
            'error_rate': 0.0,
            'transmission_count': 0
        }

    def _apply_awgn(self, signal, snr_db):
        """Apply AWGN channel effects"""
        # Calculate noise power from SNR
        signal_power = np.mean(np.abs(signal) ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))

        # Generate complex Gaussian noise
        noise = np.sqrt(noise_power / 2) * (
                np.random.randn(len(signal)) +
                1j * np.random.randn(len(signal))
        )

        # Add noise to signal
        received = signal + noise

        # Update estimated SNR
        estimated_noise_power = np.mean(np.abs(received - signal) ** 2)
        estimated_snr = 10 * np.log10(signal_power / max(estimated_noise_power, 1e-10))
        self.channel_stats['estimated_snr'] = estimated_snr

        return received

    def _apply_rayleigh(self, signal, snr_db):
        """Apply Rayleigh fading channel effects"""
        # Generate Rayleigh fading coefficients
        h = np.sqrt(1 / 2) * (
                np.random.randn(len(signal)) +
                1j * np.random.randn(len(signal))
        )

        # Apply fading to signal
        faded_signal = h * signal

        # Add AWGN
        return self._apply_awgn(faded_signal, snr_db)

    def _apply_rician(self, signal, snr_db, k_factor=4.0):
        """Apply Rician fading channel effects"""
        # K-factor: ratio of direct path power to scattered path power
        k = k_factor

        # Generate fading coefficients
        direct_path = np.sqrt(k / (k + 1))  # LOS component
        scattered_path = np.sqrt(1 / (k + 1)) * (
                np.random.randn(len(signal)) +
                1j * np.random.randn(len(signal))
        )

        h = direct_path + scattered_path

        # Apply fading to signal
        faded_signal = h * signal

        # Add AWGN
        return self._apply_awgn(faded_signal, snr_db)

    def _apply_frequency_selective(self, signal, snr_db):
        """Apply frequency selective fading channel effects"""
        # Generate multipath channel taps
        num_taps = self.delay_spread

        # Power delay profile (exponentially decaying)
        pdp = np.exp(-np.arange(num_taps) / 2)
        pdp = pdp / np.sum(pdp)  # Normalize

        # Generate complex channel taps
        h_taps = np.sqrt(pdp / 2) * (
                np.random.randn(num_taps) +
                1j * np.random.randn(num_taps)
        )

        # Apply multipath channel through convolution
        faded_signal = signal.convolve(signal, h_taps, mode='same')

        # Add AWGN
        return self._apply_awgn(faded_signal, snr_db)

    def _apply_time_varying(self, signal, snr_db):
        """Apply time-varying channel effects with realistic modeling"""
        # Generate/update channel coefficients using Jakes' model
        n_samples = len(signal)

        if self.channel_coefs is None or len(self.channel_coefs) != n_samples:
            # Need to generate new coefficients

            # Number of oscillators for Jakes' model
            n_osc = 20

            # Generate arrival angles
            theta = np.random.uniform(0, 2 * np.pi, n_osc)

            # Initialize coefficients
            t = np.arange(n_samples)
            h_real = np.zeros(n_samples)
            h_imag = np.zeros(n_samples)

            # Generate using sum of oscillators
            for i in range(n_osc):
                # Calculate Doppler shift for each path
                doppler_shift = self.doppler_frequency * np.cos(theta[i])

                # Add contribution to channel coefficients
                h_real += np.cos(2 * np.pi * doppler_shift * t + np.random.uniform(0, 2 * np.pi))
                h_imag += np.sin(2 * np.pi * doppler_shift * t + np.random.uniform(0, 2 * np.pi))

            # Normalize and combine
            h_real = h_real / np.sqrt(n_osc)
            h_imag = h_imag / np.sqrt(n_osc)

            self.channel_coefs = h_real + 1j * h_imag

        # Apply time-varying fading
        faded_signal = self.channel_coefs * signal

        # Add AWGN
        return self._apply_awgn(faded_signal, snr_db)

    def _apply_repetition_code(self, bits, rate=1 / 3):
        """Apply simple repetition code"""
        # Repeat each bit by the inverse of the rate
        repeat_count = int(1 / rate)
        coded_bits = np.repeat(bits, repeat_count)
        return coded_bits

    def _decode_repetition_code(self, bits, rate=1 / 3):
        """Decode repetition code using majority voting"""
        repeat_count = int(1 / rate)

        # Reshape bits into groups for voting
        remainder = len(bits) % repeat_count
        if remainder != 0:
            # Pad if necessary
            bits = np.append(bits, np.zeros(repeat_count - remainder))

        groups = bits.reshape(-1, repeat_count)

        # Majority vote decoding
        decoded = (np.sum(groups, axis=1) > repeat_count / 2).astype(int)

        return decoded

    def _apply_hamming_code(self, bits, code_type='7,4'):
        """Apply Hamming code"""
        # Implementation of Hamming encoder would go here
        # This is a simplified placeholder
        if code_type == '7,4':
            # Hamming(7,4) code
            # In a real implementation, apply proper encoding matrix
            padded_bits = np.pad(bits, (0, (-len(bits) % 4)), 'constant')
            coded_bits = np.repeat(padded_bits, 7 / 4)  # Simplified
            return coded_bits
        else:
            return self._apply_repetition_code(bits)  # Fallback

    def _apply_ldpc_code(self, bits, rate=1 / 2):
        """Apply LDPC code"""
        # LDPC would require a full library implementation
        # This is a simplified placeholder

        # Create a random parity-check matrix for demonstration
        n = len(bits)
        k = int(n * rate)
        m = n - k

        # Generate a random sparse parity-check matrix
        H = np.random.randint(0, 2, (m, n))
        H = H * (np.random.rand(m, n) < 0.05)  # Make it sparse

        # Simple encoding (in reality would use proper LDPC encoder)
        encoded = np.zeros(n + m, dtype=int)
        encoded[:n] = bits

        # Create parity bits (simplified)
        for i in range(m):
            encoded[n + i] = np.sum(bits * H[i, :]) % 2

        return encoded

    def _apply_polar_code(self, bits, rate=1 / 2):
        """Apply Polar code"""
        # Polar codes would require a full implementation
        # This is a simplified placeholder showing the concept

        # Determine N (power of 2)
        k = len(bits)
        n = 1
        while n < k:
            n *= 2

        N = n * 2  # Code length

        # Pad input if needed
        if k < n:
            padded_bits = np.pad(bits, (0, n - k), 'constant')
        else:
            padded_bits = bits[:n]

        # In a real implementation, we would:
        # 1. Choose the most reliable bit positions
        # 2. Apply the polar transform recursively
        # 3. Return the encoded codeword

        # Simplified placeholder
        coded_bits = np.zeros(N, dtype=int)
        coded_bits[::2] = padded_bits  # Information bits
        coded_bits[1::2] = np.cumsum(padded_bits) % 2  # Simple "polar-like" transform

        return coded_bits

    def _apply_rateless_code(self, bits, overhead=0.1):
        """Apply a rateless (fountain) code like LT codes"""
        # Rateless codes would require a full implementation
        # This is a simplified placeholder

        k = len(bits)
        n = int(k * (1 + overhead))

        # Create a random generator matrix
        # In a real implementation, this would follow specific degree distributions
        G = np.random.randint(0, 2, (n, k))

        # Apply generator matrix
        coded_bits = np.zeros(n, dtype=int)
        for i in range(n):
            # XOR the selected input bits
            selected = bits[G[i, :] == 1]
            coded_bits[i] = np.sum(selected) % 2

        return coded_bits

    def _select_optimal_modulation(self, snr_db):
        """Select optimal modulation based on SNR"""
        if snr_db < 6:
            return 'bpsk'  # Most robust
        elif snr_db < 12:
            return 'qpsk'
        elif snr_db < 18:
            return 'qam16'
        elif snr_db < 24:
            return 'qam64'
        else:
            return 'qam256'  # Highest capacity

    def _bits_to_symbols(self, bits, modulation=None):
        """Convert bits to symbols using the specified modulation"""
        if modulation is None:
            modulation = self.current_modulation

        if modulation == 'bpsk':
            # BPSK: 0 -> -1, 1 -> 1
            return 2 * bits.astype(float) - 1

        elif modulation == 'qpsk':
            # QPSK: 2 bits per symbol
            # Reshape to groups of 2 bits
            padded_bits = np.pad(bits, (0, (-len(bits) % 2)), 'constant')
            bit_groups = padded_bits.reshape(-1, 2)

            # Convert to complex symbols
            symbols = np.zeros(len(bit_groups), dtype=complex)
            for i, group in enumerate(bit_groups):
                if np.array_equal(group, [0, 0]):
                    symbols[i] = complex(1, 1) / np.sqrt(2)
                elif np.array_equal(group, [0, 1]):
                    symbols[i] = complex(1, -1) / np.sqrt(2)
                elif np.array_equal(group, [1, 0]):
                    symbols[i] = complex(-1, 1) / np.sqrt(2)
                else:  # [1, 1]
                    symbols[i] = complex(-1, -1) / np.sqrt(2)

            return symbols

        elif modulation == 'qam16':
            # 16-QAM: 4 bits per symbol
            # Implementation would construct proper 16-QAM constellation
            # Simplified version:
            padded_bits = np.pad(bits, (0, (-len(bits) % 4)), 'constant')
            bit_groups = padded_bits.reshape(-1, 4)

            # Convert first two bits to real part, last two to imaginary
            real_parts = 2 * (bit_groups[:, 0] * 2 + bit_groups[:, 1]) - 3
            imag_parts = 2 * (bit_groups[:, 2] * 2 + bit_groups[:, 3]) - 3

            # Normalize energy
            symbols = (real_parts + 1j * imag_parts) / np.sqrt(10)
            return symbols

        else:
            # Default: treat as QPSK for unimplemented schemes
            return self._bits_to_symbols(bits, 'qpsk')

    def _symbols_to_bits(self, symbols, modulation=None):
        """Convert symbols back to bits"""
        if modulation is None:
            modulation = self.current_modulation

        if modulation == 'bpsk':
            # BPSK: negative -> 0, positive -> 1
            return (np.real(symbols) > 0).astype(int)

        elif modulation == 'qpsk':
            # QPSK: 2 bits per symbol
            bits = np.zeros(len(symbols) * 2, dtype=int)

            for i, symbol in enumerate(symbols):
                # Decision based on quadrant
                real_part = np.real(symbol)
                imag_part = np.imag(symbol)

                # First bit based on real part
                bits[2 * i] = 0 if real_part > 0 else 1

                # Second bit based on imaginary part
                bits[2 * i + 1] = 0 if imag_part > 0 else 1

            return bits

        elif modulation == 'qam16':
            # 16-QAM: 4 bits per symbol
            bits = np.zeros(len(symbols) * 4, dtype=int)

            for i, symbol in enumerate(symbols):
                # Scale to correct range
                real_part = np.round(np.real(symbol) * np.sqrt(10))
                imag_part = np.round(np.imag(symbol) * np.sqrt(10))

                # Map to bit patterns (simplified)
                real_idx = int((real_part + 3) / 2)
                imag_idx = int((imag_part + 3) / 2)

                # Clamp to valid range
                real_idx = max(0, min(3, real_idx))
                imag_idx = max(0, min(3, imag_idx))

                # Extract bit patterns
                real_bits = [real_idx // 2, real_idx % 2]
                imag_bits = [imag_idx // 2, imag_idx % 2]

                # Assign bits
                bits[4 * i:4 * i + 2] = real_bits
                bits[4 * i + 2:4 * i + 4] = imag_bits

            return bits

        else:
            # Default for unimplemented schemes
            return self._symbols_to_bits(symbols, 'qpsk')

    def _apply_semantic_protection(self, bits, importance_profile):
        """Apply unequal error protection based on semantic importance"""
        if not self.semantic_protection or importance_profile is None:
            return self._apply_coding(bits)

        # Determine number of protection levels (simplifying to 3 levels)
        n_bits = len(bits)

        # Rescale importance profile and quantize to 3 levels
        if len(importance_profile) > n_bits:
            importance_profile = importance_profile[:n_bits]
        elif len(importance_profile) < n_bits:
            # Repeat or interpolate profile
            ratio = n_bits / len(importance_profile)
            importance_profile = np.repeat(importance_profile, ratio)[:n_bits]

        # Normalize to [0, 1]
        scaled_profile = (importance_profile - np.min(importance_profile)) / (
                np.max(importance_profile) - np.min(importance_profile) + 1e-10)

        # Quantize to 3 levels
        protection_levels = np.zeros(n_bits, dtype=int)
        protection_levels[scaled_profile > 0.66] = 2  # High protection
        protection_levels[(scaled_profile > 0.33) & (scaled_profile <= 0.66)] = 1  # Medium

        # Apply different coding rates based on protection level
        coded_bits = []

        # High protection segments
        high_mask = protection_levels == 2
        high_bits = bits[high_mask]
        if len(high_bits) > 0:
            high_coded = self._apply_coding(high_bits, rate=1 / 3)
            coded_bits.append(high_coded)

        # Medium protection segments
        medium_mask = protection_levels == 1
        medium_bits = bits[medium_mask]
        if len(medium_bits) > 0:
            medium_coded = self._apply_coding(medium_bits, rate=1 / 2)
            coded_bits.append(medium_coded)

        # Low protection segments
        low_mask = protection_levels == 0
        low_bits = bits[low_mask]
        if len(low_bits) > 0:
            low_coded = self._apply_coding(low_bits, rate=3 / 4)
            coded_bits.append(low_coded)

        # Combine all coded segments
        return np.concatenate(coded_bits)

    def _apply_coding(self, bits, rate=1 / 2):
        """Apply the selected coding scheme"""
        coding_func = self.coding_schemes.get(
            self.coding_scheme,
            self._apply_repetition_code  # Default
        )

        return coding_func(bits, rate)

    def _estimate_error_rate(self, original_bits, received_bits):
        """Estimate bit error rate from sample comparison"""
        min_len = min(len(original_bits), len(received_bits))
        if min_len == 0:
            return 0.0

        errors = np.sum(original_bits[:min_len] != received_bits[:min_len])
        error_rate = errors / min_len

        # Update channel statistics
        self.channel_stats['error_rate'] = 0.9 * self.channel_stats['error_rate'] + 0.1 * error_rate

        return error_rate

    def transmit(self, embedding, importance_profile=None, debug=False):
        """
        Transmit embedding through the channel with semantic protection.

        Args:
            embedding: Embedding vector to transmit
            importance_profile: Optional profile indicating semantic importance
            debug: Whether to return debug information

        Returns:
            Received embedding after channel effects
        """
        # Ensure embedding is a numpy array
        if isinstance(embedding, torch.Tensor):
            embedding = embedding.detach().cpu().numpy()

        # Record original shape and type
        original_shape = embedding.shape
        original_is_1d = len(original_shape) == 1

        # Convert 1D array to 2D with a batch dimension
        if original_is_1d:
            embedding = embedding.reshape(1, -1)

        # Handle dimension mismatch - we need feature dim to be exactly 460
        current_feature_dim = embedding.shape[1]
        expected_feature_dim = 460

        if current_feature_dim != expected_feature_dim:
            # Create properly sized array
            adjusted_embedding = np.zeros((embedding.shape[0], expected_feature_dim))
            # Copy as much as we can from the original
            copy_dim = min(current_feature_dim, expected_feature_dim)
            adjusted_embedding[:, :copy_dim] = embedding[:, :copy_dim]
            embedding = adjusted_embedding

        # Convert embeddings to bits
        # First normalize to quantization range
        quantization_levels = 256
        quantized = np.round((embedding + 1) * (quantization_levels / 2)).astype(int)
        quantized = np.clip(quantized, 0, quantization_levels - 1)

        # Convert to bits (8 bits per value for 256 levels)
        bits = np.unpackbits(quantized.astype(np.uint8))

        # Apply semantic-aware coding if enabled
        if self.semantic_protection and importance_profile is not None:
            coded_bits = self._apply_semantic_protection(bits, importance_profile)
        else:
            coded_bits = self._apply_coding(bits)

        # Choose modulation scheme based on SNR if adaptive
        if self.adaptive_modulation:
            self.current_modulation = self._select_optimal_modulation(self.snr_db)

        # Convert bits to symbols
        symbols = self._bits_to_symbols(coded_bits, self.current_modulation)

        # Apply channel effects
        channel_func = self.channel_models.get(
            self.channel_type,
            self._apply_awgn  # Default to AWGN
        )

        received_symbols = channel_func(symbols, self.snr_db)

        # Convert received symbols back to bits
        received_bits = self._symbols_to_bits(received_symbols, self.current_modulation)

        # Decode channel coding
        if self.coding_scheme == 'repetition':
            decoded_bits = self._decode_repetition_code(received_bits)
        else:
            # For other coding schemes, use appropriate decoders
            # Here we just take the first bits
            decoded_bits = received_bits[:len(bits)]

        # Store original bits for error rate estimation
        original_bits = coded_bits.copy()

        # Calculate error rate
        error_rate = self._estimate_error_rate(original_bits, received_bits)

        # Convert bits back to values
        # Reshape to 8 bits per value
        padded_length = ((len(decoded_bits) + 7) // 8) * 8
        if len(decoded_bits) < padded_length:
            decoded_bits = np.pad(decoded_bits, (0, padded_length - len(decoded_bits)), 'constant')

        # Convert bit groups to values
        decoded_values = np.packbits(decoded_bits.reshape(-1, 8))

        # Convert back to float range
        float_values = (decoded_values / (quantization_levels / 2)) - 1

        # Reshape to match embedding dimensions
        received_embedding = float_values[:embedding.size].reshape(embedding.shape)

        # Restore original shape if needed
        if original_is_1d:
            # If original was 1D, remove batch dimension
            received_embedding = received_embedding.squeeze(0)

            # If dimensions were expanded, trim to original size
            if received_embedding.size > original_shape[0]:
                received_embedding = received_embedding[:original_shape[0]]

        # Update channel statistics
        self.channel_stats['transmission_count'] += 1
        self.channel_stats['error_rate'] = 0.9 * self.channel_stats.get('error_rate', 0) + 0.1 * error_rate

        # Return received embedding and debug info if requested
        if debug:
            debug_info = {
                'snr_db': self.snr_db,
                'modulation': self.current_modulation,
                'channel_type': self.channel_type,
                'coding_scheme': self.coding_scheme,
                'error_rate': error_rate,
                'symbols_transmitted': len(symbols),
                'bits_transmitted': len(coded_bits),
                'estimated_snr': self.channel_stats.get('estimated_snr', self.snr_db)
            }
            return received_embedding, debug_info

        return received_embedding

    def get_channel_info(self):
        """Return information about channel configuration and statistics"""
        return {
            'channel_type': self.channel_type,
            'modulation': f"{self.current_modulation}",
            'snr_db': self.snr_db,
            'estimated_snr': self.channel_stats['estimated_snr'],
            'coding_scheme': self.coding_scheme,
            'coding_rate': self.coding_rate if hasattr(self, 'coding_rate') else 0.5,
            'adaptive_modulation': self.adaptive_modulation,
            'semantic_protection': self.semantic_protection,
            'error_rate': self.channel_stats['error_rate']
        }


# Backwards compatibility for existing code
class ContentAdaptivePhysicalChannel(SemanticAwarePhysicalChannel):
    """Alias class for backwards compatibility"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
