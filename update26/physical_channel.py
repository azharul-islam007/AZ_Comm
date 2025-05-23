import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import torch
import logging
import os

# Configure logging
logger = logging.getLogger(__name__)


class PhysicalChannelLayer:
    """
    Enhanced Physical channel layer for semantic communication systems.
    Implements adaptive modulation & coding and advanced error protection.
    """

    def __init__(self,
                 snr_db=20.0,
                 channel_type='awgn',
                 modulation='qam',
                 modulation_order=16,
                 fading_param=1.0,
                 coding_rate=0.75,
                 coding_type='repetition',
                 use_channel_coding=True,
                 block_size=1024,
                 importance_weighting=True,
                 enable_adaptive_modulation=True,
                 enable_unequal_error_protection=True,
                 ofdm_carriers=64):

        self.snr_db = snr_db
        self.channel_type = channel_type
        self.modulation = modulation
        self.modulation_order = modulation_order
        self.fading_param = fading_param
        self.coding_rate = coding_rate
        self.coding_type = coding_type
        self.use_channel_coding = use_channel_coding
        self.block_size = block_size
        self.importance_weighting = importance_weighting
        self.ofdm_carriers = ofdm_carriers

        # New adaptive features
        self.enable_adaptive_modulation = enable_adaptive_modulation
        self.enable_unequal_error_protection = enable_unequal_error_protection
        self.snr_threshold_high = 25.0
        self.snr_threshold_low = 15.0
        self.protection_levels = 3

        # Track channel statistics
        self.channel_stats = {
            'estimated_snr': snr_db,
            'error_rate': 0.0,
            'transmission_count': 0
        }

        # Calculate bits per symbol based on modulation order
        self.bits_per_symbol = int(np.log2(modulation_order))

        # Initialize constellation mapper
        self._init_constellation()

        # Create directory for channel state information if not exists
        os.makedirs('./channel_data', exist_ok=True)

        logger.info(f"Initialized Enhanced Physical Channel with {channel_type} channel, "
                    f"{modulation}-{modulation_order} modulation, SNR={snr_db}dB")

        if enable_adaptive_modulation:
            logger.info(
                f"Adaptive modulation enabled: thresholds at {self.snr_threshold_low}dB and {self.snr_threshold_high}dB")

        if enable_unequal_error_protection:
            logger.info(f"Unequal error protection enabled with {self.protection_levels} protection levels")

    def _init_constellation(self):
        """Initialize the constellation mapping based on modulation scheme."""
        if self.modulation == 'qam':
            # QAM constellation (square)
            m = int(np.sqrt(self.modulation_order))
            real_parts = np.linspace(-1, 1, m)
            imag_parts = np.linspace(-1, 1, m)
            self.constellation = np.array([(r + 1j * i) for r in real_parts for i in imag_parts])

            # Normalize constellation to unit energy
            energy = np.mean(np.abs(self.constellation) ** 2)
            self.constellation /= np.sqrt(energy)

        elif self.modulation == 'psk':
            # PSK constellation (circle)
            angles = np.linspace(0, 2 * np.pi, self.modulation_order, endpoint=False)
            self.constellation = np.exp(1j * angles)

        else:  # OFDM uses QAM per subcarrier
            m = int(np.sqrt(self.modulation_order))
            real_parts = np.linspace(-1, 1, m)
            imag_parts = np.linspace(-1, 1, m)
            self.constellation = np.array([(r + 1j * i) for r in real_parts for i in imag_parts])
            energy = np.mean(np.abs(self.constellation) ** 2)
            self.constellation /= np.sqrt(energy)

    def _calculate_crc(self, bits):
        """Calculate simple CRC-16 for frame integrity checking"""
        import binascii
        # Convert bits to bytes for CRC calculation
        if len(bits) % 8 != 0:
            # Pad to byte boundary
            padding = 8 - (len(bits) % 8)
            bits = np.concatenate([bits, np.zeros(padding, dtype=int)])

        # Convert to bytes
        bytes_data = []
        for i in range(0, len(bits), 8):
            byte_val = 0
            for j in range(8):
                if i + j < len(bits):
                    byte_val |= int(bits[i + j]) << (7 - j)
            bytes_data.append(byte_val)

        # Calculate CRC
        try:
            crc = binascii.crc32(bytes(bytes_data)) & 0xFFFF
            return crc
        except:
            return 0

    def _check_frame_integrity(self, original_bits, received_bits, importance_weights=None):
        """Check frame integrity and identify corruption in salient regions"""
        # Calculate CRCs
        orig_crc = self._calculate_crc(original_bits)
        recv_crc = self._calculate_crc(received_bits)

        frame_ok = (orig_crc == recv_crc)

        # Even if CRC passes, check for semantic anchor corruption
        semantic_corruption = False
        if importance_weights is not None and len(importance_weights) > 0:
            # Map importance weights to bit indices
            bits_per_dim = len(original_bits) // len(importance_weights)

            # Find high-importance regions (top 25%)
            threshold = np.percentile(importance_weights, 75)
            critical_indices = np.where(importance_weights >= threshold)[0]

            # Check for errors in critical regions
            critical_errors = 0
            total_critical_bits = 0

            for dim_idx in critical_indices:
                start_bit = dim_idx * bits_per_dim
                end_bit = min((dim_idx + 1) * bits_per_dim, len(original_bits))

                if end_bit <= len(received_bits):
                    dim_errors = np.sum(original_bits[start_bit:end_bit] != received_bits[start_bit:end_bit])
                    critical_errors += dim_errors
                    total_critical_bits += (end_bit - start_bit)

            # Flag semantic corruption if >5% of critical bits are wrong
            if total_critical_bits > 0:
                semantic_corruption = (critical_errors / total_critical_bits) > 0.05

        return frame_ok, semantic_corruption

    def _smart_arq_decision(self, frame_ok, semantic_corruption, content_type="unknown"):
        """Decide whether to trigger Smart-ARQ retransmission"""
        # Always retransmit if frame CRC failed
        if not frame_ok:
            return True, "crc_failure"

        # For critical content types, also retransmit on semantic corruption
        critical_content = content_type in ["procedural", "legislative"]

        if semantic_corruption and critical_content:
            return True, "semantic_anchor_corruption"

        # Check if we should do probabilistic retransmission for high corruption
        if semantic_corruption and np.random.random() < 0.3:  # 30% chance
            return True, "probabilistic_semantic"

        return False, "no_retransmission"

    def adapt_to_channel_conditions(self, estimated_snr=None):
        """
        ENHANCED: Adapt modulation and coding based on channel conditions with more
        aggressive thresholds for high-corruption scenarios.
        """
        if not self.enable_adaptive_modulation:
            return

        # Use provided SNR or current estimate
        snr = estimated_snr if estimated_snr is not None else self.channel_stats['estimated_snr']

        # Get thresholds from config or use default values
        from config_manager import ConfigManager
        config = ConfigManager()

        # More aggressive thresholds (fall back to robust modulations sooner)
        qam64_threshold = config.get("physical.mod_thresholds.64QAM", 25.0)
        qam16_threshold = config.get("physical.mod_thresholds.16QAM", 18.0)  # Lower from 22dB
        qpsk_threshold = config.get("physical.mod_thresholds.QPSK", 10.0)  # Lower from 12dB

        # Dynamic FEC rate selection based on SNR buckets
        if snr >= 20.0:
            new_coding_rate = config.get("physical.fec_buckets.>=20", 0.8)
        elif snr >= 15.0:
            new_coding_rate = config.get("physical.fec_buckets.15-20", 0.6)
        elif snr >= 10.0:
            new_coding_rate = config.get("physical.fec_buckets.10-15", 0.4)
        else:
            new_coding_rate = config.get("physical.fec_buckets.<10", 0.2)  # Max protection

        # Adapt modulation order with more aggressive thresholds
        if snr > qam64_threshold:  # Excellent conditions
            new_modulation_order = 64
            logger.info(
                f"Channel conditions excellent (SNR: {snr:.1f}dB): Using 64-QAM with coding rate {new_coding_rate}")
        elif snr > qam16_threshold:  # Good conditions
            new_modulation_order = 16
            logger.info(f"Channel conditions good (SNR: {snr:.1f}dB): Using 16-QAM with coding rate {new_coding_rate}")
        elif snr > qpsk_threshold:  # Fair conditions
            new_modulation_order = 4  # QPSK
            logger.info(f"Channel conditions fair (SNR: {snr:.1f}dB): Using QPSK with coding rate {new_coding_rate}")
        else:  # Poor conditions - use BPSK
            new_modulation_order = 2  # BPSK for worst case
            logger.info(f"Channel conditions poor (SNR: {snr:.1f}dB): Using BPSK with coding rate {new_coding_rate}")

        # Update parameters
        if self.modulation_order != new_modulation_order:
            self.modulation_order = new_modulation_order
            self.bits_per_symbol = int(np.log2(new_modulation_order))
            self._init_constellation()

        self.coding_rate = new_coding_rate

    def _vector_to_bits(self, vector):
        """Convert a continuous vector to bits based on quantization."""
        # Normalize vector to [-1, 1]
        vec_min, vec_max = vector.min(), vector.max()
        if vec_max > vec_min:  # Avoid division by zero
            normalized = -1 + 2 * (vector - vec_min) / (vec_max - vec_min)
        else:
            normalized = np.zeros_like(vector)

        # Calculate number of levels and quantize
        levels = 2 ** self.bits_per_symbol
        quantized = np.floor((normalized + 1) / 2 * (levels - 1)).astype(int)

        # Convert to bits
        bits = []
        for q in quantized:
            # Convert integer to binary string of the right length, then to list of integers
            bits.extend([int(b) for b in format(q, f'0{self.bits_per_symbol}b')])

        return np.array(bits)

    def _bits_to_vector(self, bits, original_shape):
        """Convert bits back to a continuous vector through dequantization."""
        # Calculate number of values in the original vector
        num_values = np.prod(original_shape)

        # Reshape bits into groups and convert to integers
        bit_groups = bits.reshape(-1, self.bits_per_symbol)
        values = []
        for group in bit_groups:
            # Convert bit group to integer
            value = int(''.join(map(str, group)), 2)
            values.append(value)

        # Truncate or pad to match original shape
        values = values[:num_values]
        if len(values) < num_values:
            values.extend([0] * (num_values - len(values)))

        # Dequantize to continuous values
        levels = 2 ** self.bits_per_symbol
        normalized = np.array(values) / (levels - 1) * 2 - 1

        # Reshape to original shape
        return normalized.reshape(original_shape)

    def _apply_channel_coding(self, bits, protection_level=1):
        """
        Enhanced: Apply forward error correction with support for unequal protection.

        Args:
            bits: Input bits to encode
            protection_level: Protection level (1-3, where 3 is highest protection)
        """
        if not self.use_channel_coding:
            return bits

        # Adjust coding parameters based on protection level if unequal protection is enabled
        if self.enable_unequal_error_protection:
            if protection_level == 3:  # High protection
                effective_coding_rate = max(0.3, self.coding_rate - 0.2)
            elif protection_level == 2:  # Medium protection
                effective_coding_rate = self.coding_rate
            else:  # Low protection
                effective_coding_rate = min(0.9, self.coding_rate + 0.1)
        else:
            effective_coding_rate = self.coding_rate

        # Select coding scheme based on coding_type
        if self.coding_type == 'repetition':
            # Simple repetition code
            repetition = int(1 / effective_coding_rate)
            encoded = np.repeat(bits, repetition)

            # Ensure the encoded bits have the correct length
            target_length = len(bits)
            if len(encoded) < target_length:
                encoded = np.pad(encoded, (0, target_length - len(encoded)), 'constant')
            else:
                encoded = encoded[:target_length]

        elif self.coding_type == 'ldpc':
            # Placeholder for LDPC coding (would require an actual LDPC implementation)
            # For now, we'll use a more advanced repetition scheme as a placeholder
            k = int(len(bits) * effective_coding_rate)
            encoded = np.zeros(len(bits), dtype=int)

            # Encode most important bits with more repetition
            high_prot_bits = bits[:k // 3]
            med_prot_bits = bits[k // 3:2 * k // 3]
            low_prot_bits = bits[2 * k // 3:k]

            # Apply different levels of repetition
            encoded[:len(high_prot_bits) * 3] = np.repeat(high_prot_bits, 3)
            encoded[len(high_prot_bits) * 3:len(high_prot_bits) * 3 + len(med_prot_bits) * 2] = np.repeat(med_prot_bits,
                                                                                                          2)
            encoded[len(high_prot_bits) * 3 + len(med_prot_bits) * 2:k] = low_prot_bits

            # Fill the rest with original bits
            if k < len(bits):
                encoded[k:] = bits[k:]

        elif self.coding_type == 'turbo':
            # Placeholder for Turbo coding
            # Similar placeholder as LDPC for now
            encoded = self._apply_channel_coding(bits, protection_level)  # Reuse repetition for now

        else:
            # Unknown coding type, fall back to repetition
            encoded = self._apply_channel_coding(bits, protection_level)

        return encoded

    def _decode_channel_coding(self, bits):
        """
        Enhanced: Decode forward error correction.

        Args:
            bits: Received bits to decode
        """
        if not self.use_channel_coding:
            return bits

        # Basic decoding for repetition code
        if self.coding_type == 'repetition':
            repetition = int(1 / self.coding_rate)

            # Reshape to get each group of repeated bits
            if repetition > 1 and len(bits) >= repetition:
                # Reshape safely accounting for possible truncation
                num_complete_groups = len(bits) // repetition
                reshaped = bits[:num_complete_groups * repetition].reshape(-1, repetition)

                # Majority vote for each group
                decoded = np.array([1 if np.sum(group) > repetition / 2 else 0 for group in reshaped])

                # Append any remaining bits
                if len(bits) > num_complete_groups * repetition:
                    remaining = bits[num_complete_groups * repetition:]
                    decoded = np.concatenate([decoded, remaining])
            else:
                decoded = bits

        elif self.coding_type in ['ldpc', 'turbo']:
            # Placeholder for more advanced decoders
            # For now, we'll use a basic scheme similar to repetition
            decoded = self._decode_channel_coding(bits)  # Recursively use repetition decoder

        else:
            # Unknown coding type, fall back to simple decoding
            decoded = bits

        return decoded

    def _bits_to_symbols(self, bits):
        """Map bits to complex symbols from the constellation."""
        # Group bits into chunks of bits_per_symbol
        bit_groups = [bits[i:i + self.bits_per_symbol] for i in range(0, len(bits), self.bits_per_symbol)]

        # If the last group is incomplete, pad with zeros
        if len(bit_groups[-1]) < self.bits_per_symbol:
            bit_groups[-1] = np.pad(bit_groups[-1], (0, self.bits_per_symbol - len(bit_groups[-1])), 'constant')

        # Convert each group to integer to index the constellation
        symbols = []
        for group in bit_groups:
            idx = int(''.join(map(str, group)), 2)
            symbols.append(self.constellation[idx % len(self.constellation)])

        return np.array(symbols)

    def _symbols_to_bits(self, symbols):
        """Map complex symbols back to bits."""
        bits = []

        # For each symbol, find the closest constellation point
        for s in symbols:
            distances = np.abs(self.constellation - s)
            idx = np.argmin(distances)

            # Convert index to binary representation
            bit_string = format(idx, f'0{self.bits_per_symbol}b')
            bits.extend([int(b) for b in bit_string])

        return np.array(bits)

    def _apply_ofdm_modulation(self, symbols):
        """Apply OFDM modulation to the symbols."""
        # No changes to OFDM implementation
        # If not using OFDM, just return the symbols
        if self.modulation != 'ofdm':
            return symbols

        # Reshape symbols into OFDM symbols
        num_symbols = len(symbols)
        num_ofdm_symbols = int(np.ceil(num_symbols / self.ofdm_carriers))

        # Pad to fit a whole number of OFDM symbols
        padded_symbols = np.pad(symbols, (0, num_ofdm_symbols * self.ofdm_carriers - num_symbols), 'constant')

        # Reshape and apply IFFT
        ofdm_data = padded_symbols.reshape(num_ofdm_symbols, self.ofdm_carriers)
        time_signal = np.fft.ifft(ofdm_data, axis=1)

        # Add cyclic prefix (25% of symbol length)
        cp_len = self.ofdm_carriers // 4
        ofdm_signal = np.zeros((num_ofdm_symbols, self.ofdm_carriers + cp_len), dtype=complex)

        for i in range(num_ofdm_symbols):
            # Copy the end part to the beginning (cyclic prefix)
            ofdm_signal[i, :cp_len] = time_signal[i, -cp_len:]
            # Copy the rest of the symbol
            ofdm_signal[i, cp_len:] = time_signal[i, :]

        # Flatten the signal
        return ofdm_signal.flatten()

    def _apply_ofdm_demodulation(self, received_signal):
        """Apply OFDM demodulation to the received signal."""
        # No changes to OFDM demodulation
        # If not using OFDM, just return the signal
        if self.modulation != 'ofdm':
            return received_signal

        # Parameters
        cp_len = self.ofdm_carriers // 4
        symbol_len = self.ofdm_carriers + cp_len

        # Calculate the number of complete OFDM symbols
        num_ofdm_symbols = len(received_signal) // symbol_len

        # Reshape to separate the OFDM symbols
        ofdm_signal = received_signal[:num_ofdm_symbols * symbol_len].reshape(num_ofdm_symbols, symbol_len)

        # Remove cyclic prefix and extract data
        time_signal = np.zeros((num_ofdm_symbols, self.ofdm_carriers), dtype=complex)
        for i in range(num_ofdm_symbols):
            time_signal[i] = ofdm_signal[i, cp_len:]

        # Apply FFT to get back to frequency domain
        freq_domain = np.fft.fft(time_signal, axis=1)

        # Flatten and return
        return freq_domain.flatten()

    def _apply_channel_effects(self, signal):
        """Apply channel effects (AWGN, fading, etc.) to the transmitted signal."""
        # Calculate noise power from SNR
        signal_power = np.mean(np.abs(signal) ** 2)
        noise_power = signal_power / (10 ** (self.snr_db / 10))

        # Generate complex Gaussian noise
        noise = np.sqrt(noise_power / 2) * (np.random.randn(len(signal)) + 1j * np.random.randn(len(signal)))

        # Apply channel effects based on channel type - no changes to these implementations
        if self.channel_type == 'awgn':
            # Just add noise for AWGN channel
            received = signal + noise

        elif self.channel_type == 'rayleigh':
            # Generate Rayleigh fading coefficients
            h = np.sqrt(self.fading_param / 2) * (np.random.randn(len(signal)) + 1j * np.random.randn(len(signal)))

            # Apply fading and add noise
            received = h * signal + noise

        elif self.channel_type == 'rician':
            # Generate Rician fading with K-factor (fading_param)
            K = self.fading_param

            # LOS component (deterministic)
            los = np.sqrt(K / (K + 1))

            # NLOS component (random)
            nlos_var = 1 / (K + 1)
            nlos = np.sqrt(nlos_var / 2) * (np.random.randn(len(signal)) + 1j * np.random.randn(len(signal)))

            # Combined channel
            h = los + nlos

            # Apply fading and add noise
            received = h * signal + noise

        elif self.channel_type == 'frequency_selective':
            # Apply multipath with delay spread
            num_taps = 5  # Number of channel taps

            # Generate random channel taps
            taps = np.sqrt(1 / num_taps) * (np.random.randn(num_taps) + 1j * np.random.randn(num_taps))

            # Apply multipath channel through convolution
            # Use scipy.signal.convolve instead of expecting signal to be the module
            from scipy import signal as sp_signal
            received = sp_signal.convolve(signal, taps, mode='same') + noise

        else:
            # Unknown channel type, default to AWGN
            logger.warning(f"Unknown channel type: {self.channel_type}. Using AWGN instead.")
            received = signal + noise

        # NEW: Estimate SNR from received signal for adaptive modulation
        estimated_noise_power = np.mean(np.abs(received - signal) ** 2)
        estimated_snr = 10 * np.log10(signal_power / max(estimated_noise_power, 1e-10))

        # Update channel statistics
        self.channel_stats['estimated_snr'] = estimated_snr
        self.channel_stats['transmission_count'] += 1

        # Store some channel state information periodically
        if self.channel_stats['transmission_count'] % 50 == 0:
            self._save_channel_state()

        return received

    def _save_channel_state(self):
        """
        NEW: Save channel state information for analysis and training.
        """
        try:
            filename = f"./channel_data/channel_state_{self.channel_stats['transmission_count']}.npz"
            np.savez(
                filename,
                estimated_snr=self.channel_stats['estimated_snr'],
                error_rate=self.channel_stats['error_rate'],
                modulation_order=self.modulation_order,
                coding_rate=self.coding_rate,
                channel_type=self.channel_type
            )
            logger.debug(f"Saved channel state to {filename}")
        except Exception as e:
            logger.warning(f"Failed to save channel state: {e}")

    def _estimate_error_rate(self, original_bits, received_bits):
        """
        NEW: Estimate bit error rate from a sample comparison.
        """
        min_len = min(len(original_bits), len(received_bits))
        if min_len == 0:
            return 0.0

        errors = np.sum(original_bits[:min_len] != received_bits[:min_len])
        error_rate = errors / min_len

        # Update channel statistics
        self.channel_stats['error_rate'] = 0.9 * self.channel_stats['error_rate'] + 0.1 * error_rate

        return error_rate

    def transmit(self, embedding, importance_weights=None, debug=False):
        """
        Enhanced: Transmit a semantic embedding through the physical channel.
        Now supports adaptive modulation and unequal error protection.

        Args:
            embedding: Numpy array or torch tensor containing the semantic embedding
            importance_weights: Optional weights to prioritize important dimensions
            debug: If True, plot constellation diagrams and return extra info

        Returns:
            Received embedding after transmission through the physical channel
        """
        # Adapt to current channel conditions if adaptive modulation is enabled
        if self.enable_adaptive_modulation:
            self.adapt_to_channel_conditions()

        # Convert torch tensor to numpy if needed
        if isinstance(embedding, torch.Tensor):
            embedding = embedding.detach().cpu().numpy()

        # Save original shape for reconstruction
        original_shape = embedding.shape
        flattened = embedding.flatten()

        # Apply importance weighting if provided
        weighted = flattened
        actual_weights = np.ones_like(flattened)

        if self.importance_weighting and importance_weights is not None:
            if isinstance(importance_weights, torch.Tensor):
                importance_weights = importance_weights.detach().cpu().numpy()

            # Reshape weights if needed
            if importance_weights.shape != flattened.shape:
                importance_weights = np.ones_like(flattened)

            # Apply weighting
            weighted = flattened * importance_weights
            actual_weights = importance_weights

        # Convert vector to bits
        bits = self._vector_to_bits(weighted)

        # Split bits into protection groups if using unequal error protection
        if self.enable_unequal_error_protection and len(bits) > 0:
            # Determine protection levels based on importance weights
            if self.importance_weighting and importance_weights is not None:
                # Flatten and normalize weights to 0-1 range for easy thresholding
                flat_weights = importance_weights.flatten()
                if len(flat_weights) > 0:  # Check if we have weights
                    weight_min, weight_max = flat_weights.min(), flat_weights.max()
                    if weight_max > weight_min:  # Avoid division by zero
                        norm_weights = (flat_weights - weight_min) / (weight_max - weight_min)
                    else:
                        norm_weights = np.zeros_like(flat_weights)

                    # Determine how to divide dimensions into protection groups
                    third_point = 2 / 3
                    two_third_point = 1 / 3

                    # Assign protection levels based on importance
                    high_prot_indices = np.where(norm_weights >= third_point)[0]
                    med_prot_indices = np.where((norm_weights < third_point) & (norm_weights >= two_third_point))[0]
                    low_prot_indices = np.where(norm_weights < two_third_point)[0]

                    # Apply different protection levels to bits
                    # Since we can't directly map dimension indices to bit indices easily,
                    # we'll use a simplified approach based on bit position
                    total_bit_len = len(bits)
                    high_prot_len = int(total_bit_len * len(high_prot_indices) / len(norm_weights)) if len(
                        norm_weights) > 0 else 0
                    med_prot_len = int(total_bit_len * len(med_prot_indices) / len(norm_weights)) if len(
                        norm_weights) > 0 else 0

                    # Apply channel coding to each segment
                    encoded_high = self._apply_channel_coding(bits[:high_prot_len], 3)
                    encoded_med = self._apply_channel_coding(bits[high_prot_len:high_prot_len + med_prot_len], 2)
                    encoded_low = self._apply_channel_coding(bits[high_prot_len + med_prot_len:], 1)

                    # Recombine encoded bits
                    encoded_bits = np.concatenate([encoded_high, encoded_med, encoded_low])
                else:
                    # Fall back to uniform protection if no valid weights
                    encoded_bits = self._apply_channel_coding(bits, 2)  # Medium protection
            else:
                # No weights or unequal protection disabled - use uniform protection
                encoded_bits = self._apply_channel_coding(bits, 2)  # Medium protection
        else:
            # Standard channel coding
            encoded_bits = self._apply_channel_coding(bits)

        # Map bits to symbols
        symbols = self._bits_to_symbols(encoded_bits)

        # Apply OFDM modulation if selected
        signal = self._apply_ofdm_modulation(symbols)

        # Store original bits for error rate estimation
        original_bits = encoded_bits.copy()

        # Transmit through the physical channel (apply channel effects)
        received_signal = self._apply_channel_effects(signal)

        # Apply OFDM demodulation if needed
        received_symbols = self._apply_ofdm_demodulation(received_signal)

        # Map symbols back to bits
        received_bits = self._symbols_to_bits(received_symbols)

        # Estimate error rate
        error_rate = self._estimate_error_rate(original_bits, received_bits)
        if debug:
            logger.debug(f"Transmission BER: {error_rate:.4f}")

        # Apply channel decoding
        decoded_bits = self._decode_channel_coding(received_bits)

        # Convert bits back to vector
        received_vector = self._bits_to_vector(decoded_bits, original_shape)

        # If importance weighting was applied, unapply it
        if self.importance_weighting and np.any(actual_weights != 1.0):
            # Avoid division by zero
            safe_weights = np.where(actual_weights > 1e-10, actual_weights, 1.0)
            received_vector = received_vector / safe_weights

        # Return the received embedding and debug info if requested
        if debug:
            return received_vector, {
                'bits': bits,
                'encoded_bits': encoded_bits,
                'symbols': symbols,
                'received_symbols': received_symbols,
                'decoded_bits': decoded_bits,
                'error_rate': error_rate,
                'estimated_snr': self.channel_stats['estimated_snr']
            }
        else:
            return received_vector

    def get_ber(self, original_bits, received_bits):
        """Calculate Bit Error Rate."""
        # Unchanged
        min_len = min(len(original_bits), len(received_bits))
        errors = np.sum(original_bits[:min_len] != received_bits[:min_len])
        return errors / min_len

    def get_channel_info(self):
        """
        Enhanced: Return information about the channel configuration and statistics.
        """
        return {
            'channel_type': self.channel_type,
            'modulation': f"{self.modulation}-{self.modulation_order}",
            'snr_db': self.snr_db,
            'estimated_snr': self.channel_stats['estimated_snr'],
            'coding_rate': self.coding_rate,
            'bits_per_symbol': self.bits_per_symbol,
            'adaptive_modulation': self.enable_adaptive_modulation,
            'unequal_error_protection': self.enable_unequal_error_protection,
            'error_rate': self.channel_stats['error_rate']
        }

    def optimize_transmission_parameters(self, embedding, text=None):
        """
        Optimize physical channel parameters based on content importance.

        Args:
            embedding: The embedding to transmit
            text: Optional text for content analysis

        Returns:
            Tuple of (importance_weights, protection_levels)
        """
        # Default uniform importance
        if isinstance(embedding, torch.Tensor):
            embedding = embedding.detach().cpu().numpy()

        # Ensure it's a numpy array
        if not isinstance(embedding, np.ndarray):
            embedding = np.array(embedding)

        importance_weights = np.ones_like(embedding)

        # Default protection levels (1 = low, 3 = high)
        protection_levels = np.ones_like(embedding, dtype=np.int8)

        # Without text, use variance-based importance
        if text is None:
            # Calculate variance along feature dimensions
            if len(embedding.shape) > 1:
                variance = np.var(embedding, axis=0)
                importance_weights = 0.3 + 0.7 * (variance / (np.max(variance) + 1e-8))
            return importance_weights, protection_levels

        # With text, use content-based importance
        try:
            # Detect critical parliamentary terms
            critical_terms = [
                "Parliament", "Commission", "Council", "Presidency",
                "Rule", "Article", "vote", "procedure", "codecision"
            ]

            # Check for presence of critical terms
            has_critical_terms = any(term in text for term in critical_terms)

            # Content type detection
            content_types = {
                "procedural": ["agenda", "vote", "procedure", "Rule", "order"],
                "legislative": ["proposal", "directive", "regulation", "amendment"],
                "debate": ["opinion", "agree", "disagree", "support", "oppose"]
            }

            # Determine content type
            content_scores = {ctype: 0 for ctype in content_types}
            for ctype, terms in content_types.items():
                content_scores[ctype] = sum(1 for term in terms if term in text)

            primary_type = max(content_scores.items(), key=lambda x: x[1])[0]

            # Apply importance profile based on content type
            if primary_type == "procedural":
                # Higher protection for early dimensions (65% to 25%)
                importance_profile = np.linspace(0.65, 0.25, len(importance_weights))
            elif primary_type == "legislative":
                # More uniform but still prioritize early dimensions (60% to 30%)
                importance_profile = np.linspace(0.6, 0.3, len(importance_weights))
            else:  # debate
                # More balanced importance (55% to 35%)
                importance_profile = np.linspace(0.55, 0.35, len(importance_weights))

            # Boost importance if critical terms present
            if has_critical_terms:
                importance_profile = 0.3 + 0.7 * importance_profile

            # Apply to importance weights
            importance_weights = importance_profile

            # Determine protection levels based on importance
            protection_levels = np.ones_like(embedding, dtype=np.int8)
            high_prot_threshold = 0.6
            med_prot_threshold = 0.4

            # Assign protection levels (3=high, 2=medium, 1=low)
            protection_levels = np.where(importance_weights >= high_prot_threshold, 3,
                                         np.where(importance_weights >= med_prot_threshold, 2, 1))

            return importance_weights, protection_levels

        except Exception as e:
            logger.warning(f"Error in transmission parameter optimization: {e}")
            return importance_weights, protection_levels

    def configure(self, **kwargs):
        """Update channel configuration parameters."""
        # Unchanged
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                if key in ['modulation', 'modulation_order']:
                    self._init_constellation()  # Reinitialize constellation if modulation changes
            else:
                logger.warning(f"Unknown parameter: {key}")

        logger.info(f"Reconfigured Physical Channel: {self.channel_type} channel, "
                    f"{self.modulation}-{self.modulation_order} modulation, SNR={self.snr_db}dB")