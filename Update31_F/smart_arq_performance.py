# smart_arq_performance.py
"""
Enhanced Smart-ARQ performance logging and tracking for semantic communication system
"""

import numpy as np
import logging
import time
from collections import defaultdict
import json
import inspect
logger = logging.getLogger(__name__)


class SmartARQPerformanceTracker:
    """Comprehensive performance tracker for Smart-ARQ system"""

    def __init__(self):
        self.reset_stats()

    def reset_stats(self):
        """Reset all statistics"""
        self.stats = {
            # Overall statistics
            'total_transmissions': 0,
            'total_retransmissions': 0,
            'total_frames_sent': 0,
            'total_frames_failed': 0,

            # Retransmission reasons
            'crc_failures': 0,
            'semantic_anchor_corruptions': 0,
            'probabilistic_retransmissions': 0,
            'no_retransmissions': 0,

            # Per-content-type statistics
            'content_type_stats': defaultdict(lambda: {
                'transmissions': 0,
                'retransmissions': 0,
                'avg_retrans_count': 0,
                'semantic_corruptions': 0
            }),

            # Performance metrics
            'avg_retransmissions_per_frame': 0.0,
            'retransmission_rate': 0.0,
            'semantic_preservation_rate': 0.0,

            # Time-based metrics
            'transmission_times': [],
            'retransmission_overhead_ms': [],

            # Detailed transmission log
            'transmission_log': []
        }

    def log_transmission_start(self, content_type, importance_score, snr_db):
        """Log the start of a transmission"""
        self.current_transmission = {
            'start_time': time.time(),
            'content_type': content_type,
            'importance_score': importance_score,
            'snr_db': snr_db,
            'retransmissions': [],
            'final_success': False
        }

    def log_retransmission(self, reason, attempt_number, ber, semantic_corruption):
        """Log a retransmission event"""
        if hasattr(self, 'current_transmission'):
            self.current_transmission['retransmissions'].append({
                'reason': reason,
                'attempt': attempt_number,
                'ber': ber,
                'semantic_corruption': semantic_corruption,
                'timestamp': time.time()
            })

            # Update reason counters
            if reason == 'crc_failure':
                self.stats['crc_failures'] += 1
            elif reason == 'semantic_anchor_corruption':
                self.stats['semantic_anchor_corruptions'] += 1
            elif reason == 'probabilistic_semantic':
                self.stats['probabilistic_retransmissions'] += 1

    def log_transmission_complete(self, success, final_ber=None):
        """Log the completion of a transmission"""
        if hasattr(self, 'current_transmission'):
            end_time = time.time()
            transmission_time = end_time - self.current_transmission['start_time']

            self.current_transmission['end_time'] = end_time
            self.current_transmission['transmission_time_ms'] = transmission_time * 1000
            self.current_transmission['final_success'] = success
            self.current_transmission['final_ber'] = final_ber
            self.current_transmission['retransmission_count'] = len(self.current_transmission['retransmissions'])

            # Update statistics
            self.stats['total_transmissions'] += 1
            self.stats['total_retransmissions'] += self.current_transmission['retransmission_count']

            if not success:
                self.stats['total_frames_failed'] += 1
            else:
                self.stats['no_retransmissions'] += 1 if self.current_transmission['retransmission_count'] == 0 else 0

            # Update content type statistics
            content_type = self.current_transmission['content_type']
            self.stats['content_type_stats'][content_type]['transmissions'] += 1
            self.stats['content_type_stats'][content_type]['retransmissions'] += self.current_transmission[
                'retransmission_count']

            # Calculate retransmission overhead
            if self.current_transmission['retransmission_count'] > 0:
                base_time = transmission_time / (1 + self.current_transmission['retransmission_count'])
                overhead = transmission_time - base_time
                self.stats['retransmission_overhead_ms'].append(overhead * 1000)

            # Add to transmission log (keep last 100 for memory efficiency)
            self.stats['transmission_log'].append(self.current_transmission)
            if len(self.stats['transmission_log']) > 100:
                self.stats['transmission_log'].pop(0)

            # Clear current transmission
            delattr(self, 'current_transmission')

    def calculate_performance_metrics(self):
        """Calculate aggregate performance metrics"""
        if self.stats['total_transmissions'] > 0:
            # Average retransmissions per frame
            self.stats['avg_retransmissions_per_frame'] = (
                    self.stats['total_retransmissions'] / self.stats['total_transmissions']
            )

            # Retransmission rate
            frames_with_retrans = sum(1 for log in self.stats['transmission_log']
                                      if log['retransmission_count'] > 0)
            self.stats['retransmission_rate'] = frames_with_retrans / self.stats['total_transmissions']

            # Semantic preservation rate
            semantic_corruptions = self.stats['semantic_anchor_corruptions']
            total_critical = sum(1 for log in self.stats['transmission_log']
                                 if log['content_type'] in ['procedural', 'legislative'])
            if total_critical > 0:
                self.stats['semantic_preservation_rate'] = 1.0 - (semantic_corruptions / total_critical)

            # Average overhead
            if self.stats['retransmission_overhead_ms']:
                self.stats['avg_retransmission_overhead_ms'] = np.mean(self.stats['retransmission_overhead_ms'])

            # Per content type averages
            for content_type, type_stats in self.stats['content_type_stats'].items():
                if type_stats['transmissions'] > 0:
                    type_stats['avg_retrans_count'] = (
                            type_stats['retransmissions'] / type_stats['transmissions']
                    )

    def get_performance_summary(self):
        """Get a comprehensive performance summary"""
        self.calculate_performance_metrics()

        summary = {
            'overview': {
                'total_transmissions': self.stats['total_transmissions'],
                'total_retransmissions': self.stats['total_retransmissions'],
                'avg_retransmissions_per_frame': round(self.stats['avg_retransmissions_per_frame'], 3),
                'retransmission_rate': round(self.stats['retransmission_rate'], 3),
                'frames_failed': self.stats['total_frames_failed']
            },
            'retransmission_reasons': {
                'crc_failures': self.stats['crc_failures'],
                'semantic_anchor_corruptions': self.stats['semantic_anchor_corruptions'],
                'probabilistic_retransmissions': self.stats['probabilistic_retransmissions'],
                'no_retransmissions_needed': self.stats['no_retransmissions']
            },
            'semantic_performance': {
                'semantic_preservation_rate': round(self.stats['semantic_preservation_rate'], 3),
                'critical_content_protected': self.stats['semantic_anchor_corruptions'] > 0
            },
            'efficiency': {
                'avg_overhead_ms': round(self.stats.get('avg_retransmission_overhead_ms', 0), 2),
                'total_overhead_ms': round(sum(self.stats['retransmission_overhead_ms']), 2)
            },
            'per_content_type': dict(self.stats['content_type_stats'])
        }

        return summary

    def log_to_file(self, filepath):
        """Save detailed log to file"""
        log_data = {
            'summary': self.get_performance_summary(),
            'detailed_stats': self.stats,
            'transmission_history': self.stats['transmission_log'][-20:]  # Last 20 transmissions
        }

        with open(filepath, 'w') as f:
            json.dump(log_data, f, indent=2)

    def print_performance_report(self):
        """Print a formatted performance report"""
        summary = self.get_performance_summary()

        print("\n" + "=" * 60)
        print("SMART-ARQ PERFORMANCE REPORT")
        print("=" * 60)

        print("\nOVERVIEW:")
        print(f"  Total Transmissions: {summary['overview']['total_transmissions']}")
        print(f"  Total Retransmissions: {summary['overview']['total_retransmissions']}")
        print(f"  Average Retransmissions/Frame: {summary['overview']['avg_retransmissions_per_frame']}")
        print(f"  Retransmission Rate: {summary['overview']['retransmission_rate'] * 100:.1f}%")
        print(f"  Failed Frames: {summary['overview']['frames_failed']}")

        print("\nRETRANSMISSION TRIGGERS:")
        print(f"  CRC Failures: {summary['retransmission_reasons']['crc_failures']}")
        print(f"  Semantic Anchor Corruptions: {summary['retransmission_reasons']['semantic_anchor_corruptions']}")
        print(f"  Probabilistic Retransmissions: {summary['retransmission_reasons']['probabilistic_retransmissions']}")
        print(f"  Clean Transmissions: {summary['retransmission_reasons']['no_retransmissions_needed']}")

        print("\nSEMANTIC PRESERVATION:")
        print(f"  Preservation Rate: {summary['semantic_performance']['semantic_preservation_rate'] * 100:.1f}%")
        print(f"  Critical Content Protection Active: {summary['semantic_performance']['critical_content_protected']}")

        print("\nEFFICIENCY METRICS:")
        print(f"  Average Retransmission Overhead: {summary['efficiency']['avg_overhead_ms']:.2f} ms")
        print(f"  Total Overhead: {summary['efficiency']['total_overhead_ms']:.2f} ms")

        if summary['per_content_type']:
            print("\nPER CONTENT TYPE PERFORMANCE:")
            for content_type, stats in summary['per_content_type'].items():
                if stats['transmissions'] > 0:
                    print(f"\n  {content_type.upper()}:")
                    print(f"    Transmissions: {stats['transmissions']}")
                    print(f"    Avg Retransmissions: {stats['avg_retrans_count']:.2f}")

        print("\n" + "=" * 60)


# Global tracker instance
smart_arq_tracker = SmartARQPerformanceTracker()


def integrate_smart_arq_logging(physical_channel):
    """
    Integrate Smart-ARQ logging into the physical channel.
    This patches the channel to add comprehensive logging.
    """

    # Store original transmit method
    original_transmit = physical_channel.transmit

    def logged_transmit(embedding, importance_weights=None, debug=False, max_retransmissions=2):
        """Enhanced transmit with comprehensive Smart-ARQ logging"""

        # Get content classification
        content_type = "unknown"
        importance_score = 0.0

        if hasattr(physical_channel, 'content_classifier') and physical_channel.content_classifier:
            try:
                content_type, probs = physical_channel.content_classifier.classify(embedding)
                importance_score = max(probs) if isinstance(probs, list) else 0.5
            except:
                pass

        # Get current SNR
        current_snr = getattr(physical_channel, 'snr_db', 20.0)
        if hasattr(physical_channel, 'channel_stats'):
            current_snr = physical_channel.channel_stats.get('estimated_snr', current_snr)

        # Log transmission start
        smart_arq_tracker.log_transmission_start(content_type, importance_score, current_snr)

        # Store retransmission count for this transmission
        physical_channel._current_retrans_count = 0

        # Call original transmit with logging
        try:
            # If the physical channel supports Smart-ARQ
            if hasattr(physical_channel, '_smart_arq_decision'):
                # Override the _smart_arq_decision to add logging
                original_arq_decision = physical_channel._smart_arq_decision

                def logged_arq_decision(frame_ok, semantic_corruption, content_type="unknown"):
                    should_retrans, reason = original_arq_decision(frame_ok, semantic_corruption, content_type)

                    if should_retrans:
                        # Calculate approximate BER
                        ber = 0.1 if not frame_ok else 0.05 if semantic_corruption else 0.01
                        smart_arq_tracker.log_retransmission(
                            reason,
                            physical_channel._current_retrans_count + 1,
                            ber,
                            semantic_corruption
                        )
                        physical_channel._current_retrans_count += 1

                    return should_retrans, reason

                # Temporarily replace the method
                physical_channel._smart_arq_decision = logged_arq_decision

                # Check if transmit accepts max_retransmissions
                import inspect
                sig = inspect.signature(original_transmit)
                if 'max_retransmissions' in sig.parameters:
                    result = original_transmit(embedding, importance_weights, debug, max_retransmissions)
                else:
                    result = original_transmit(embedding, importance_weights, debug)
            else:
                # Standard transmit without Smart-ARQ
                result = original_transmit(embedding, importance_weights, debug)

            # Log completion
            smart_arq_tracker.log_transmission_complete(True, final_ber=0.0)

            # Store retransmission count for pipeline tracking
            physical_channel._last_retransmission_count = physical_channel._current_retrans_count

            return result

        except Exception as e:
            # Log failure
            smart_arq_tracker.log_transmission_complete(False)
            raise e
        finally:
            # Restore original method if it was replaced
            if hasattr(physical_channel, '_smart_arq_decision') and 'original_arq_decision' in locals():
                physical_channel._smart_arq_decision = original_arq_decision

    # Replace transmit method with logged version
    physical_channel.transmit = logged_transmit

    # Add method to get Smart-ARQ statistics
    def get_smart_arq_performance():
        return smart_arq_tracker.get_performance_summary()

    physical_channel.get_smart_arq_performance = get_smart_arq_performance

    logger.info("Smart-ARQ performance logging integrated")

    return physical_channel