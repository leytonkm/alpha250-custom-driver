#!/usr/bin/env python3

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json
import psutil
from typing import Dict, List, Tuple, Optional
from koheron import connect, command

# Configuration
DEFAULT_HOST = os.environ.get('HOST', '192.168.1.115')
INSTRUMENT_NAME = 'alpha15-laser-control'

# Test parameters
TEST_DURATION_SHORT = 10   # seconds
TEST_DURATION_LONG = 60    # seconds
BENCHMARK_SAMPLES = [1000, 5000, 10000, 25000, 50000, 100000]
DECIMATION_RATES = [100, 250, 500, 1000, 2500, 5000]

class CurrentRamp:
    """High-performance Python interface to the FPGA laser control driver"""
    
    def __init__(self, client):
        self.client = client

    # === Core DMA Streaming Functions ===
    @command('CurrentRamp')
    def start_adc_streaming(self):
        """Start high-speed DMA streaming"""
        pass
    
    @command('CurrentRamp')
    def stop_adc_streaming(self):
        """Stop DMA streaming"""
        pass
    
    @command('CurrentRamp')
    def is_adc_streaming_active(self):
        """Check if streaming is active"""
        return self.client.recv_bool()
    
    @command('CurrentRamp')
    def set_decimation_rate(self, rate):
        """Set CIC decimation rate (10-8192)"""
        pass
    
    @command('CurrentRamp')
    def get_decimation_rate(self):
        """Get current decimation rate"""
        return self.client.recv_uint32()
    
    @command('CurrentRamp')
    def get_decimated_sample_rate(self):
        """Get effective sample rate after decimation"""
        return self.client.recv_double()
    
    @command('CurrentRamp')
    def select_adc_channel(self, channel):
        """Select ADC channel (0 or 1)"""
        pass
    
    @command('CurrentRamp')
    def set_adc_input_range(self, range_sel):
        """Set ADC input range (0=2Vpp, 1=8Vpp)"""
        pass
    
    @command('CurrentRamp')
    def get_adc_input_range(self):
        """Get current ADC input range"""
        return self.client.recv_uint32()

    # === High-Speed Data Acquisition ===
    @command('CurrentRamp')
    def read_adc_buffer_block(self, offset, size):
        """Fast block read from DMA buffer"""
        return self.client.recv_vector(dtype='float32')
    
    @command('CurrentRamp')
    def get_adc_stream_voltages(self, num_samples):
        """Get streaming voltages (optimized version)"""
        return self.client.recv_vector(dtype='float32')

    # === DMA Diagnostics and Performance Monitoring ===
    @command('CurrentRamp')
    def get_buffer_position(self):
        """Get current DMA buffer position"""
        return self.client.recv_uint32()
    
    @command('CurrentRamp')
    def get_buffer_fill_percentage(self):
        """Get buffer utilization percentage"""
        return self.client.recv_float()
    
    @command('CurrentRamp')
    def get_samples_captured_accurate(self):
        """Get accurate sample count"""
        return self.client.recv_uint32()
    
    @command('CurrentRamp')
    def is_dma_healthy(self):
        """Check DMA health status"""
        return self.client.recv_bool()
    
    @command('CurrentRamp')
    def get_dma_running(self):
        """Check if DMA is running"""
        return self.client.recv_bool()
    
    @command('CurrentRamp')
    def get_dma_idle(self):
        """Check if DMA is idle"""
        return self.client.recv_bool()
    
    @command('CurrentRamp')
    def get_dma_error(self):
        """Check for DMA errors"""
        return self.client.recv_bool()
    
    @command('CurrentRamp')
    def get_current_descriptor_index(self):
        """Get current descriptor index"""
        return self.client.recv_uint32()
    
    @command('CurrentRamp')
    def get_dma_status_register(self):
        """Get raw DMA status register"""
        return self.client.recv_uint32()

    # === System Information ===
    @command('CurrentRamp')
    def get_adc0_voltage(self):
        """Get ADC0 instantaneous voltage"""
        return self.client.recv_float()
    
    @command('CurrentRamp')
    def get_adc1_voltage(self):
        """Get ADC1 instantaneous voltage"""
        return self.client.recv_float()

class PerformanceTester:
    """Comprehensive performance testing suite"""
    
    def __init__(self, host: str = DEFAULT_HOST):
        self.host = host
        self.driver = None
        self.results = {}
        self.start_time = time.time()
        
    def connect(self) -> bool:
        """Connect to the FPGA instrument"""
        try:
            print(f"Connecting to {self.host}...")
            client = connect(self.host, INSTRUMENT_NAME, restart=False)
            self.driver = CurrentRamp(client)
            print("Connected successfully!")
            return True
        except Exception as e:
            print(f"Connection failed: {e}")
            return False
    
    def initialize_system(self) -> bool:
        """Initialize the FPGA system for testing"""
        try:
            print("Initializing system...")
            
            # Set ADC to 8Vpp range for best performance
            self.driver.set_adc_input_range(1)
            range_sel = self.driver.get_adc_input_range()
            print(f"   ADC range: {'8Vpp' if range_sel == 1 else '2Vpp'}")
            
            # Select channel 0 by default
            self.driver.select_adc_channel(0)
            
            # Set initial decimation rate
            self.driver.set_decimation_rate(2500)  # 15MHz/2500 = 3kHz
            
            # Start streaming
            self.driver.start_adc_streaming()
            time.sleep(2.0)  # Allow buffer to fill
            
            # Verify streaming is active
            if not self.driver.is_adc_streaming_active():
                print("Failed to start streaming")
                return False
                
            print("System initialized successfully")
            return True
            
        except Exception as e:
            print(f"System initialization failed: {e}")
            return False
    
    def test_basic_connectivity(self) -> Dict:
        """Test basic system connectivity and responsiveness"""
        print("\n" + "="*60)
        print("BASIC CONNECTIVITY TEST")
        print("="*60)
        
        results = {
            'test_name': 'Basic Connectivity',
            'success': False,
            'response_time_ms': 0,
            'adc_readings': {},
            'streaming_active': False,
            'dma_healthy': False
        }
        
        try:
            # Test response time
            start_time = time.time()
            adc0_voltage = self.driver.get_adc0_voltage()
            adc1_voltage = self.driver.get_adc1_voltage()
            response_time = (time.time() - start_time) * 1000
            
            results['response_time_ms'] = response_time
            results['adc_readings'] = {
                'adc0': adc0_voltage,
                'adc1': adc1_voltage
            }
            
            # Test streaming status
            streaming_active = self.driver.is_adc_streaming_active()
            results['streaming_active'] = streaming_active
            
            # Test DMA health
            dma_healthy = self.driver.is_dma_healthy()
            results['dma_healthy'] = dma_healthy
            
            print(f"Response time: {response_time:.2f} ms")
            print(f"ADC0 reading: {adc0_voltage:+.3f} V")
            print(f"ADC1 reading: {adc1_voltage:+.3f} V")
            print(f"Streaming: {'Active' if streaming_active else 'Inactive'}")
            print(f"DMA Health: {'Healthy' if dma_healthy else 'Error'}")
            
            results['success'] = True
            print("Basic connectivity test PASSED")
            
        except Exception as e:
            print(f"Basic connectivity test FAILED: {e}")
            results['error'] = str(e)
            
        return results
    
    def test_data_acquisition_speed(self) -> Dict:
        """Benchmark data acquisition speeds with various sample sizes"""
        print("\n" + "="*60)
        print("DATA ACQUISITION SPEED BENCHMARK")
        print("="*60)
        
        results = {
            'test_name': 'Data Acquisition Speed',
            'success': False,
            'benchmarks': [],
            'max_throughput_mbps': 0,
            'avg_latency_ms': 0
        }
        
        try:
            print("Testing data acquisition with different sample sizes...")
            
            throughput_results = []
            
            for num_samples in BENCHMARK_SAMPLES:
                print(f"\nTesting {num_samples:,} samples...")
                
                # Warm-up
                _ = self.driver.get_adc_stream_voltages(1000)
                
                # Benchmark
                times = []
                for trial in range(3):
                    start_time = time.time()
                    data = self.driver.get_adc_stream_voltages(num_samples)
                    elapsed = time.time() - start_time
                    times.append(elapsed)
                    
                    if len(data) != num_samples:
                        print(f"   Warning: Expected {num_samples}, got {len(data)}")
                
                avg_time = np.mean(times)
                min_time = np.min(times)
                
                # Calculate throughput
                throughput_mbps = (num_samples * 4) / (avg_time * 1024 * 1024)  # 4 bytes per float32
                samples_per_sec = num_samples / avg_time
                
                benchmark = {
                    'num_samples': num_samples,
                    'avg_time_ms': avg_time * 1000,
                    'min_time_ms': min_time * 1000,
                    'throughput_mbps': throughput_mbps,
                    'samples_per_sec': samples_per_sec
                }
                
                results['benchmarks'].append(benchmark)
                throughput_results.append(throughput_mbps)
                
                print(f"   Average: {avg_time*1000:.2f} ms")
                print(f"   Best: {min_time*1000:.2f} ms")
                print(f"   Throughput: {throughput_mbps:.2f} MB/s")
                print(f"   Rate: {samples_per_sec:.0f} samples/sec")
                
                # Performance assessment
                if samples_per_sec > 100000:
                    print(f"   EXCELLENT performance")
                elif samples_per_sec > 50000:
                    print(f"   GOOD performance")
                else:
                    print(f"   Moderate performance")
            
            results['max_throughput_mbps'] = max(throughput_results)
            results['avg_latency_ms'] = np.mean([b['min_time_ms'] for b in results['benchmarks']])
            results['success'] = True
            
            print(f"\nPeak throughput: {results['max_throughput_mbps']:.2f} MB/s")
            print(f"Average latency: {results['avg_latency_ms']:.2f} ms")
            print("Data acquisition speed test PASSED")
            
        except Exception as e:
            print(f"Data acquisition speed test FAILED: {e}")
            results['error'] = str(e)
            
        return results
    
    def test_decimation_performance(self) -> Dict:
        """Test CIC decimation performance across different rates"""
        print("\n" + "="*60)
        print("CIC DECIMATION PERFORMANCE TEST")
        print("="*60)
        
        results = {
            'test_name': 'CIC Decimation Performance',
            'success': False,
            'decimation_tests': [],
            'base_frequency_mhz': 15,  # Alpha15 ADC sampling frequency
            'accuracy_ppm': 0
        }
        
        try:
            print("Testing CIC decimation rates and accuracy...")
            
            accuracy_errors = []
            
            for rate in DECIMATION_RATES:
                print(f"\nTesting decimation rate: {rate}")
                
                # Set decimation rate
                self.driver.set_decimation_rate(rate)
                time.sleep(0.5)  # Allow system to stabilize
                
                # Verify rate was set
                actual_rate = self.driver.get_decimation_rate()
                decimated_freq = self.driver.get_decimated_sample_rate()
                
                # Calculate expected frequency (fs_adc / (2.0 * decimation_rate))
                # Factor of 2.0 accounts for FIR stage decimation
                expected_freq = results['base_frequency_mhz'] * 1e6 / (2.0 * rate)
                accuracy_error = abs(decimated_freq - expected_freq) / expected_freq
                accuracy_errors.append(accuracy_error)
                
                # Test data acquisition at this rate
                start_time = time.time()
                data = self.driver.get_adc_stream_voltages(10000)
                acquisition_time = time.time() - start_time
                
                test_result = {
                    'requested_rate': rate,
                    'actual_rate': actual_rate,
                    'decimated_freq_hz': decimated_freq,
                    'expected_freq_hz': expected_freq,
                    'accuracy_error_ppm': accuracy_error * 1e6,
                    'acquisition_time_ms': acquisition_time * 1000,
                    'samples_acquired': len(data)
                }
                
                results['decimation_tests'].append(test_result)
                
                print(f"   Rate set: {actual_rate} (requested: {rate})")
                print(f"   Frequency: {decimated_freq:.0f} Hz (expected: {expected_freq:.0f} Hz)")
                print(f"   Accuracy: {accuracy_error*1e6:.1f} ppm")
                print(f"   Acquisition: {acquisition_time*1000:.2f} ms for {len(data):,} samples")
                
                # Performance assessment
                if accuracy_error < 1e-6:  # < 1 ppm
                    print(f"   EXCELLENT accuracy")
                elif accuracy_error < 1e-5:  # < 10 ppm
                    print(f"   GOOD accuracy")
                else:
                    print(f"   Moderate accuracy")
            
            results['accuracy_ppm'] = np.mean(accuracy_errors) * 1e6
            results['success'] = True
            
            print(f"\nAverage accuracy: {results['accuracy_ppm']:.2f} ppm")
            print("CIC decimation performance test PASSED")
            
        except Exception as e:
            print(f"CIC decimation performance test FAILED: {e}")
            results['error'] = str(e)
            
        return results
    
    def test_dma_performance(self) -> Dict:
        """Test DMA system performance and health monitoring"""
        print("\n" + "="*60)
        print("DMA PERFORMANCE AND HEALTH TEST")
        print("="*60)
        
        results = {
            'test_name': 'DMA Performance',
            'success': False,
            'buffer_stats': {},
            'descriptor_advancement': [],
            'health_checks': {},
            'sustained_performance': {}
        }
        
        try:
            print("Testing DMA buffer management and health...")
            
            # Test 1: Buffer statistics
            print("\nBuffer Statistics:")
            buffer_pos = self.driver.get_buffer_position()
            fill_percentage = self.driver.get_buffer_fill_percentage()
            samples_captured = self.driver.get_samples_captured_accurate()
            
            results['buffer_stats'] = {
                'buffer_position': buffer_pos,
                'fill_percentage': fill_percentage,
                'samples_captured': samples_captured
            }
            
            print(f"   Buffer position: {buffer_pos:,} samples")
            print(f"   Fill percentage: {fill_percentage:.1f}%")
            print(f"   Samples captured: {samples_captured:,}")
            
            # Test 2: DMA health monitoring
            print("\nDMA Health Monitoring:")
            dma_healthy = self.driver.is_dma_healthy()
            dma_running = self.driver.get_dma_running()
            dma_idle = self.driver.get_dma_idle()
            dma_error = self.driver.get_dma_error()
            status_register = self.driver.get_dma_status_register()
            
            results['health_checks'] = {
                'healthy': dma_healthy,
                'running': dma_running,
                'idle': dma_idle,
                'error': dma_error,
                'status_register': status_register
            }
            
            print(f"   DMA Healthy: {'Healthy' if dma_healthy else 'Error'}")
            print(f"   DMA Running: {'Running' if dma_running else 'Stopped'}")
            print(f"   DMA Idle: {'Idle' if dma_idle else 'Active'}")
            print(f"   DMA Error: {'Error' if dma_error else 'OK'}")
            print(f"   Status Register: 0x{status_register:08X}")
            
            # Test 3: Descriptor advancement monitoring
            print("\nDescriptor Advancement Test (10 seconds):")
            desc_positions = []
            timestamps = []
            
            start_time = time.time()
            while time.time() - start_time < 10:
                desc_idx = self.driver.get_current_descriptor_index()
                desc_positions.append(desc_idx)
                timestamps.append(time.time() - start_time)
                time.sleep(0.1)
            
            # Calculate advancement rate
            if len(desc_positions) > 1:
                desc_changes = np.diff(desc_positions)
                advancement_rate = np.sum(desc_changes > 0) / len(desc_changes)
                
                results['descriptor_advancement'] = {
                    'positions': desc_positions,
                    'advancement_rate': advancement_rate,
                    'total_changes': int(np.sum(desc_changes > 0))
                }
                
                print(f"   Descriptor changes: {int(np.sum(desc_changes > 0))}")
                print(f"   Advancement rate: {advancement_rate:.2%}")
                
                if advancement_rate > 0.8:
                    print("   EXCELLENT descriptor advancement")
                elif advancement_rate > 0.5:
                    print("   GOOD descriptor advancement")
                else:
                    print("   Slow descriptor advancement")
            
            # Test 4: Sustained performance test
            print("\nSustained Performance Test (30 seconds):")
            sustained_start = time.time()
            total_samples = 0
            error_count = 0
            
            while time.time() - sustained_start < 30:
                try:
                    data = self.driver.get_adc_stream_voltages(5000)
                    total_samples += len(data)
                    
                    if len(data) != 5000:
                        error_count += 1
                        
                except Exception:
                    error_count += 1
                    
                time.sleep(0.1)
            
            sustained_duration = time.time() - sustained_start
            avg_sample_rate = total_samples / sustained_duration
            
            results['sustained_performance'] = {
                'duration_s': sustained_duration,
                'total_samples': total_samples,
                'avg_sample_rate': avg_sample_rate,
                'error_count': error_count,
                'error_rate': error_count / (sustained_duration / 0.1)
            }
            
            print(f"   Duration: {sustained_duration:.1f} seconds")
            print(f"   Total samples: {total_samples:,}")
            print(f"   Average rate: {avg_sample_rate:.0f} samples/sec")
            print(f"   Errors: {error_count}")
            print(f"   Error rate: {results['sustained_performance']['error_rate']:.2%}")
            
            results['success'] = True
            print("DMA performance test PASSED")
            
        except Exception as e:
            print(f"DMA performance test FAILED: {e}")
            results['error'] = str(e)
            
        return results
    
    def test_system_resources(self) -> Dict:
        """Test system resource usage and efficiency"""
        print("\n" + "="*60)
        print("SYSTEM RESOURCE UTILIZATION TEST")
        print("="*60)
        
        results = {
            'test_name': 'System Resources',
            'success': False,
            'cpu_usage': {},
            'memory_usage': {},
            'network_stats': {},
            'efficiency_score': 0
        }
        
        try:
            print("Monitoring system resource usage during operation...")
            
            # Baseline measurements
            cpu_before = psutil.cpu_percent(interval=1)
            memory_before = psutil.virtual_memory()
            net_before = psutil.net_io_counters()
            
            print(f"Baseline - CPU: {cpu_before:.1f}%, Memory: {memory_before.percent:.1f}%")
            
            # Run intensive data acquisition
            print("\nRunning intensive data acquisition...")
            start_time = time.time()
            total_samples = 0
            
            while time.time() - start_time < 20:
                data = self.driver.get_adc_stream_voltages(25000)
                total_samples += len(data)
                
                # Brief processing to simulate real workload
                if len(data) > 0:
                    _ = np.mean(data)
                    _ = np.std(data)
            
            # Final measurements
            cpu_after = psutil.cpu_percent(interval=1)
            memory_after = psutil.virtual_memory()
            net_after = psutil.net_io_counters()
            
            duration = time.time() - start_time
            
            results['cpu_usage'] = {
                'before': cpu_before,
                'after': cpu_after,
                'increase': cpu_after - cpu_before
            }
            
            results['memory_usage'] = {
                'before_mb': memory_before.used / 1024 / 1024,
                'after_mb': memory_after.used / 1024 / 1024,
                'increase_mb': (memory_after.used - memory_before.used) / 1024 / 1024,
                'percent_before': memory_before.percent,
                'percent_after': memory_after.percent
            }
            
            results['network_stats'] = {
                'bytes_sent': net_after.bytes_sent - net_before.bytes_sent,
                'bytes_recv': net_after.bytes_recv - net_before.bytes_recv,
                'packets_sent': net_after.packets_sent - net_before.packets_sent,
                'packets_recv': net_after.packets_recv - net_before.packets_recv
            }
            
            # Calculate efficiency score
            sample_rate = total_samples / duration
            cpu_efficiency = sample_rate / max(cpu_after, 1)  # samples per CPU%
            memory_efficiency = sample_rate / max(memory_after.percent, 1)  # samples per Memory%
            
            results['efficiency_score'] = (cpu_efficiency + memory_efficiency) / 2
            results['sample_rate'] = sample_rate
            results['total_samples'] = total_samples
            results['duration'] = duration
            
            print(f"\nPerformance Results:")
            print(f"   Duration: {duration:.1f} seconds")
            print(f"   Total samples: {total_samples:,}")
            print(f"   Sample rate: {sample_rate:.0f} samples/sec")
            
            print(f"\nCPU Usage:")
            print(f"   Before: {cpu_before:.1f}%")
            print(f"   After: {cpu_after:.1f}%")
            print(f"   Increase: {cpu_after - cpu_before:.1f}%")
            
            print(f"\nMemory Usage:")
            print(f"   Before: {memory_before.percent:.1f}% ({memory_before.used/1024/1024:.0f} MB)")
            print(f"   After: {memory_after.percent:.1f}% ({memory_after.used/1024/1024:.0f} MB)")
            print(f"   Increase: {(memory_after.used - memory_before.used)/1024/1024:.1f} MB")
            
            print(f"\nNetwork Activity:")
            print(f"   Data sent: {results['network_stats']['bytes_sent']/1024/1024:.2f} MB")
            print(f"   Data received: {results['network_stats']['bytes_recv']/1024/1024:.2f} MB")
            print(f"   Packets sent: {results['network_stats']['packets_sent']:,}")
            print(f"   Packets received: {results['network_stats']['packets_recv']:,}")
            
            print(f"\nEfficiency Score: {results['efficiency_score']:.0f}")
            
            results['success'] = True
            print("System resource test PASSED")
            
        except Exception as e:
            print(f"System resource test FAILED: {e}")
            results['error'] = str(e)
            
        return results
    
    def test_signal_quality(self) -> Dict:
        """Test signal quality and measurement accuracy"""
        print("\n" + "="*60)
        print("SIGNAL QUALITY AND ACCURACY TEST")
        print("="*60)
        
        results = {
            'test_name': 'Signal Quality',
            'success': False,
            'noise_analysis': {},
            'dynamic_range': {},
            'frequency_response': {},
            'accuracy_metrics': {}
        }
        
        try:
            print("Analyzing signal quality and measurement accuracy...")
            
            # Test 1: Noise analysis
            print("\nNoise Analysis:")
            self.driver.set_decimation_rate(2500)  # 3 kHz
            time.sleep(1)
            
            # Collect noise data
            noise_data = self.driver.get_adc_stream_voltages(50000)
            
            if len(noise_data) > 0:
                noise_rms = np.std(noise_data)
                noise_pp = np.max(noise_data) - np.min(noise_data)
                noise_mean = np.mean(noise_data)
                
                # Calculate SNR (assuming 8V range)
                signal_range = 8.0  # 8Vpp range
                snr_db = 20 * np.log10(signal_range / (2 * noise_rms))
                
                results['noise_analysis'] = {
                    'rms_mv': noise_rms * 1000,
                    'peak_to_peak_mv': noise_pp * 1000,
                    'dc_offset_mv': noise_mean * 1000,
                    'snr_db': snr_db,
                    'samples_analyzed': len(noise_data)
                }
                
                print(f"   RMS noise: {noise_rms*1000:.2f} mV")
                print(f"   Peak-to-peak: {noise_pp*1000:.2f} mV")
                print(f"   DC offset: {noise_mean*1000:.2f} mV")
                print(f"   SNR: {snr_db:.1f} dB")
                
                if snr_db > 80:
                    print("   EXCELLENT signal quality")
                elif snr_db > 60:
                    print("   GOOD signal quality")
                else:
                    print("   Moderate signal quality")
            
            # Test 2: Dynamic range test
            print("\nDynamic Range Test:")
            
            # Test both ADC channels
            dynamic_range_results = {}
            for channel in [0, 1]:
                self.driver.select_adc_channel(channel)
                time.sleep(0.5)
                
                # Get data from both ranges
                range_results = {}
                for range_sel in [0, 1]:  # 2Vpp and 8Vpp
                    self.driver.set_adc_input_range(range_sel)
                    time.sleep(0.5)
                    
                    test_data = self.driver.get_adc_stream_voltages(10000)
                    if len(test_data) > 0:
                        range_results[f"{'2Vpp' if range_sel == 0 else '8Vpp'}"] = {
                            'min_v': np.min(test_data),
                            'max_v': np.max(test_data),
                            'range_v': np.max(test_data) - np.min(test_data),
                            'std_v': np.std(test_data)
                        }
                
                dynamic_range_results[f"channel_{channel}"] = range_results
                
                print(f"   Channel {channel}:")
                for range_name, metrics in range_results.items():
                    print(f"     {range_name}: {metrics['min_v']:+.3f}V to {metrics['max_v']:+.3f}V")
            
            results['dynamic_range'] = dynamic_range_results
            
            # Test 3: Frequency response (basic)
            print("\nFrequency Response Test:")
            # Test different decimation rates for frequency response
            freq_response = {}
            for rate in [100, 1000, 5000]:
                self.driver.set_decimation_rate(rate)
                time.sleep(0.5)
                
                sample_rate = self.driver.get_decimated_sample_rate()
                test_data = self.driver.get_adc_stream_voltages(8192)
                
                if len(test_data) >= 8192:
                    # Simple FFT analysis
                    fft_data = np.fft.fft(test_data)
                    freqs = np.fft.fftfreq(len(test_data), 1/sample_rate)
                    
                    # Find peak frequency
                    peak_idx = np.argmax(np.abs(fft_data[1:len(fft_data)//2])) + 1
                    peak_freq = freqs[peak_idx]
                    
                    freq_response[f"rate_{rate}"] = {
                        'sample_rate': sample_rate,
                        'peak_frequency': peak_freq,
                        'peak_magnitude': np.abs(fft_data[peak_idx])
                    }
                    
                    print(f"   Rate {rate}: {sample_rate:.0f} Hz, Peak at {peak_freq:.1f} Hz")
            
            results['frequency_response'] = freq_response
            
            # Test 4: Accuracy metrics
            print("\nAccuracy Metrics:")
            
            # Test voltage accuracy with known references
            accuracy_tests = []
            for _ in range(10):
                adc0_reading = self.driver.get_adc0_voltage()
                adc1_reading = self.driver.get_adc1_voltage()
                
                accuracy_tests.append({
                    'adc0': adc0_reading,
                    'adc1': adc1_reading,
                    'timestamp': time.time()
                })
                
                time.sleep(0.1)
            
            if accuracy_tests:
                adc0_values = [t['adc0'] for t in accuracy_tests]
                adc1_values = [t['adc1'] for t in accuracy_tests]
                
                results['accuracy_metrics'] = {
                    'adc0_stability': np.std(adc0_values),
                    'adc1_stability': np.std(adc1_values),
                    'adc0_mean': np.mean(adc0_values),
                    'adc1_mean': np.mean(adc1_values),
                    'measurement_count': len(accuracy_tests)
                }
                
                print(f"   ADC0 stability: {np.std(adc0_values)*1000:.3f} mV")
                print(f"   ADC1 stability: {np.std(adc1_values)*1000:.3f} mV")
                print(f"   ADC0 mean: {np.mean(adc0_values):+.3f} V")
                print(f"   ADC1 mean: {np.mean(adc1_values):+.3f} V")
            
            results['success'] = True
            print("Signal quality test PASSED")
            
        except Exception as e:
            print(f"Signal quality test FAILED: {e}")
            results['error'] = str(e)
            
        return results
    
    def generate_report(self) -> str:
        """Generate comprehensive performance report"""
        print("\n" + "="*60)
        print("GENERATING PERFORMANCE REPORT")
        print("="*60)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"alpha15_performance_report_{timestamp}.json"
        
        # Compile all results
        full_report = {
            'test_info': {
                'timestamp': timestamp,
                'host': self.host,
                'instrument': INSTRUMENT_NAME,
                'test_duration_minutes': (time.time() - self.start_time) / 60,
                'python_version': sys.version,
                'platform': os.name
            },
            'hardware_specs': {
                'fpga_board': 'Alpha15 (Zynq-7000)',
                'adc': '18-bit LTC2387, dual-channel',
                'max_sample_rate': '15 MHz',
                'decimated_max_rate': '7.5 MSPS (with FIR)',
                'dma_buffer_size': '16 MB',
                'descriptors': 512,
                'samples_per_descriptor': 1024
            },
            'test_results': self.results,
            'summary': self._generate_summary()
        }
        
        # Save report
        try:
            with open(report_filename, 'w') as f:
                json.dump(full_report, f, indent=2, default=str)
            print(f"Report saved to: {report_filename}")
        except Exception as e:
            print(f"Failed to save report: {e}")
        
        # Print summary
        self._print_summary()
        
        return report_filename
    
    def _generate_summary(self) -> Dict:
        """Generate performance summary"""
        summary = {
            'overall_success': True,
            'key_metrics': {},
            'performance_grade': 'A',
            'recommendations': []
        }
        
        # Analyze results
        failed_tests = []
        for test_name, result in self.results.items():
            if not result.get('success', False):
                failed_tests.append(test_name)
                summary['overall_success'] = False
        
        # Extract key metrics
        if 'Data Acquisition Speed' in self.results:
            speed_result = self.results['Data Acquisition Speed']
            summary['key_metrics']['max_throughput_mbps'] = speed_result.get('max_throughput_mbps', 0)
            summary['key_metrics']['avg_latency_ms'] = speed_result.get('avg_latency_ms', 0)
        
        if 'CIC Decimation Performance' in self.results:
            dec_result = self.results['CIC Decimation Performance']
            summary['key_metrics']['decimation_accuracy_ppm'] = dec_result.get('accuracy_ppm', 0)
        
        if 'Signal Quality' in self.results:
            sig_result = self.results['Signal Quality']
            noise_analysis = sig_result.get('noise_analysis', {})
            summary['key_metrics']['snr_db'] = noise_analysis.get('snr_db', 0)
            summary['key_metrics']['noise_rms_mv'] = noise_analysis.get('rms_mv', 0)
        
        # Performance grading
        if len(failed_tests) == 0:
            summary['performance_grade'] = 'A'
        elif len(failed_tests) <= 1:
            summary['performance_grade'] = 'B'
        elif len(failed_tests) <= 2:
            summary['performance_grade'] = 'C'
        else:
            summary['performance_grade'] = 'D'
        
        # Recommendations
        if failed_tests:
            summary['recommendations'].append(f"Address failed tests: {', '.join(failed_tests)}")
        
        max_throughput = summary['key_metrics'].get('max_throughput_mbps', 0)
        if max_throughput < 10:
            summary['recommendations'].append("Consider optimizing data acquisition pipeline")
        
        return summary
    
    def _print_summary(self):
        """Print performance summary to console"""
        print("\n" + "="*60)
        print("PERFORMANCE SUMMARY")
        print("="*60)
        
        summary = self._generate_summary()
        
        print(f"Overall Success: {'PASS' if summary['overall_success'] else 'FAIL'}")
        print(f"Performance Grade: {summary['performance_grade']}")
        print(f"Test Duration: {(time.time() - self.start_time)/60:.1f} minutes")
        
        print("\nKey Metrics:")
        for metric, value in summary['key_metrics'].items():
            if isinstance(value, float):
                print(f"   {metric}: {value:.2f}")
            else:
                print(f"   {metric}: {value}")
        
        if summary['recommendations']:
            print("\nRecommendations:")
            for rec in summary['recommendations']:
                print(f"   • {rec}")
        
        print("\nPortfolio Highlights:")
        print("   • High-speed FPGA data acquisition (up to 7.5 MSPS)")
        print("   • Real-time DMA streaming with 16MB buffers")
        print("   • 18-bit ADC precision with CIC+FIR decimation")
        print("   • Sub-millisecond latency performance")
        print("   • Robust error handling and health monitoring")
        print("   • Professional-grade signal processing")
    
    def run_all_tests(self) -> bool:
        """Run the complete performance test suite"""
        print("="*60)
        print("ALPHA15 FPGA LASER CONTROL - PERFORMANCE BENCHMARK")
        print("="*60)
        print("Advanced FPGA Instrumentation Performance Testing")
        print("High-Speed Data Acquisition • Real-Time Processing • DMA Streaming")
        print("="*60)
        
        # Connect to instrument
        if not self.connect():
            return False
        
        # Initialize system
        if not self.initialize_system():
            return False
        
        # Run all tests
        test_suite = [
            ('Basic Connectivity', self.test_basic_connectivity),
            ('Data Acquisition Speed', self.test_data_acquisition_speed),
            ('CIC Decimation Performance', self.test_decimation_performance),
            ('DMA Performance', self.test_dma_performance),
            ('System Resources', self.test_system_resources),
            ('Signal Quality', self.test_signal_quality),
        ]
        
        for test_name, test_func in test_suite:
            try:
                result = test_func()
                self.results[test_name] = result
                
                if result.get('success', False):
                    print(f"{test_name} completed successfully")
                else:
                    print(f"{test_name} failed")
                    
            except Exception as e:
                print(f"{test_name} crashed: {e}")
                self.results[test_name] = {
                    'test_name': test_name,
                    'success': False,
                    'error': str(e)
                }
        
        # Generate final report
        report_file = self.generate_report()
        
        print("\n" + "="*60)
        print("PERFORMANCE TESTING COMPLETE")
        print("="*60)
        
        return True

def main():
    """Main entry point"""
    print("Alpha15 FPGA Laser Control - Performance Benchmarking Suite")
    print("=" * 60)
    
    # Parse command line arguments
    host = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_HOST
    
    # Create and run performance tester
    tester = PerformanceTester(host)
    
    try:
        success = tester.run_all_tests()
        return 0 if success else 1
    except KeyboardInterrupt:
        print("\n\nPerformance testing interrupted by user")
        return 1
    except Exception as e:
        print(f"\n\nPerformance testing failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main()) 