/// Current Ramp Driver
///
/// Driver for controlling laser current ramp (temperature and laser power)
/// Hardware-based precise timing using DDS phase accumulator
/// (c) Koheron

#ifndef __DRIVERS_CURRENT_RAMP_HPP__
#define __DRIVERS_CURRENT_RAMP_HPP__

#include <context.hpp>
#include <boards/alpha250/drivers/precision-dac.hpp>
#include <boards/alpha250/drivers/clock-generator.hpp>
#include <array>
#include <cmath>
#include <thread>
#include <chrono>

class CurrentRamp
{
  public:
    CurrentRamp(Context& ctx_)
    : ctx(ctx_)
    , precision_dac(ctx.get<PrecisionDac>())
    , clk_gen(ctx.get<ClockGenerator>())
    , ctl(ctx.mm.get<mem::control>())
    , sts(ctx.mm.get<mem::status>())
    , adc_map(ctx.mm.get<mem::adc>())
    , dc_voltage(0.0f)
    , dc_enabled(false)
    , ramp_offset(1.5f)
    , ramp_amplitude(1.0f)
    , ramp_frequency(10.0)
    , hardware_ramp_enabled(false)
    {
        // Get ADC sampling frequency for phase increment calculation
        fs_adc = clk_gen.get_adc_sampling_freq();
        
        // Initialize hardware ramp disabled
        ctl.write<reg::ramp_enable>(0);
        
        ctx.log<INFO>("CurrentRamp: Hardware ramp generator initialized, fs_adc = %.1f MHz", fs_adc / 1e6);
    }

    // === DC TEMPERATURE CONTROL FUNCTIONS ===
    // Uses precision DAC channel 0 for DC temperature control
    
    void set_temperature_dc_voltage(float voltage) {
        if (voltage < 0.0f || voltage > 2.5f) {
            ctx.log<ERROR>("Invalid voltage range: %.3f V (must be 0-2.5V)", static_cast<double>(voltage));
            return;
        }
        
        dc_voltage = voltage;
        if (dc_enabled) {
            precision_dac.set_dac_value_volts(0, voltage);
        }
        ctx.log<INFO>("DC temperature voltage set to: %.3f V", static_cast<double>(voltage));
    }
    
    void enable_temperature_dc_output(bool enable) {
        dc_enabled = enable;
        if (enable) {
            precision_dac.set_dac_value_volts(0, dc_voltage);
            ctx.log<INFO>("DC output enabled: %.3f V", static_cast<double>(dc_voltage));
        } else {
            precision_dac.set_dac_value_volts(0, 0.0f);
            ctx.log<INFO>("DC output disabled");
        }
    }
    
    float get_temperature_dc_voltage() {
        return dc_voltage;
    }
    
    bool get_temperature_dc_enabled() {
        return dc_enabled;
    }

    // === LASER CURRENT RAMP CONTROL FUNCTIONS ===
    // Uses hardware ramp generator connected to precision DAC channel 2
    
    void set_ramp_offset(float offset) {
        if (offset < 0.0f || offset > 2.5f) {
            ctx.log<ERROR>("Invalid offset range: %.3f V (must be 0-2.5V)", static_cast<double>(offset));
            return;
        }
        ramp_offset = offset;
        
        // Convert voltage to DAC codes (assuming 16-bit DAC, 2.5V range)
        uint16_t offset_dac = static_cast<uint16_t>((offset / 2.5f) * 65535.0f);
        ctl.write<reg::ramp_offset_reg>(offset_dac);
        
        ctx.log<INFO>("Ramp offset set to: %.3f V (DAC code: %u)", static_cast<double>(offset), offset_dac);
    }
    
    void set_ramp_amplitude(float amplitude) {
        if (amplitude < 0.0f || amplitude > 2.5f) {
            ctx.log<ERROR>("Invalid amplitude range: %.3f V (must be 0-2.5V)", static_cast<double>(amplitude));
            return;
        }
        ramp_amplitude = amplitude;
        
        // Convert voltage to DAC codes (assuming 16-bit DAC, 2.5V range)
        uint16_t amplitude_dac = static_cast<uint16_t>((amplitude / 2.5f) * 65535.0f);
        ctl.write<reg::ramp_amplitude_reg>(amplitude_dac);
        
        ctx.log<INFO>("Ramp amplitude set to: %.3f V (DAC code: %u)", static_cast<double>(amplitude), amplitude_dac);
    }
    
    void set_ramp_frequency(double frequency) {
        if (frequency < 0.001 || frequency > 1000.0) {
            ctx.log<ERROR>("Invalid frequency: %.3f Hz (must be 0.001-1000 Hz)", frequency);
            return;
        }
        ramp_frequency = frequency;
        
        // Calculate phase increment for DDS
        // Phase increment = (desired_freq * 2^32) / sampling_freq
        double phase_inc_factor = (1ULL << 32) / fs_adc;
        uint32_t phase_increment = static_cast<uint32_t>(frequency * phase_inc_factor);
        
        ctl.write<reg::ramp_freq_incr>(phase_increment);
        
        ctx.log<INFO>("Ramp frequency set to: %.3f Hz (phase increment: %u)", frequency, phase_increment);
    }
    
    float get_ramp_offset() {
        return ramp_offset;
    }
    
    float get_ramp_amplitude() {
        return ramp_amplitude;
    }
    
    double get_ramp_frequency() {
        return ramp_frequency;
    }
    
    void generate_ramp_waveform() {
        // Safety check
        if (ramp_amplitude + ramp_offset > 2.5f) {
            ctx.log<ERROR>("Ramp amplitude + offset exceeds maximum voltage: %.3f V (must be ≤2.5V)", 
                          static_cast<double>(ramp_amplitude + ramp_offset));
            return;
        }
        
        // Update hardware registers
        set_ramp_offset(ramp_offset);
        set_ramp_amplitude(ramp_amplitude);
        set_ramp_frequency(ramp_frequency);
        
        ctx.log<INFO>("Hardware ramp waveform configured: offset=%.3f V, amplitude=%.3f V, frequency=%.3f Hz", 
                      static_cast<double>(ramp_offset), static_cast<double>(ramp_amplitude), ramp_frequency);
    }
    
    void start_ramp() {
        // Safety check
        if (ramp_amplitude + ramp_offset > 2.5f) {
            ctx.log<ERROR>("Ramp amplitude + offset exceeds maximum voltage: %.3f V (must be ≤2.5V)", 
                          static_cast<double>(ramp_amplitude + ramp_offset));
            return;
        }
        
        if (hardware_ramp_enabled) {
            ctx.log<WARNING>("Hardware ramp already running");
            return;
        }
        
        // Configure hardware registers
        generate_ramp_waveform();
        
        // Reset phase accumulator and cycle counter
        ctl.write<reg::ramp_reset>(1);
        ctl.write<reg::ramp_reset>(0);
        
        // Enable hardware ramp generator
        ctl.write<reg::ramp_enable>(1);
        hardware_ramp_enabled = true;
        
        ctx.log<INFO>("Hardware ramp started at %.3f Hz", ramp_frequency);
    }
    
    void stop_ramp() {
        if (!hardware_ramp_enabled) {
            ctx.log<WARNING>("Hardware ramp is not running");
            return;
        }
        
        // Disable hardware ramp generator (will output 0V)
        ctl.write<reg::ramp_enable>(0);
        hardware_ramp_enabled = false;
        
        ctx.log<INFO>("Hardware ramp stopped - output set to 0V");
    }
    
    // Manual ramp control for testing
    void set_ramp_manual(float sawtooth_position) {
        // sawtooth_position: 0.0 = start (offset), 1.0 = peak (offset + amplitude)
        if (sawtooth_position < 0.0f) sawtooth_position = 0.0f;
        if (sawtooth_position > 1.0f) sawtooth_position = 1.0f;
        
        // For manual control, use precision DAC directly (bypass hardware ramp)
        stop_ramp();  // Disable hardware ramp first
        
        float voltage = ramp_offset + (sawtooth_position * ramp_amplitude);
        precision_dac.set_dac_value_volts(2, voltage);
        
        ctx.log<INFO>("Manual ramp: position=%.3f, voltage=%.3fV", 
                      static_cast<double>(sawtooth_position), 
                      static_cast<double>(voltage));
    }

    bool get_ramp_enabled() {
        return hardware_ramp_enabled;
    }
    
    // === HARDWARE STATUS FUNCTIONS ===
    
    uint32_t get_ramp_phase() {
        return sts.read<reg::ramp_phase>();
    }
    
    uint32_t get_cycle_count() {
        return sts.read<reg::cycle_count>();
    }
    
    // Get current ramp output normalized to 0-1 range
    float get_ramp_position() {
        uint32_t phase = get_ramp_phase();
        return static_cast<float>(phase) / static_cast<float>(0xFFFFFFFF);
    }

    // === MANUAL TESTING FUNCTIONS ===
    
    void set_test_voltage_channel_0(float voltage) {
        if (voltage >= 0.0f && voltage <= 2.5f) {
            precision_dac.set_dac_value_volts(0, voltage);
            ctx.log<INFO>("Test voltage set on Channel 0: %.3f V", static_cast<double>(voltage));
        } else {
            ctx.log<ERROR>("Invalid test voltage: %.3f V (must be 0-2.5V)", static_cast<double>(voltage));
        }
    }

    void set_test_voltage_channel_1(float voltage) {
        // Note: This will conflict with hardware ramp if enabled
        if (hardware_ramp_enabled) {
            ctx.log<WARNING>("Hardware ramp is enabled - manual voltage may be overridden");
        }
        
        if (voltage >= 0.0f && voltage <= 2.5f) {
            precision_dac.set_dac_value_volts(1, voltage);
            ctx.log<INFO>("Test voltage set on Channel 1: %.3f V", static_cast<double>(voltage));
        } else {
            ctx.log<ERROR>("Invalid test voltage: %.3f V (must be 0-2.5V)", static_cast<double>(voltage));
        }
    }

    void set_test_voltage_channel_2(float voltage) {
        // Note: This will conflict with hardware ramp if enabled
        if (hardware_ramp_enabled) {
            ctx.log<WARNING>("Hardware ramp is enabled - manual voltage may be overridden");
        }
        
        if (voltage >= 0.0f && voltage <= 2.5f) {
            precision_dac.set_dac_value_volts(2, voltage);
            ctx.log<INFO>("Test voltage set on Channel 2: %.3f V", static_cast<double>(voltage));
        } else {
            ctx.log<ERROR>("Invalid test voltage: %.3f V (must be 0-2.5V)", static_cast<double>(voltage));
        }
    }

    // === STATUS FUNCTIONS ===
    
    // Direct ADC reading using Precision ADC (8-channel SPI ADC)
    float get_photodiode_voltage() {
        // Get precision ADC values (8 channels)
        auto adc_values = precision_adc.get_adc_values();
        
        // Return channel 0 by default (can be changed with set_photodiode_channel)
        return adc_values[photodiode_channel];
    }
    
    uint32_t get_photodiode_raw() {
        // For precision ADC, we don't have raw values, so convert voltage back to a representative raw value
        float voltage = get_photodiode_voltage();
        // Assuming ±1.25V range for precision ADC (typical for AD7124)
        return static_cast<uint32_t>((voltage / 1.25f + 1.0f) * 32768.0f);
    }
    
    void set_photodiode_channel(uint32_t channel) {
        if (channel < 8) {  // Precision ADC has 8 channels
            photodiode_channel = channel;
            ctx.log<INFO>("Selected precision ADC channel: %u", channel);
        } else {
            ctx.log<ERROR>("Invalid precision ADC channel: %u (must be 0-7)", channel);
        }
    }
    
    // Get voltage from specific precision ADC channel
    float get_photodiode_precision(uint32_t channel) {
        auto adc_values = precision_adc.get_adc_values();
        if (channel < 8) {
            return adc_values[channel];
        } else {
            ctx.log<ERROR>("Invalid precision ADC channel: %u", channel);
            return 0.0f;
        }
    }
    
    float get_photodiode_reading() {
        // Legacy function - redirect to new implementation
        return get_photodiode_voltage();
    }
    
    // Get hardware timing diagnostics
    auto get_timing_status() {
        return std::make_tuple(
            fs_adc,
            get_ramp_phase(),
            get_cycle_count(), 
            get_ramp_position(),
            hardware_ramp_enabled
        );
    }

    // === CONTINUOUS BRAM DATA ACQUISITION ===
    // Functions for 10kHz decimated continuous capture using software decimation
    
    void trigger_acquisition() {
        ctl.set_bit<reg::trig, 0>();
        std::this_thread::sleep_for(std::chrono::microseconds(1));
        ctl.clear_bit<reg::trig, 0>();
    }
    
    bool is_acquisition_complete() {
        // For continuous acquisition, always return true after a brief delay
        return true;
    }
    
    // Get 1000 ADC samples (representing 100ms of 10kHz data = 1 complete period)
    std::array<uint32_t, 1000> get_adc_samples_1000() {
        // Read entire BRAM using the proven pattern from working examples
        // ADC BRAM now contains: [ramp_output:16][adc_input:16]
        auto full_buffer = adc_map.read_array<uint32_t, adc_size>();
        
        // For 10kHz effective rate from 250MHz capture:
        // Take every 25,000th sample to achieve 10kHz decimation
        // 250MHz / 25,000 = 10kHz
        std::array<uint32_t, 1000> samples;
        
        // Get current write position
        uint32_t current_addr = sts.read<reg::acquisition_count>() % adc_size;
        
        // Start from a position that gives us good coverage of the circular buffer
        uint32_t start_addr = (current_addr >= 25000000) ? (current_addr - 25000000) : 0;
        
        for (size_t i = 0; i < 1000; i++) {
            // Take every 25,000th sample for 10kHz decimation
            uint32_t addr = (start_addr + i * 25000) % adc_size;
            uint32_t combined_data = full_buffer[addr];
            
            // Extract ADC input from lower 16 bits
            // Format: [ramp_output:16][adc_input:16]
            uint32_t adc_input = combined_data & 0xFFFF;
            samples[i] = adc_input;
        }
        
        ctx.log<INFO>("Read 1000 decimated ADC samples from BRAM lower bits (10kHz rate)");
        return samples;
    }
    
    // Get 1000 DAC samples (extracted from upper 16 bits of ADC BRAM)
    std::array<uint32_t, 1000> get_dac_samples_1000() {
        // Following corrected Alpha250 pattern: ramp output is captured in ADC BRAM upper 16 bits
        // Read entire ADC BRAM (which contains both ADC input and ramp output)
        auto full_buffer = adc_map.read_array<uint32_t, adc_size>();
        
        // For 10kHz effective rate from 250MHz capture:
        // Take every 25,000th sample to achieve 10kHz decimation
        // 250MHz / 25,000 = 10kHz
        std::array<uint32_t, 1000> samples;
        
        // Get current write position
        uint32_t current_addr = sts.read<reg::acquisition_count>() % adc_size;
        
        // Start from a position that gives us good coverage of the circular buffer
        uint32_t start_addr = (current_addr >= 25000000) ? (current_addr - 25000000) : 0;
        
        for (size_t i = 0; i < 1000; i++) {
            // Take every 25,000th sample for 10kHz decimation
            uint32_t addr = (start_addr + i * 25000) % adc_size;
            uint32_t combined_data = full_buffer[addr];
            
            // Extract ramp output from upper 16 bits
            // Format: [ramp_output:16][adc_input:16]
            uint32_t ramp_output = (combined_data >> 16) & 0xFFFF;
            samples[i] = ramp_output;
        }
        
        ctx.log<INFO>("Read 1000 decimated ramp output samples from ADC BRAM upper bits (10kHz rate)");
        return samples;
    }
    
    // Get 10,000 ADC samples (representing 1000ms of 10kHz data = 10 complete periods)
    std::array<uint32_t, 10000> get_adc_samples_10000() {
        // Read entire BRAM using the proven pattern from working examples
        // ADC BRAM now contains: [ramp_output:16][adc_input:16]
        auto full_buffer = adc_map.read_array<uint32_t, adc_size>();
        
        // For 10kHz effective rate from 250MHz capture:
        // Take every 25,000th sample to achieve 10kHz decimation
        // 250MHz / 25,000 = 10kHz
        std::array<uint32_t, 10000> samples;
        
        // Get current write position
        uint32_t current_addr = sts.read<reg::acquisition_count>() % adc_size;
        
        // Start from a position that gives us good coverage of the circular buffer
        uint32_t start_addr = (current_addr >= 250000000) ? (current_addr - 250000000) : 0;
        
        for (size_t i = 0; i < 10000; i++) {
            // Take every 25,000th sample for 10kHz decimation
            uint32_t addr = (start_addr + i * 25000) % adc_size;
            uint32_t combined_data = full_buffer[addr];
            
            // Extract ADC input from lower 16 bits
            // Format: [ramp_output:16][adc_input:16]
            uint32_t adc_input = combined_data & 0xFFFF;
            samples[i] = adc_input;
        }
        
        ctx.log<INFO>("Read 10,000 decimated ADC samples from BRAM lower bits (10kHz rate)");
        return samples;
    }
    
    // Get 10,000 DAC samples (extracted from upper 16 bits of ADC BRAM)
    std::array<uint32_t, 10000> get_dac_samples_10000() {
        // Following corrected Alpha250 pattern: ramp output is captured in ADC BRAM upper 16 bits
        // Read entire ADC BRAM (which contains both ADC input and ramp output)
        auto full_buffer = adc_map.read_array<uint32_t, adc_size>();
        
        // For 10kHz effective rate from 250MHz capture:
        // Take every 25,000th sample to achieve 10kHz decimation
        // 250MHz / 25,000 = 10kHz
        std::array<uint32_t, 10000> samples;
        
        // Get current write position
        uint32_t current_addr = sts.read<reg::acquisition_count>() % adc_size;
        
        // Start from a position that gives us good coverage of the circular buffer
        uint32_t start_addr = (current_addr >= 250000000) ? (current_addr - 250000000) : 0;
        
        for (size_t i = 0; i < 10000; i++) {
            // Take every 25,000th sample for 10kHz decimation
            uint32_t addr = (start_addr + i * 25000) % adc_size;
            uint32_t combined_data = full_buffer[addr];
            
            // Extract ramp output from upper 16 bits
            // Format: [ramp_output:16][adc_input:16]
            uint32_t ramp_output = (combined_data >> 16) & 0xFFFF;
            samples[i] = ramp_output;
        }
        
        ctx.log<INFO>("Read 10,000 decimated ramp output samples from ADC BRAM upper bits (10kHz rate)");
        return samples;
    }
    
    // Get acquisition status
    uint32_t get_acquisition_count() {
        return sts.read<reg::acquisition_count>();
    }
    
    // Get buffer sizes for web interface
    uint32_t get_adc_size() {
        return adc_size;
    }
    
    uint32_t get_dac_size() {
        return adc_size;  // DAC data is now extracted from ADC BRAM
    }

  private:
    Context& ctx;
    PrecisionDac& precision_dac;
    ClockGenerator& clk_gen;
    Memory<mem::control>& ctl;
    Memory<mem::status>& sts;
    
    // BRAM memory map for continuous data acquisition
    // ADC BRAM contains both ADC input (lower 16 bits) and ramp output (upper 16 bits)
    Memory<mem::adc>& adc_map;
    
    // Clock and timing
    double fs_adc;  // ADC sampling frequency
    
    // BRAM buffer size (calculated from memory range)
    static constexpr uint32_t adc_size = mem::adc_range / sizeof(uint32_t);
    
    // DC control state
    float dc_voltage;
    bool dc_enabled;
    
    // Ramp control state
    float ramp_offset;
    float ramp_amplitude;
    double ramp_frequency;
    bool hardware_ramp_enabled;
};

#endif // __DRIVERS_CURRENT_RAMP_HPP__ 