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
#include <boards/alpha250/drivers/ltc2157.hpp>
#include <array>
#include <cmath>
#include <chrono>
#include <thread>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

class CurrentRamp
{
  public:
    static constexpr uint32_t adc_buffer_size = mem::adc_range/sizeof(uint32_t);  // Use actual BRAM size

    CurrentRamp(Context& ctx_)
    : ctx(ctx_)
    , precision_dac(ctx.get<PrecisionDac>())
    , clk_gen(ctx.get<ClockGenerator>())
    , ltc2157(ctx.get<Ltc2157>())
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
        
        ctx.log<INFO>("CurrentRamp initialized: fs_adc=%.0f Hz, buffer_size=%u", fs_adc, adc_buffer_size);
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
    
    // === ADC OSCILLOSCOPE FUNCTIONS ===
    // Real ADC data acquisition using BRAM
    
    void trigger_adc_acquisition() {
        ctx.log<INFO>("Triggering ADC acquisition...");
        ctl.set_bit<reg::trig, 0>();
        ctl.clear_bit<reg::trig, 0>();
    }
    
    std::array<uint32_t, adc_buffer_size> get_adc_data() {
        trigger_adc_acquisition();
        ctx.log<INFO>("Reading ADC data from BRAM, buffer_size=%u", adc_buffer_size);
        return adc_map.read_array<uint32_t, adc_buffer_size>();
    }
    
    std::array<float, adc_buffer_size> get_adc_data_volts() {
        ctx.log<INFO>("get_adc_data_volts called, buffer_size=%u", adc_buffer_size);
        
        // Get raw ADC data
        auto raw_data = get_adc_data();
        
        // Get calibration parameters for channel 0
        float gain = ltc2157.get_gain(0);  // LSB per Volt
        float offset = ltc2157.get_offset(0);  // LSB offset
        
        // Convert to voltages using LTC2157 calibration
        std::array<float, adc_buffer_size> voltage_data;
        for (size_t i = 0; i < adc_buffer_size; i++) {
            // Use channel 0 data (extract from 32-bit word)
            uint16_t adc_value = static_cast<uint16_t>(raw_data[i] & 0xFFFF);
            
            // Convert to signed 16-bit value
            int16_t signed_adc = static_cast<int16_t>(adc_value);
            
            // Apply calibration: voltage = (adc_value - offset) / gain
            voltage_data[i] = (static_cast<float>(signed_adc) - offset) / gain;
        }
        
        ctx.log<INFO>("Converted ADC data to voltages: %u samples", adc_buffer_size);
        return voltage_data;
    }
    
    uint32_t get_adc_buffer_size() const {
        return adc_buffer_size;
    }
    
    // Return sampling rate for time axis calculation  
    double get_adc_sampling_rate() const {
        return fs_adc;
    }
    
    // Oscilloscope configuration
    void set_time_range(double time_range_ms) {
        if (time_range_ms < 0.1 || time_range_ms > 1000.0) {
            ctx.log<ERROR>("Invalid time range: %.3f ms (must be 0.1-1000 ms)", time_range_ms);
            return;
        }
        oscilloscope_time_range = time_range_ms;
        ctx.log<INFO>("Oscilloscope time range set to: %.3f ms", time_range_ms);
    }
    
    double get_time_range() const {
        return oscilloscope_time_range;
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
    
    float get_photodiode_reading() {
        // This would read from ADC - placeholder for now
        return 0.0f;
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

  private:
    Context& ctx;
    PrecisionDac& precision_dac;
    ClockGenerator& clk_gen;
    Ltc2157& ltc2157;
    Memory<mem::control>& ctl;
    Memory<mem::status>& sts;
    Memory<mem::adc>& adc_map;
    
    // Clock and timing
    double fs_adc;  // ADC sampling frequency
    
    // DC control state
    float dc_voltage;
    bool dc_enabled;
    
    // Ramp control state
    float ramp_offset;
    float ramp_amplitude;
    double ramp_frequency;
    bool hardware_ramp_enabled;
    
    // Oscilloscope state
    double oscilloscope_time_range = 10.0;  // Default 10ms
};

#endif // __DRIVERS_CURRENT_RAMP_HPP__ 