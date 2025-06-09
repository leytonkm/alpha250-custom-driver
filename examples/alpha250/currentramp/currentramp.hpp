/// Current Ramp Driver
///
/// Driver for controlling laser current ramp (temperature and laser power)
/// Based on working vouttest pattern
/// (c) Koheron

#ifndef __DRIVERS_CURRENT_RAMP_HPP__
#define __DRIVERS_CURRENT_RAMP_HPP__

#include <context.hpp>
#include <boards/alpha250/drivers/precision-dac.hpp>
#include <array>
#include <thread>
#include <atomic>
#include <chrono>

class CurrentRamp
{
  public:
    CurrentRamp(Context& ctx_)
    : ctx(ctx_)
    , precision_dac(ctx.get<PrecisionDac>())
    , dc_voltage(0.0f)
    , dc_enabled(false)
    , ramp_offset(1.5f)
    , ramp_amplitude(1.0f)
    , ramp_frequency(10.0)
    , ramp_running(false)
    , ramp_sample_count(0)
    {
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
    // Uses precision DAC channel 1 for laser current ramp
    
    void set_ramp_offset(float offset) {
        if (offset < 0.0f || offset > 2.5f) {
            ctx.log<ERROR>("Invalid offset range: %.3f V (must be 0-2.5V)", static_cast<double>(offset));
            return;
        }
        ramp_offset = offset;
        ctx.log<INFO>("Ramp offset set to: %.3f V", static_cast<double>(offset));
    }
    
    void set_ramp_amplitude(float amplitude) {
        if (amplitude < 0.0f || amplitude > 2.5f) {
            ctx.log<ERROR>("Invalid amplitude range: %.3f V (must be 0-2.5V)", static_cast<double>(amplitude));
            return;
        }
        ramp_amplitude = amplitude;
        ctx.log<INFO>("Ramp amplitude set to: %.3f V", static_cast<double>(amplitude));
    }
    
    void set_ramp_frequency(double frequency) {
        if (frequency < 0.1 || frequency > 1000.0) {
            ctx.log<ERROR>("Invalid frequency: %.2f Hz (must be 0.1-1000 Hz)", frequency);
            return;
        }
        ramp_frequency = frequency;
        ctx.log<INFO>("Ramp frequency set to: %.2f Hz", frequency);
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
        ctx.log<INFO>("Ramp waveform configured: offset=%.3f V, amplitude=%.3f V, frequency=%.2f Hz", 
                      static_cast<double>(ramp_offset), static_cast<double>(ramp_amplitude), ramp_frequency);
    }
    
    void start_ramp() {
        if (ramp_running.load()) {
            ctx.log<WARNING>("Ramp already running");
            return;
        }
        
        // Safety check
        if (ramp_amplitude + ramp_offset > 2.5f) {
            ctx.log<ERROR>("Ramp amplitude + offset exceeds maximum voltage: %.3f V (must be ≤2.5V)", 
                          static_cast<double>(ramp_amplitude + ramp_offset));
            return;
        }
        
        ramp_running.store(true);
        ramp_sample_count = 0;
        
        ctx.log<INFO>("Current ramp started at %.2f Hz on Precision DAC Channel 1", ramp_frequency);
        
        // Start ramp thread
        ramp_thread = std::thread([this]() {
            const int samples_per_cycle = 1000;
            const double period_ms = 1000.0 / ramp_frequency;  // Period in milliseconds
            const double delay_ms = period_ms / samples_per_cycle;
            
            while (ramp_running.load()) {
                // Calculate sawtooth wave position (0 to 1)
                float progress = static_cast<float>(ramp_sample_count % samples_per_cycle) / static_cast<float>(samples_per_cycle - 1);
                
                // Sawtooth wave: linear ramp from 0 to 1, then jump back to 0
                float sawtooth = progress;  // 0 to 1 linearly over full cycle
                
                float voltage = ramp_offset + (sawtooth * ramp_amplitude);
                precision_dac.set_dac_value_volts(1, voltage);
                
                ramp_sample_count++;
                
                // For debugging - log occasionally
                if (ramp_sample_count % 200 == 0) {
                    ctx.log<INFO>("Ramp sample %d: %.3fV (sawtooth=%.3f)", ramp_sample_count, static_cast<double>(voltage), static_cast<double>(sawtooth));
                }
                
                std::this_thread::sleep_for(std::chrono::microseconds(static_cast<int>(delay_ms * 1000)));
            }
            
            // Set to 0V when stopped (no output)
            precision_dac.set_dac_value_volts(1, 0.0f);
            ctx.log<INFO>("Ramp thread stopped");
        });
    }
    
    void stop_ramp() {
        if (!ramp_running.load()) {
            ctx.log<WARNING>("Ramp is not running");
            return;
        }
        
        ramp_running.store(false);
        
        if (ramp_thread.joinable()) {
            ramp_thread.join();
        }
        
        // Set to 0V when stopped (no output)
        precision_dac.set_dac_value_volts(1, 0.0f);
        ctx.log<INFO>("Current ramp stopped - output set to 0V");
    }
    
    // Manual ramp control for testing
    void set_ramp_manual(float sawtooth_position) {
        // sawtooth_position: 0.0 = start (offset), 1.0 = peak (offset + amplitude)
        if (sawtooth_position < 0.0f) sawtooth_position = 0.0f;
        if (sawtooth_position > 1.0f) sawtooth_position = 1.0f;
        
        float voltage = ramp_offset + (sawtooth_position * ramp_amplitude);
        precision_dac.set_dac_value_volts(1, voltage);
        
        ctx.log<INFO>("Manual ramp: position=%.3f, voltage=%.3fV", 
                      static_cast<double>(sawtooth_position), 
                      static_cast<double>(voltage));
    }

    bool get_ramp_enabled() {
        return ramp_running.load();
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
        if (voltage >= 0.0f && voltage <= 2.5f) {
            precision_dac.set_dac_value_volts(1, voltage);
            ctx.log<INFO>("Test voltage set on Channel 1: %.3f V", static_cast<double>(voltage));
        } else {
            ctx.log<ERROR>("Invalid test voltage: %.3f V (must be 0-2.5V)", static_cast<double>(voltage));
        }
    }

    // === STATUS FUNCTIONS ===
    
    float get_photodiode_reading() {
        // This would read from ADC - placeholder for now
        return 0.0f;
    }

  private:
    Context& ctx;
    PrecisionDac& precision_dac;  // Reference to precision DAC (same pattern as vouttest)
    
    // DC control state
    float dc_voltage;
    bool dc_enabled;
    
    // Ramp control state
    float ramp_offset;
    float ramp_amplitude;
    double ramp_frequency;
    std::atomic<bool> ramp_running;
    int ramp_sample_count;
    std::thread ramp_thread;
};

#endif // __DRIVERS_CURRENT_RAMP_HPP__ 