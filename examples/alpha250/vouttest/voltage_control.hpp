/// Voltage Control Driver
///
/// Simple driver for testing precision DAC voltage output
/// (c) Koheron

#ifndef __DRIVERS_VOLTAGE_CONTROL_HPP__
#define __DRIVERS_VOLTAGE_CONTROL_HPP__

#include <context.hpp>
#include <boards/alpha250/drivers/precision-dac.hpp>
#include <array>

class VoltageControl
{
  public:
    VoltageControl(Context& ctx_)
    : ctx(ctx_)
    , precision_dac(ctx.get<PrecisionDac>())
    , output_voltage(0.0f)
    , output_enabled(false)
    {
    }

    // === PRECISION DAC VOLTAGE CONTROL FUNCTIONS ===
    
    void set_voltage_output(float voltage) {
        if (voltage < 0.0f || voltage > 2.5f) {
            ctx.log<ERROR>("Invalid voltage range: %.3f V (must be 0-2.5V)", static_cast<double>(voltage));
            return;
        }
        
        output_voltage = voltage;
        if (output_enabled) {
            precision_dac.set_dac_value_volts(0, voltage);
        }
        ctx.log<INFO>("Voltage set to: %.3f V", static_cast<double>(voltage));
    }
    
    void enable_output(bool enable) {
        output_enabled = enable;
        if (enable) {
            precision_dac.set_dac_value_volts(0, output_voltage);
            ctx.log<INFO>("Output enabled: %.3f V", static_cast<double>(output_voltage));
        } else {
            precision_dac.set_dac_value_volts(0, 0.0f);
            ctx.log<INFO>("Output disabled");
        }
    }
    
    void toggle_output() {
        enable_output(!output_enabled);
    }
    
    // Convenience function for the test voltage (0.5V)
    void set_test_voltage() {
        set_voltage_output(0.5f);
        enable_output(true);
    }
    
    void disable_test_voltage() {
        enable_output(false);
    }
    
    // Getters
    float get_output_voltage() {
        return output_voltage;
    }
    
    bool is_output_enabled() {
        return output_enabled;
    }

  private:
    Context& ctx;
    PrecisionDac& precision_dac;  // Reference to precision DAC
    
    // State variables
    float output_voltage;
    bool output_enabled;
};

#endif // __DRIVERS_VOLTAGE_CONTROL_HPP__ 