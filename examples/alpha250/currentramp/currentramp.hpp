/// Current Ramp Driver
///
/// Driver for controlling laser current ramp (temperature and laser power)
/// Hardware-based precise timing using DDS phase accumulator
/// Added: DMA-based high-rate ADC streaming for monitoring ramp response
/// (c) Koheron

#ifndef __DRIVERS_CURRENTRAMP_HPP__
#define __DRIVERS_CURRENTRAMP_HPP__

#include <context.hpp>
#include <boards/alpha250/drivers/precision-dac.hpp>
#include <boards/alpha250/drivers/clock-generator.hpp>
#include <boards/alpha250/drivers/ltc2157.hpp>
#include <array>
#include <cmath>
#include <vector>

// AXI DMA Registers (following adc-dac-dma example)
namespace Dma_regs {
    constexpr uint32_t s2mm_dmacr = 0x30;    // S2MM DMA Control register
    constexpr uint32_t s2mm_dmasr = 0x34;    // S2MM DMA Status register
    constexpr uint32_t s2mm_curdesc = 0x38;  // S2MM DMA Current Descriptor Pointer register
    constexpr uint32_t s2mm_taildesc = 0x40; // S2MM DMA Tail Descriptor Pointer register
}

// Scatter Gather Descriptor
namespace Sg_regs {
    constexpr uint32_t nxtdesc = 0x0;        // Next Descriptor Pointer
    constexpr uint32_t buffer_address = 0x8; // Buffer address
    constexpr uint32_t control = 0x18;       // Control
    constexpr uint32_t status = 0x1C;        // Status
}

// System Level Control Registers
namespace Sclr_regs {
    constexpr uint32_t sclr_unlock = 0x8;       // SLCR Write Protection Unlock
    constexpr uint32_t fpga0_clk_ctrl = 0x170;  // PL Clock 0 Output control
    constexpr uint32_t fpga1_clk_ctrl = 0x180;  // PL Clock 1 Output control
    constexpr uint32_t ocm_rst_ctrl = 0x238;    // OCM Software Reset Control
    constexpr uint32_t fpga_rst_ctrl = 0x240;   // FPGA Software Reset Control
    constexpr uint32_t ocm_cfg = 0x910;         // FPGA Software Reset Control
}

// DMA streaming parameters
constexpr uint32_t n_pts = 64 * 1024; // Number of words in one descriptor
constexpr uint32_t n_desc = 64; // Number of descriptors

class CurrentRamp
{
  public:
    CurrentRamp(Context& ctx_)
    : ctx(ctx_)
    , precision_dac(ctx.get<PrecisionDac>())
    , clk_gen(ctx.get<ClockGenerator>())
    , ltc2157(ctx.get<Ltc2157>())
    , ctl(ctx.mm.get<mem::control>())
    , sts(ctx.mm.get<mem::status>())
    , dma(ctx.mm.get<mem::dma>())
    , ram_s2mm(ctx.mm.get<mem::ram_s2mm>())
    , axi_hp0(ctx.mm.get<mem::axi_hp0>())
    , axi_hp2(ctx.mm.get<mem::axi_hp2>())
    , ocm_s2mm(ctx.mm.get<mem::ocm_s2mm>())
    , sclr(ctx.mm.get<mem::sclr>())
    , dc_voltage(0.0f)
    , dc_enabled(false)
    , ramp_offset(1.5f)
    , ramp_amplitude(1.0f)
    , ramp_frequency(10.0)
    , hardware_ramp_enabled(false)
    , streaming_active(false)
    , decimation_rate(2500)
    {
        // Get ADC sampling frequency for phase increment calculation
        fs_adc = clk_gen.get_adc_sampling_freq();
        
        // Initialize hardware ramp disabled
        ctl.write<reg::ramp_enable>(0);
        
        // Initialize DMA system (following adc-dac-dma example)
        init_dma_system();
        
        // Initialize CIC decimation rate to default (250MHz → 100kHz)
        set_decimation_rate(2500);
        
        ctx.log<INFO>("CurrentRamp: Hardware ramp generator initialized, fs_adc = %.1f MHz", fs_adc / 1e6);
        ctx.log<INFO>("CurrentRamp: DMA streaming system initialized");
    }

    ~CurrentRamp() {
        // Stop streaming if active
        if (streaming_active) {
            stop_streaming();
        }
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

    // === STATUS AND MONITORING ===
    
    uint32_t get_ramp_phase() {
        return sts.read<reg::ramp_phase>();
    }
    
    uint32_t get_cycle_count() {
        return sts.read<reg::cycle_count>();
    }
    
    // Calculate ramp position from phase accumulator (0.0 to 1.0)
    float get_ramp_position() {
        uint32_t phase = get_ramp_phase();
        return static_cast<float>(phase) / static_cast<float>(0xFFFFFFFF);
    }

    // === DMA STREAMING FUNCTIONS (Fixed implementation) ===
    
    void select_adc_channel(uint32_t channel) {
        ctl.write<reg::channel_select>(channel % 2);
        ctx.log<INFO>("ADC streaming channel selected: %u", channel % 2);
    }
    
    void set_decimation_rate(uint32_t rate) {
        // Validate decimation rate
        if (rate < 10 || rate > 8192) {
            ctx.log<ERROR>("Invalid decimation rate: %u (must be 10-8192)", rate);
            return;
        }
        
        decimation_rate = rate;
        ctl.write<reg::cic_rate>(rate);
        
        // Calculate actual output sample rate
        double decimated_fs = fs_adc / rate;
        ctx.log<INFO>("CIC decimation rate set to %u (%.0f MHz → %.0f kHz)", 
                      rate, fs_adc/1e6, decimated_fs/1e3);
    }
    
    uint32_t get_decimation_rate() {
        return decimation_rate;
    }
    
    double get_decimated_sample_rate() {
        return fs_adc / decimation_rate;
    }
    
    void start_streaming() {
        if (streaming_active) {
            ctx.log<WARNING>("DMA streaming already active");
            return;
        }
        
        // Clear RAM buffer
        for (uint32_t i = 0; i < n_pts * n_desc; i++) {
            ram_s2mm.write_reg(4*i, 0);
        }
        
        // Set up descriptors and start DMA (following adc-dac-dma exactly)
        set_descriptors();
        
        // Write address of the starting descriptor
        dma.write<Dma_regs::s2mm_curdesc>(mem::ocm_s2mm_addr + 0x0);
        
        // Start S2MM channel
        dma.set_bit<Dma_regs::s2mm_dmacr, 0>();
        
        // Write address of the tail descriptor (this starts the transfer)
        dma.write<Dma_regs::s2mm_taildesc>(mem::ocm_s2mm_addr + (n_desc-1) * 0x40);
        
        streaming_active = true;
        ctx.log<INFO>("DMA streaming started with %u descriptors", n_desc);
    }
    
    void stop_streaming() {
        if (!streaming_active) {
            ctx.log<WARNING>("DMA streaming is not active");
            return;
        }
        
        // Stop S2MM channel
        dma.clear_bit<Dma_regs::s2mm_dmacr, 0>();
        dma.write<Dma_regs::s2mm_taildesc>(mem::ocm_s2mm_addr + (n_desc-1) * 0x40);
        
        streaming_active = false;
        ctx.log<INFO>("DMA streaming stopped");
    }
    
    bool get_streaming_active() {
        return streaming_active;
    }
    
    uint32_t get_samples_captured() {
        if (!streaming_active) return 0;
        
        // For continuous DMA operation, we should return the buffer fill level
        // not a cumulative count that overflows
        uint32_t current_desc_addr = dma.read<Dma_regs::s2mm_curdesc>();
        if (current_desc_addr < mem::ocm_s2mm_addr) {
            ctx.log<WARNING>("Invalid descriptor address: 0x%08x", current_desc_addr);
            return 0;
        }
        
        uint32_t desc_offset = current_desc_addr - mem::ocm_s2mm_addr;
        uint32_t current_desc = desc_offset / 0x40;
        
        // Ensure descriptor index is in valid range
        current_desc = current_desc % n_desc;
        
        // Return current buffer position (samples available for reading)
        uint32_t buffer_samples = current_desc * n_pts;
        
        return buffer_samples;
    }
    
    uint32_t get_buffer_fill_level() {
        return get_samples_captured();
    }
    
    double get_streaming_sample_rate() {
        return get_decimated_sample_rate(); // Return decimated rate, not full ADC rate
    }
    
    uint32_t get_dma_status() {
        return dma.read<Dma_regs::s2mm_dmasr>();
    }
    
    uint32_t get_current_descriptor_address() {
        return dma.read<Dma_regs::s2mm_curdesc>();
    }
    
    // Fixed: Use the correct array reading pattern from adc-dac-dma
    auto& get_adc_stream_data() {
        adc_data = ram_s2mm.read_array<uint32_t, n_desc * n_pts>();
        return adc_data;
    }
    
    // Convert raw data to voltages - return array reference for recv_array compatibility
    auto& get_adc_stream_voltages(uint32_t num_samples) {
        if (num_samples > n_desc * n_pts) {
            num_samples = n_desc * n_pts;
            ctx.log<WARNING>("Requested samples exceeds buffer size, limiting to %u", num_samples);
        }
        
        // Get raw data using the working pattern
        auto& raw_data = get_adc_stream_data();
        
        // Convert to voltages in the voltage_data array
        for (uint32_t i = 0; i < num_samples; i++) {
            // Extract 16-bit ADC value from 32-bit word (lower 16 bits)
            uint16_t adc_raw = static_cast<uint16_t>(raw_data[i] & 0xFFFF);
            
            // Convert to signed and then to voltage (±1.8V range)
            int16_t adc_signed = static_cast<int16_t>(adc_raw);
            float voltage = (static_cast<float>(adc_signed) / 32768.0f) * 1.8f;
            
            voltage_data[i] = voltage;
        }
        
        ctx.log<INFO>("Retrieved %u voltage samples from DMA buffer", num_samples);
        return voltage_data;
    }

    // === TEST FUNCTIONS ===
    // For precision DAC testing - direct voltage control on all channels
    
    void set_test_voltage_channel_0(float voltage) {
        if (voltage < 0.0f || voltage > 2.5f) {
            ctx.log<ERROR>("Invalid voltage: %.3f V (must be 0-2.5V)", static_cast<double>(voltage));
            return;
        }
        precision_dac.set_dac_value_volts(0, voltage);
        ctx.log<INFO>("Test voltage channel 0: %.3f V", static_cast<double>(voltage));
    }
    
    void set_test_voltage_channel_1(float voltage) {
        if (voltage < 0.0f || voltage > 2.5f) {
            ctx.log<ERROR>("Invalid voltage: %.3f V (must be 0-2.5V)", static_cast<double>(voltage));
            return;
        }
        precision_dac.set_dac_value_volts(1, voltage);
        ctx.log<INFO>("Test voltage channel 1: %.3f V", static_cast<double>(voltage));
    }
    
    // Channel 2 is used by hardware ramp
    void set_test_voltage_channel_2(float voltage) {
        if (voltage < 0.0f || voltage > 2.5f) {
            ctx.log<ERROR>("Invalid voltage: %.3f V (must be 0-2.5V)", static_cast<double>(voltage));
            return;
        }
        
        if (hardware_ramp_enabled) {
            ctx.log<WARNING>("Hardware ramp is active - stopping ramp to set manual voltage");
            stop_ramp();
        }
        
        precision_dac.set_dac_value_volts(2, voltage);
        ctx.log<INFO>("Test voltage channel 2: %.3f V (hardware ramp disabled)", static_cast<double>(voltage));
    }

    // === ADC READING FUNCTIONS ===
    
    float get_photodiode_reading() {
        uint32_t adc_raw = sts.read<reg::adc0>();
        return adc_to_voltage(adc_raw);
    }
    
    auto get_timing_status() {
        return std::make_tuple(
            get_ramp_enabled(),
            get_ramp_frequency(),
            get_ramp_phase(),
            get_cycle_count(),
            get_ramp_position()
        );
    }

    // Raw ADC access
    uint32_t get_adc0() {
        return sts.read<reg::adc0>();
    }
    
    uint32_t get_adc1() {
        return sts.read<reg::adc1>();
    }
    
    std::array<uint32_t, 2> get_adc_both() {
        return {{get_adc0(), get_adc1()}};
    }
    
    float adc_to_voltage(uint32_t adc_raw) {
        // Convert 16-bit ADC to voltage (±1.8V range)
        int16_t signed_adc = static_cast<int16_t>(adc_raw & 0xFFFF);
        return (static_cast<float>(signed_adc) / 32768.0f) * 1.8f;
    }
    
    float get_adc0_voltage() {
        return adc_to_voltage(get_adc0());
    }
    
    float get_adc1_voltage() {
        return adc_to_voltage(get_adc1());
    }

  private:
    Context& ctx;
    PrecisionDac& precision_dac;
    ClockGenerator& clk_gen;
    Ltc2157& ltc2157;
    Memory<mem::control>& ctl;
    Memory<mem::status>& sts;
    
    // DMA memory interfaces
    Memory<mem::dma>& dma;
    Memory<mem::ram_s2mm>& ram_s2mm;
    Memory<mem::axi_hp0>& axi_hp0;
    Memory<mem::axi_hp2>& axi_hp2;
    Memory<mem::ocm_s2mm>& ocm_s2mm;
    Memory<mem::sclr>& sclr;
    
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
    
    // DMA streaming state
    bool streaming_active;
    uint32_t decimation_rate;
    
    // DMA data buffers (following adc-dac-dma pattern)
    std::array<uint32_t, n_desc * n_pts> adc_data;
    std::array<float, n_desc * n_pts> voltage_data;
    
    void init_dma_system() {
        // Unlock SCLR (following adc-dac-dma example)
        sclr.write<Sclr_regs::sclr_unlock>(0xDF0D);
        sclr.clear_bit<Sclr_regs::fpga_rst_ctrl, 1>();

        // Make sure that the width of the AXI HP port is 64 bit
        axi_hp0.clear_bit<0x0, 0>();
        axi_hp0.clear_bit<0x14, 0>();
        axi_hp2.clear_bit<0x0, 0>();
        axi_hp2.clear_bit<0x14, 0>();

        // Map the last 64 kB of OCM RAM to the high address space
        sclr.write<Sclr_regs::ocm_cfg>(0b1000);
    }
    
    void set_descriptor_s2mm(uint32_t idx, uint32_t buffer_address, uint32_t buffer_length) {
        ocm_s2mm.write_reg(0x40 * idx + Sg_regs::nxtdesc, mem::ocm_s2mm_addr + 0x40 * ((idx+1) % n_desc));
        ocm_s2mm.write_reg(0x40 * idx + Sg_regs::buffer_address, buffer_address);
        ocm_s2mm.write_reg(0x40 * idx + Sg_regs::control, buffer_length);
        ocm_s2mm.write_reg(0x40 * idx + Sg_regs::status, 0);
    }
    
    void set_descriptors() {
        for (uint32_t i = 0; i < n_desc; i++) {
            set_descriptor_s2mm(i, mem::ram_s2mm_addr + i * 4 * n_pts, 4 * n_pts);
        }
    }
};

#endif // __DRIVERS_CURRENTRAMP_HPP__ 