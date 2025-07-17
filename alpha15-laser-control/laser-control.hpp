/// Laser-Control Driver for the Alpha15 FPGA instrument
///
/// Driver for controlling laser current ramp (temperature and laser power)
/// Hardware-based precise timing using DDS phase accumulator
/// Added: DMA-based high-rate ADC streaming for monitoring ramp response
/// See block_design.tcl for vivado configuration
/// (c) Koheron

#ifndef __DRIVERS_CURRENTRAMP_HPP__
#define __DRIVERS_CURRENTRAMP_HPP__

#include <cstdint>
#include <context.hpp>
#include <boards/alpha250/drivers/precision-dac.hpp>
#include <boards/alpha15/drivers/clock-generator.hpp>
#include <boards/alpha15/drivers/ltc2387.hpp>
#include <array>
#include <cmath>
#include <vector>

// AXI DMA Registers
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
#define LASER_CONTROL_N_PTS 1024
constexpr uint32_t n_pts = LASER_CONTROL_N_PTS;      // Number of 32-bit samples per descriptor
constexpr uint32_t n_desc = 512; // Number of descriptors - could increase this?

class CurrentRamp
{
  public:
    CurrentRamp(Context& ctx_)
    : ctx(ctx_)
    , precision_dac(ctx.get<PrecisionDac>())
    , clk_gen(ctx.get<ClockGenerator>())
    , ltc2387(ctx.get<Ltc2387>())
    , ctl(ctx.mm.get<mem::control>())
    , sts(ctx.mm.get<mem::status>())
    , dma(ctx.mm.get<mem::dma>())
    , ram_s2mm(ctx.mm.get<mem::ram_s2mm>())
    , ocm_s2mm(ctx.mm.get<mem::ocm_s2mm>())
    , ocm_mm2s(ctx.mm.get<mem::ocm_mm2s>())
    , sclr(ctx.mm.get<mem::sclr>())
    , dc_voltage(0.0f)
    , dc_enabled(false)
    , ramp_offset(1.5f)
    , ramp_amplitude(1.0f)
    , ramp_frequency(10.0)
    , hardware_ramp_enabled(false)
    , streaming_active(false)
    , decimation_rate(100)
    {
        // Alpha15 clock generator returns std::array<double,2>
        fs_adc = clk_gen.get_adc_sampling_freq()[0];
        
        // Initialize hardware ramp disabled
        ctl.write<reg::ramp_enable>(0);
        
        // Initialize DMA system 
        init_dma_system();
        
        // Initialize CIC decimation rate to default
        set_decimation_rate(100);

        // Set ADC to 8Vpp range 
        ltc2387.range_select(0, 1); // Set channel 0 to 8Vpp
        ltc2387.range_select(1, 1); // Set channel 1 to 8Vpp
        
        ctx.log<INFO>("CurrentRamp: Hardware ramp generator initialized, fs_adc = %.1f MHz", fs_adc / 1e6);
        ctx.log<INFO>("CurrentRamp: DMA streaming system initialized");
        ctx.log<INFO>("CurrentRamp: ADC range set to 8Vpp");
    }

    ~CurrentRamp() {
        // Stop streaming if active
        if (streaming_active) {
            stop_streaming();
        }
    }

    // === DC TEMPERATURE CONTROL FUNCTIONS ===
    // Uses precision DAC channel 0 for DC temperature control (150 mV works well for 60 degrees celsius on our setup)
    
    void set_temperature_dc_voltage(float voltage) {
        if (voltage < 0.0f || voltage > 2.5f) {
            ctx.log<ERROR>("Invalid voltage range: %.3f V (must be 0-2.5V)", static_cast<double>(voltage));
            return;
        }
        
        dc_voltage = voltage;
        
        if (dc_enabled) {
            precision_dac.set_dac_value_volts(0, dc_voltage);
        }
        
        ctx.log<INFO>("DC temperature voltage set to: %.3f V", static_cast<double>(voltage));
    }
    
    void enable_temperature_dc_output(bool enable) {
        dc_enabled = enable;
        
        if (dc_enabled) {
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
    // Uses hardware ramp generator connected to precision DAC channel 2 (voltage settings depend on the amplifier gain) targeting ~3.8 V w/ 250 mV amplitude
    
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
        if (frequency < 0.001 || frequency > 1000000.0) {
            ctx.log<ERROR>("Invalid frequency: %.3f Hz (must be 0.001-1000000 Hz)", frequency);
            return;
        }
        
        // Update the member variable
        ramp_frequency = frequency;
        
        // Calculate phase increment for DDS
        // Phase increment = (desired_freq * 2^32) / DDS_clock_freq
        // NOTE: DDS runs at adc_clk (240 MHz), which is different from the ADC sampling frequency (15 MHz)
        constexpr double dds_clock_freq = 240e6;  // 240 MHz from config adc_clk parameter (verify with config.yml)
        double phase_inc_factor = (1ULL << 32) / dds_clock_freq;
        uint32_t phase_increment = static_cast<uint32_t>(frequency * phase_inc_factor);
        
        // Safety check for phase increment overflow (should not exceed 2^31 - 1)
        if (phase_increment > 0x7FFFFFFF) {
            ctx.log<ERROR>("Frequency too high: %.3f Hz results in phase increment overflow", frequency);
            return;
        }
        
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
        if (rate < 10 || rate > 8192) {
            ctx.log<ERROR>("Invalid decimation rate: %u (must be 10-8192)", rate);
            return;
        }
        
        decimation_rate = rate;
        
        // Sample rate calculation: fs_adc / (CIC_rate × FIR_rate)
        // FIR has fixed decimation rate of 2, CIC has programmable rate
        double fs_decimated = fs_adc / (2.0 * static_cast<double>(rate));
        
        ctx.log<INFO>("CIC decimation rate set to: %u", rate);
        ctx.log<INFO>("Effective sample rate: %.1f kS/s", fs_decimated / 1000.0);
        
        // Write to CIC rate control register
        ctl.write<reg::decimation_rate>(rate);
    }
    
    uint32_t get_decimation_rate() {
        return decimation_rate;
    }
    
    double get_decimated_sample_rate() {
        // Account for both CIC and FIR (factor of 2) decimation
        return fs_adc / (2.0 * static_cast<double>(decimation_rate));
    }
    
    void start_streaming() {
        if (streaming_active) {
            ctx.log<WARNING>("DMA streaming already active");
            return;
        }
        
        // Buffer is circular and will be overwritten by DMA.  No need to pre-clear it here – doing so
        // saved >100 ms but produced a blank gap at the start of every acquisition.
        
        // Set up descriptors and start DMA (following adc-dac-dma exactly)
        set_descriptors();
        
        // Write address of the starting descriptor
        dma.write<Dma_regs::s2mm_curdesc>(mem::ocm_mm2s_addr + 0x0);
        
        // Start S2MM channel with cyclic mode enabled
        dma.set_bit<Dma_regs::s2mm_dmacr, 4>(); // Enable cyclic mode
        dma.set_bit<Dma_regs::s2mm_dmacr, 0>(); // Start DMA
        
        // Write address of the tail descriptor (this starts the transfer)
        dma.write<Dma_regs::s2mm_taildesc>(mem::ocm_mm2s_addr + (n_desc-1) * 0x40);
        
        streaming_active = true;
        ctx.log<INFO>("DMA streaming started with %u descriptors in cyclic mode", n_desc);
    }
    
    void stop_streaming() {
        if (!streaming_active) {
            ctx.log<WARNING>("DMA streaming is not active");
            return;
        }
        
        // Stop S2MM channel
        dma.clear_bit<Dma_regs::s2mm_dmacr, 0>();
        dma.write<Dma_regs::s2mm_taildesc>(mem::ocm_mm2s_addr + (n_desc-1) * 0x40);
        
        streaming_active = false;
        ctx.log<INFO>("DMA streaming stopped");
    }
    
    bool get_streaming_active() {
        return streaming_active;
    }
    
    // Alternative function names for compatibility
    void start_adc_streaming() {
        start_streaming();
    }
    
    void stop_adc_streaming() {
        stop_streaming();
    }
    
    bool is_adc_streaming_active() {
        return get_streaming_active();
    }
    
    void set_cic_decimation_rate(uint32_t rate) {
        set_decimation_rate(rate);
    }
    
    uint32_t get_cic_decimation_rate() {
        return get_decimation_rate();
    }
    
    uint32_t get_samples_captured() {
        if (!streaming_active) return 0;
        
        // For continuous DMA operation, we should return the buffer fill level
        // not a cumulative count that overflows
        uint32_t current_desc_addr = dma.read<Dma_regs::s2mm_curdesc>();
        if (current_desc_addr < mem::ocm_mm2s_addr) {
            ctx.log<WARNING>("Invalid descriptor address: 0x%08x", current_desc_addr);
            return 0;
        }
        
        uint32_t desc_offset = current_desc_addr - mem::ocm_mm2s_addr;
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
    
    // === ADVANCED DMA DIAGNOSTICS ===
    
    uint32_t get_dma_status_register() {
        return dma.read<Dma_regs::s2mm_dmasr>();
    }
    
    uint32_t get_current_descriptor_index() {
        uint32_t current_desc_addr = dma.read<Dma_regs::s2mm_curdesc>();
        
        // Validate descriptor address
        if (current_desc_addr == 0 || current_desc_addr < mem::ocm_mm2s_addr) {
            ctx.log<WARNING>("Invalid descriptor address: 0x%08x", current_desc_addr);
            return 0;
        }
        
        uint32_t desc_offset = current_desc_addr - mem::ocm_mm2s_addr;
        uint32_t desc_idx = (desc_offset / 0x40) % n_desc;
        
        // Additional validation
        if (desc_idx >= n_desc) {
            ctx.log<WARNING>("Descriptor index out of range: %u (max %u)", desc_idx, n_desc - 1);
            return 0;
        }
        
        return desc_idx;
    }
    
    uint32_t get_buffer_position() {
        uint32_t desc_idx = get_current_descriptor_index();
        return desc_idx * n_pts;
    }
    
    bool get_dma_running() {
        uint32_t status_reg = get_dma_status_register();
        return (status_reg & 0x1) != 0;  // RS bit
    }
    
    bool get_dma_idle() {
        uint32_t status_reg = get_dma_status_register();
        return (status_reg & 0x2) != 0;  // Idle bit
    }
    
    bool get_dma_error() {
        uint32_t status_reg = get_dma_status_register();
        return (status_reg & 0x70) != 0;  // Error bits
    }
    
    // Improved sample counting with wraparound handling
    uint32_t get_samples_captured_accurate() {
        // For circular buffer, we can't count total samples easily
        // Instead, return current buffer fill level
        return get_buffer_position();
    }
    
    // Get current buffer fill percentage
    float get_buffer_fill_percentage() {
        uint32_t total_buffer_size = n_desc * n_pts;
        uint32_t current_fill = get_samples_captured_accurate();
        return (static_cast<float>(current_fill) / static_cast<float>(total_buffer_size)) * 100.0f;
    }
    
    // Check if DMA is healthy
    bool is_dma_healthy() {
        // DMA is healthy if no errors, regardless of running state
        // (DMA may pause between descriptor fills but still be healthy)
        return !get_dma_error() && streaming_active;
    }

    std::vector<float> read_adc_buffer_chunk(uint32_t offset, uint32_t size) 
    {
        if (!streaming_active) {
            return {};
        }

        const uint32_t total_buffer_samples = n_desc * n_pts;

        if (size > total_buffer_samples) {
            size = total_buffer_samples;
        }

        std::vector<float> result;
        result.reserve(size);

        for (uint32_t i = 0; i < size; ++i) {
            uint32_t buffer_idx = (offset + i) % total_buffer_samples;

            // Read 32-bit CIC+FIR output directly
            uint32_t sample_byte_offset = buffer_idx * 4;
            int32_t adc_raw = static_cast<int32_t>(ram_s2mm.read_reg(sample_byte_offset));

            // Apply corrected voltage scaling
            result.push_back(adc_to_voltage(static_cast<uint32_t>(adc_raw)));
        }

        return result;
    }

    // === NEW FAST CHUNK READER ===
    // Reads a block of samples much faster than read_adc_buffer_chunk by reading
    // 32-bit CIC+FIR samples directly. This reduces complexity and correctly handles
    // the 32-bit output format.
    // offset and size are expressed in 32-bit samples.
    std::vector<float> read_adc_buffer_block(uint32_t offset, uint32_t size) {
        if (!streaming_active || size == 0) {
            return {};
        }

        const uint32_t total_buffer_samples = n_desc * n_pts;
        if (offset >= total_buffer_samples) {
            // Wrap offset explicitly so caller can pass huge values safely
            offset = offset % total_buffer_samples;
        }
        if (size > total_buffer_samples) {
            size = total_buffer_samples;
        }

        std::vector<float> result(size);
        
        for (uint32_t i = 0; i < size; i++) {
            uint32_t buf_idx = (offset + i) % total_buffer_samples;
            // Read 32-bit CIC+FIR output directly
            uint32_t sample_byte_offset = buf_idx * 4;
            int32_t adc_raw = static_cast<int32_t>(ram_s2mm.read_reg(sample_byte_offset));
            
            // Apply corrected voltage scaling
            result[i] = adc_to_voltage(static_cast<uint32_t>(adc_raw));
        }
        
        return result;
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
        // Use calibrated voltage conversion matching working alpha15 examples
        // The CIC+FIR pipeline output needs proper scaling
        int32_t adc_signed = static_cast<int32_t>(adc_raw);
        
        // Get voltage range and apply scaling like working examples
        double vrange = adc_range_sel_ ? 8.192 : 2.048;  // 8Vpp or 2Vpp range
        constexpr double nmax = 262144.0; // 2^18 for 18-bit ADC
        constexpr double cic_fir_compensation = 4096.0; // 2^12 scaling factor
        
        // Apply the same scaling as working decimator example
        double voltage = vrange * static_cast<double>(adc_signed) / nmax / cic_fir_compensation;
        
        return static_cast<float>(voltage);
    }

    // === ADC RANGE CONTROL ===

    void set_adc_input_range(uint8_t range_sel) {
        adc_range_sel_ = (range_sel ? 1 : 0);
        ltc2387.range_select(0, adc_range_sel_);
        ltc2387.range_select(1, adc_range_sel_);
        adc_range_vpp_ = adc_range_sel_ ? 8.0f : 2.0f;
        ctx.log<INFO>("ADC input range set to %.1f Vpp", static_cast<double>(adc_range_vpp_));
    }
    
    uint32_t get_adc_input_range() {
        return static_cast<uint32_t>(adc_range_sel_);
    }
    
    float get_adc0_voltage() {
        return adc_to_voltage(get_adc0());
    }
    
    float get_adc1_voltage() {
        return adc_to_voltage(get_adc1());
    }

    // === DATA RETRIEVAL FUNCTIONS ===
    
    // This is the new, robust data retrieval function.
    // Reading most recent samples from the DMA circular buffer.
    std::vector<float> get_adc_stream_voltages(uint32_t num_samples) {
        if (!streaming_active) {
            ctx.log<WARNING>("DMA streaming is not active, returning empty vector.");
            return {};
        }

        //  can't request more data than is available in the buffer history.
        if (num_samples > (n_desc - 1) * n_pts) {
            num_samples = (n_desc - 1) * n_pts;
            ctx.log<WARNING>("Requested samples exceeds buffer history, limiting to %u", num_samples);
        }

        // 1. Find the current write position of the DMA. We don't stop the DMA,
        // which prevents sample loss and allows for continuous monitoring.
        uint32_t current_desc_idx = get_current_descriptor_index();
        
        // 2. Determine the most recent "safe" block of data to read.
        // We go back two full descriptors from the current one to ensure we are reading from a region that is not actively being written to.
        const uint32_t total_buffer_samples = n_desc * n_pts;
        uint32_t last_safe_desc_idx = (current_desc_idx + n_desc - 2) % n_desc;
        uint32_t end_sample_in_buffer = (last_safe_desc_idx + 1) * n_pts;

        // 3. Calculate the start position for the read, handling buffer wrap-around.
        uint32_t start_sample_in_buffer = (end_sample_in_buffer + total_buffer_samples - num_samples) % total_buffer_samples;

        std::vector<float> result(num_samples);
        ctx.log<INFO>("Reading %u fresh samples from DMA circular buffer...", num_samples);

        // 4. Read the data block, unpacking each 32-bit sample correctly.
        for (uint32_t i = 0; i < num_samples; i++) {
            uint32_t buffer_idx = (start_sample_in_buffer + i) % total_buffer_samples;
            
            // Read 32-bit CIC+FIR output directly 
            uint32_t sample_byte_offset = buffer_idx * 4;
            int32_t adc_raw = static_cast<int32_t>(ram_s2mm.read_reg(sample_byte_offset));
            
            // Scale voltage
            result[i] = adc_to_voltage(static_cast<uint32_t>(adc_raw));
        }

        ctx.log<INFO>("Data retrieval complete.");
        return result;
    }

  private:
    Context& ctx;
    PrecisionDac& precision_dac;
    ClockGenerator& clk_gen;
    Ltc2387& ltc2387;
    Memory<mem::control>& ctl;
    Memory<mem::status>& sts;
    
    // DMA memory interfaces
    Memory<mem::dma>& dma;
    Memory<mem::ram_s2mm>& ram_s2mm;
    Memory<mem::ocm_s2mm>& ocm_s2mm;
    Memory<mem::ocm_mm2s>& ocm_mm2s;
    Memory<mem::sclr>& sclr;
    
    // Clock and timing
    double fs_adc;  // ADC sampling frequency
    uint8_t adc_range_sel_ = 1; // 0=2Vpp,1=8Vpp 
    float   adc_range_vpp_ = 8.0f;
    
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
    
    void init_dma_system() {
        // Unlock SCLR (following adc-dac-dma example)
        sclr.write<Sclr_regs::sclr_unlock>(0xDF0D);
        sclr.clear_bit<Sclr_regs::fpga_rst_ctrl, 1>();

        // Map the last 64 kB of OCM RAM to the high address space
        sclr.write<Sclr_regs::ocm_cfg>(0b1000);
    }
    
    void set_descriptor_s2mm(uint32_t idx, uint32_t buffer_address, uint32_t buffer_length) {
        ocm_mm2s.write_reg(0x40 * idx + Sg_regs::nxtdesc, mem::ocm_mm2s_addr + 0x40 * ((idx+1) % n_desc));
        ocm_mm2s.write_reg(0x40 * idx + Sg_regs::buffer_address, buffer_address);
        ocm_mm2s.write_reg(0x40 * idx + Sg_regs::control, buffer_length);
        ocm_mm2s.write_reg(0x40 * idx + Sg_regs::status, 0);
    }
    
    void set_descriptors() {
        for (uint32_t i = 0; i < n_desc; i++) {
            // The buffer length for the DMA is also 4 * n_pts.
            set_descriptor_s2mm(i, mem::ram_s2mm_addr + i * 4 * n_pts, 4 * n_pts);
        }
    }
};

#endif // __DRIVERS_CURRENTRAMP_HPP__ 