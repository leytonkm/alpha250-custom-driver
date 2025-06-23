source ${board_path}/starting_point.tcl

####################################
# Hardware Ramp Generation
####################################

# Phase accumulator for ramp timing - uses DDS architecture for timing only
cell xilinx.com:ip:dds_compiler:6.0 ramp_timer {
  PartsPresent Phase_Generator_Only
  DDS_Clock_Rate [expr [get_parameter adc_clk] / 1000000]
  Parameter_Entry Hardware_Parameters
  Phase_Width [get_parameter ramp_phase_width]
  Phase_Increment Programmable
  Latency_Configuration Configurable
  Latency 3
  Has_Phase_Out true
} {
  aclk adc_dac/adc_clk
}

# Phase increment control for ramp frequency - use axis_variable for proper AXI interface
cell pavel-demin:user:axis_variable:1.0 ramp_phase_increment {
  AXIS_TDATA_WIDTH [get_parameter ramp_phase_width]
} {
  cfg_data [ctl_pin ramp_freq_incr]
  aclk adc_dac/adc_clk
  M_AXIS ramp_timer/S_AXIS_CONFIG
}

# Ramp waveform generator - converts phase to sawtooth
# Take MSBs of phase accumulator as sawtooth (linear ramp)
cell xilinx.com:ip:xlslice:1.0 ramp_sawtooth {
  DIN_WIDTH [get_parameter ramp_phase_width]
  DIN_FROM [expr [get_parameter ramp_phase_width] - 1]
  DIN_TO [expr [get_parameter ramp_phase_width] - 16]
  DOUT_WIDTH 16
} {
  Din ramp_timer/m_axis_phase_tdata
}

# Amplitude scaling
cell xilinx.com:ip:mult_gen:12.0 ramp_amplitude_mult {
  PortAWidth 16
  PortBWidth 16
  Multiplier_Construction Use_Mults
  OptGoal Speed
  Use_Custom_Output_Width true
  OutputWidthHigh 31
  OutputWidthLow 16
  PortAType Unsigned
  PortBType Unsigned
} {
  CLK adc_dac/adc_clk
  A ramp_sawtooth/Dout
  B [ctl_pin ramp_amplitude_reg]
}

# Offset addition
cell xilinx.com:ip:c_addsub:12.0 ramp_offset_add {
  A_Width 16
  B_Width 16
  Out_Width 16
  A_Type Unsigned
  B_Type Signed
  Latency 1
  CE false
} {
  CLK adc_dac/adc_clk
  A ramp_amplitude_mult/P
  B [ctl_pin ramp_offset_reg]
}

# Ramp enable control - mux between ramp output and zero
cell koheron:user:latched_mux:1.0 ramp_output_mux {
  WIDTH 16
  N_INPUTS 2
  SEL_WIDTH 1
} {
  clk adc_dac/adc_clk
  clken [get_constant_pin 1 1]
  sel [ctl_pin ramp_enable]
  din [get_concat_pin [list [get_constant_pin 0 16] ramp_offset_add/S]]
}

# Format precision DAC data register for channels 2 and 3
# precision_dac_data1 = (channel3_data << 16) + channel2_data
# Channel 2 = ramp output, Channel 3 = zero (unused)
cell xilinx.com:ip:xlconcat:2.1 precision_dac_data1_concat {
  NUM_PORTS 2
  IN0_WIDTH 16
  IN1_WIDTH 16
} {
  In0 ramp_output_mux/dout
  In1 [get_constant_pin 0 16]
}

# Connect ramp data to precision DAC data1 register (channels 2 and 3)
# Delete existing net first, then connect our hardware ramp
delete_bd_objs [get_bd_nets ctl_precision_dac_data1]
connect_pins precision_dac_data1_concat/dout concat_precision_dac_data/In1

# Status outputs - connect phase accumulator to status register
connect_pins [sts_pin ramp_phase] ramp_timer/m_axis_phase_tdata

# Extract MSB of phase accumulator for cycle counter
cell xilinx.com:ip:xlslice:1.0 ramp_phase_msb {
  DIN_WIDTH 32
  DIN_FROM 31
  DIN_TO 31
} {
  Din ramp_timer/m_axis_phase_tdata
}

# Cycle counter - count phase accumulator overflows (detect MSB transitions)
cell xilinx.com:ip:c_counter_binary:12.0 cycle_counter {
  Output_Width 32
  CE true
  SCLR true
} {
  CLK adc_dac/adc_clk
  CE ramp_phase_msb/Dout
  SCLR [ctl_pin ramp_reset]
}

connect_pins [sts_pin cycle_count] cycle_counter/Q

####################################
# DMA Infrastructure (following adc-dac-dma example exactly)
####################################

# Configure Zynq Processing System for DMA
set_cell_props ps_0 {
  PCW_USE_S_AXI_HP0 1
  PCW_S_AXI_HP0_DATA_WIDTH 64
  PCW_USE_S_AXI_HP2 1
  PCW_S_AXI_HP2_DATA_WIDTH 64
  PCW_USE_HIGH_OCM 1
  PCW_USE_S_AXI_GP0 1
}

connect_pins ps_0/S_AXI_GP0_ACLK ps_0/FCLK_CLK0
connect_pins ps_0/S_AXI_HP0_ACLK ps_0/FCLK_CLK0
connect_pins ps_0/S_AXI_HP2_ACLK ps_0/FCLK_CLK0

# Create DMA interconnect first, then connect
cell xilinx.com:ip:axi_interconnect:2.1 dma_interconnect {
  NUM_SI 2
  NUM_MI 3
  S01_HAS_REGSLICE 1
} {
  ACLK ps_0/FCLK_CLK0
  ARESETN proc_sys_reset_0/peripheral_aresetn
  S00_ACLK ps_0/FCLK_CLK0
  S00_ARESETN proc_sys_reset_0/peripheral_aresetn
  S01_ACLK ps_0/FCLK_CLK0
  S01_ARESETN proc_sys_reset_0/peripheral_aresetn
  M00_ACLK ps_0/FCLK_CLK0
  M00_ARESETN proc_sys_reset_0/peripheral_aresetn
  M01_ACLK ps_0/FCLK_CLK0
  M01_ARESETN proc_sys_reset_0/peripheral_aresetn
  M02_ACLK ps_0/FCLK_CLK0
  M02_ARESETN proc_sys_reset_0/peripheral_aresetn
}

# Connect DMA interconnect to PS interfaces
connect_bd_intf_net [get_bd_intf_pins dma_interconnect/M00_AXI] [get_bd_intf_pins ps_0/S_AXI_GP0]
connect_bd_intf_net [get_bd_intf_pins dma_interconnect/M01_AXI] [get_bd_intf_pins ps_0/S_AXI_HP0]
connect_bd_intf_net [get_bd_intf_pins dma_interconnect/M02_AXI] [get_bd_intf_pins ps_0/S_AXI_HP2]

####################################
# ADC Streaming Pipeline with CIC Decimation
####################################

# ADC channel multiplexer
cell koheron:user:bus_multiplexer:1.0 adc_mux {
  WIDTH 16
} {
  din0 adc_dac/adc0
  din1 adc_dac/adc1
  sel [get_slice_pin [ctl_pin channel_select] 0 0]
}

# ADC Streaming Pipeline with CIC Decimation (following phase-noise-analyzer exactly)

# Define CIC parameters from config.yml
set diff_delay [get_parameter cic_differential_delay]
set dec_rate_default [get_parameter cic_decimation_rate_default]
set dec_rate_min [get_parameter cic_decimation_rate_min]
set dec_rate_max [get_parameter cic_decimation_rate_max]
set n_stages [get_parameter cic_n_stages]

# CIC Decimator - The Missing Core!
cell xilinx.com:ip:cic_compiler:4.0 cic_decimator {
  Filter_Type Decimation
  Number_Of_Stages $n_stages
  Fixed_Or_Initial_Rate $dec_rate_default
  Sample_Rate_Changes Programmable
  Minimum_Rate $dec_rate_min
  Maximum_Rate $dec_rate_max
  Differential_Delay $diff_delay
  Input_Sample_Frequency [expr [get_parameter adc_clk] / 1000000.]
  Clock_Frequency [expr [get_parameter adc_clk] / 1000000.]
  Input_Data_Width 16
  Quantization Truncation
  Output_Data_Width 32
  Use_Xtreme_DSP_Slice false
  HAS_DOUT_TREADY true
} {
  aclk adc_dac/adc_clk
  s_axis_data_tdata adc_mux/dout
  s_axis_data_tvalid [get_constant_pin 1 1]
  s_axis_data_aresetn rst_adc_clk/peripheral_aresetn
}

# CIC Rate Control (connect the control register to the CIC core)
cell pavel-demin:user:axis_variable:1.0 cic_rate_control {
  AXIS_TDATA_WIDTH 16
} {
  cfg_data [ctl_pin cic_rate]
  aclk adc_dac/adc_clk
  aresetn rst_adc_clk/peripheral_aresetn
  M_AXIS cic_decimator/S_AXIS_CONFIG
}

# Width converter: 32-bit CIC output to 64-bit for DMA efficiency
cell xilinx.com:ip:axis_dwidth_converter:1.1 axis_dwidth_converter_0 {
  S_TDATA_NUM_BYTES 4
  M_TDATA_NUM_BYTES 8
} {
  aclk adc_dac/adc_clk
  aresetn rst_adc_clk/peripheral_aresetn
  S_AXIS cic_decimator/M_AXIS_DATA
}

# Clock domain crossing: ADC clock to fabric clock
cell xilinx.com:ip:axis_clock_converter:1.1 axis_clock_converter_0 {
  TDATA_NUM_BYTES 8
} {
  s_axis_aclk adc_dac/adc_clk
  s_axis_aresetn rst_adc_clk/peripheral_aresetn
  m_axis_aclk ps_0/FCLK_CLK0
  m_axis_aresetn proc_sys_reset_0/peripheral_aresetn
  S_AXIS axis_dwidth_converter_0/M_AXIS
}

# Packet generator for DMA transfers
cell koheron:user:tlast_gen:1.0 tlast_gen_0 {
  TDATA_WIDTH 64
  PKT_LENGTH [expr 1024 * 1024]
} {
  aclk ps_0/FCLK_CLK0
  resetn proc_sys_reset_0/peripheral_aresetn
  s_axis axis_clock_converter_0/M_AXIS
}

####################################
# AXI DMA (S2MM only - no MM2S needed)
####################################

cell xilinx.com:ip:axi_dma:7.1 axi_dma_0 {
  c_include_sg 1
  c_sg_include_stscntrl_strm 0
  c_sg_length_width 20
  c_s2mm_burst_size 16
  c_m_axi_s2mm_data_width 64
  c_include_mm2s 0
} {
  S_AXI_LITE axi_mem_intercon_0/M[add_master_interface]_AXI
  s_axi_lite_aclk ps_0/FCLK_CLK0
  M_AXI_SG dma_interconnect/S00_AXI
  m_axi_sg_aclk ps_0/FCLK_CLK0
  M_AXI_S2MM dma_interconnect/S01_AXI
  m_axi_s2mm_aclk ps_0/FCLK_CLK0
  S_AXIS_S2MM tlast_gen_0/m_axis
  axi_resetn proc_sys_reset_0/peripheral_aresetn
  s2mm_introut [get_interrupt_pin]
}

# DMA AXI Lite address assignment
assign_bd_address [get_bd_addr_segs {axi_dma_0/S_AXI_LITE/Reg }]
set_property range [get_memory_range dma] [get_bd_addr_segs {ps_0/Data/SEG_axi_dma_0_Reg}]
set_property offset [get_memory_offset dma] [get_bd_addr_segs {ps_0/Data/SEG_axi_dma_0_Reg}]

# Scatter Gather interface in On Chip Memory (use GP0 HIGH_OCM like adc-dac-dma)
assign_bd_address [get_bd_addr_segs {ps_0/S_AXI_GP0/GP0_HIGH_OCM }]
set_property range 64K [get_bd_addr_segs {axi_dma_0/Data_SG/SEG_ps_0_GP0_HIGH_OCM}]
set_property offset [get_memory_offset ocm_mm2s] [get_bd_addr_segs {axi_dma_0/Data_SG/SEG_ps_0_GP0_HIGH_OCM}]

# S2MM interface in DDR (use HP2 like adc-dac-dma)
assign_bd_address [get_bd_addr_segs {ps_0/S_AXI_HP2/HP2_DDR_LOWOCM }]
set_property range [get_memory_range ram_s2mm] [get_bd_addr_segs {axi_dma_0/Data_S2MM/SEG_ps_0_HP2_DDR_LOWOCM}]
set_property offset [get_memory_offset ram_s2mm] [get_bd_addr_segs {axi_dma_0/Data_S2MM/SEG_ps_0_HP2_DDR_LOWOCM}]

# Unmap unused segments to avoid conflicts
delete_bd_objs [get_bd_addr_segs axi_dma_0/Data_SG/SEG_ps_0_HP0_DDR_LOWOCM]
delete_bd_objs [get_bd_addr_segs axi_dma_0/Data_SG/SEG_ps_0_HP2_DDR_LOWOCM]
delete_bd_objs [get_bd_addr_segs axi_dma_0/Data_S2MM/SEG_ps_0_GP0_HIGH_OCM]
delete_bd_objs [get_bd_addr_segs axi_dma_0/Data_S2MM/SEG_ps_0_HP0_DDR_LOWOCM]

# Validate design 