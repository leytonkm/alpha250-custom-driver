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

# Phase increment control for ramp frequency
cell pavel-demin:user:axis_constant:1.0 ramp_phase_increment {
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

# TODO: Connect ramp output to precision DAC channel 2
# For now, just hardware ramp generation without output connection 