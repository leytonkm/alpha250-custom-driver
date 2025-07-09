# Expansion connector constraints for test signals
# Alpha250 expansion connector pin assignments for debugging

# Test signal outputs for oscilloscope verification
set_property -dict {PACKAGE_PIN Y18 IOSTANDARD LVCMOS33} [get_ports exp_io_0_p] ; # Ramp overflow
set_property -dict {PACKAGE_PIN Y19 IOSTANDARD LVCMOS33} [get_ports exp_io_1_p] ; # Sawtooth MSB  
set_property -dict {PACKAGE_PIN W18 IOSTANDARD LVCMOS33} [get_ports exp_io_2_p] ; # Ramp enable
set_property -dict {PACKAGE_PIN W19 IOSTANDARD LVCMOS33} [get_ports exp_io_3_p] ; # Final output MSB

# Additional test points for debugging
set_property -dict {PACKAGE_PIN V18 IOSTANDARD LVCMOS33} [get_ports exp_io_4_p] ; # Reserved
set_property -dict {PACKAGE_PIN V19 IOSTANDARD LVCMOS33} [get_ports exp_io_5_p] ; # Reserved
set_property -dict {PACKAGE_PIN U18 IOSTANDARD LVCMOS33} [get_ports exp_io_6_p] ; # Reserved
set_property -dict {PACKAGE_PIN U19 IOSTANDARD LVCMOS33} [get_ports exp_io_7_p] ; # Reserved 