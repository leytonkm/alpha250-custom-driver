# ADC Input Testing for CurrentRamp

This directory contains tests to verify that the ADC input functionality is working correctly with your CurrentRamp hardware.

## Overview

The ADC input tests are designed to verify that:
1. ADC channels are reading data correctly
2. BRAM data capture is working
3. Loopback connection between DAC output and ADC input is functioning
4. Data correlation between output and input signals is good

## Test Files

### `test_adc_simple.py`
- **Purpose**: Quick verification of ADC functionality
- **Tests**: Basic ADC readings, BRAM capture, loopback correlation
- **Output**: Console results + matplotlib plots
- **Runtime**: ~30 seconds

### `test_adc_input.py`
- **Purpose**: Comprehensive ADC testing with live plotting
- **Tests**: All functionality + live graphing demonstration
- **Output**: Console results + live animated plots
- **Runtime**: ~1-2 minutes (includes 10s live demo)

## Hardware Setup

**CRITICAL**: Connect your DAC output to your ADC input for loopback testing!

1. **Physical Connection**: Connect DAC output channel to ADC input channel
   - This allows the test to verify that what you output matches what you read
   - Use appropriate cables/connectors for your Alpha250 setup

2. **Verification**: The tests will output a known ramp signal and verify it appears correctly on the input

## Running the Tests

### Prerequisites
```bash
# Make sure you have the required Python packages
pip install numpy matplotlib koheron
```

### Environment Setup
```bash
# Set your device IP address (if different from default)
export HOST=192.168.1.20  # Replace with your Alpha250 IP
```

### Quick Test
```bash
cd examples/alpha250/currentramp/python
python test_adc_simple.py
```

### Comprehensive Test
```bash
cd examples/alpha250/currentramp/python
python test_adc_input.py
```

## Expected Results

### ✅ **Success Indicators**
- **ADC Readings**: Show variation (not stuck at one value)
- **BRAM Capture**: Returns 1000 samples of data
- **Loopback Correlation**: > 0.7 (strong correlation)
- **Voltage Ranges**: Sensible values within expected ranges

### ⚠️ **Warning Signs**
- **Moderate Correlation**: 0.3-0.7 (check connections)
- **No ADC Variation**: Channels show constant values
- **Timeout Errors**: BRAM acquisition takes too long

### ❌ **Failure Indicators**
- **Connection Failed**: Can't connect to device
- **Poor Correlation**: < 0.3 (loopback not working)
- **No BRAM Data**: Acquisition completely fails

## Troubleshooting

### Connection Issues
```
❌ Connection failed: [Errno 111] Connection refused
```
**Solution**: Check device IP address and network connection

### Poor Loopback Correlation
```
❌ Poor correlation - check loopback connection
Correlation: 0.1
```
**Solution**: 
1. Verify physical cable connection
2. Check that DAC output is enabled
3. Ensure correct ADC channel selection

### No ADC Variation
```
⚠️ Channel 0: ADC shows no variation (check connection)
Range: 0.0001V
```
**Solution**:
1. Check ADC input connection
2. Verify signal source
3. Check ADC channel configuration

### BRAM Timeout
```
⚠️ BRAM acquisition timeout
```
**Solution**:
1. Check FPGA configuration
2. Restart device
3. Check for hardware conflicts

## Understanding the Data

### ADC Data Format
- **Raw Data**: 32-bit values containing two 16-bit ADC channels
- **Voltage Conversion**: Raw values converted to ±1.8V range
- **Sampling Rate**: 250 MHz (4 ns per sample)

### DAC Data Format
- **Raw Data**: 32-bit values containing two 16-bit DAC channels  
- **Voltage Conversion**: Raw values converted to 0-2.5V range
- **Update Rate**: 250 MHz (synchronized with ADC)

### Correlation Analysis
- **Perfect Correlation**: 1.0 (output exactly matches input)
- **Good Correlation**: > 0.8 (loopback working well)
- **Acceptable**: > 0.5 (some signal coupling)
- **Poor**: < 0.3 (connection problems)

## Next Steps

### If Tests Pass ✅
1. **Web Interface**: Access the live ADC monitor at: `http://your-device-ip/adc-monitor.html`
2. **Integration**: Your photodiode input system is ready for use
3. **Calibration**: Consider voltage/current calibration for your specific setup

### If Tests Fail ❌
1. **Hardware Check**: Verify all connections
2. **FPGA Check**: Ensure correct bitstream is loaded
3. **Software Check**: Verify driver compilation
4. **Support**: Contact Koheron support with test output

## Web Interface

After successful testing, you can use the live web interface:

1. **Navigate to**: `http://your-device-ip/adc-monitor.html`
2. **Features**:
   - Live plotting of ADC/DAC data
   - Real-time correlation monitoring
   - Data export capabilities
   - Multiple update rates
   - Snapshot capture

## Technical Details

### BRAM Configuration
- **ADC Buffer**: 256K samples (1000 samples accessible via driver)
- **DAC Buffer**: 256K samples (1000 samples accessible via driver)
- **Synchronization**: Hardware-triggered simultaneous capture

### Data Processing
- **Time Base**: Calculated from 250 MHz sampling rate
- **Bit Packing**: Two 16-bit channels per 32-bit word
- **Voltage Scaling**: Hardware-specific conversion factors

### Performance
- **Acquisition Time**: ~4 μs for 1000 samples at 250 MHz
- **Processing Time**: ~10 ms for conversion and analysis
- **Update Rate**: Up to 20 Hz for live monitoring

## File Structure
```
python/
├── test_adc_simple.py      # Quick ADC verification
├── test_adc_input.py       # Comprehensive testing
├── test_currentramp.py     # Original functionality tests
└── README_ADC_TESTS.md     # This documentation
``` 