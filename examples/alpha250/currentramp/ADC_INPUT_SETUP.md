# ADC Input System - Setup Complete

## What I've Built

I've created a complete ADC input testing and monitoring system for your CurrentRamp project that will help you verify that your photodiode input is working correctly.

## Files Created

### Python Tests (`python/` directory)
1. **`test_adc_simple.py`** - Quick 30-second test to verify ADC functionality
2. **`test_adc_input.py`** - Comprehensive test with live plotting capabilities
3. **`README_ADC_TESTS.md`** - Complete documentation for running tests

### Web Interface (`web/` directory)
1. **`adc-monitor.ts`** - TypeScript component for live ADC monitoring
2. **`adc-monitor.html`** - Web interface for real-time graphing
3. **Updated `config.yml`** - Includes new web components
4. **Updated `index.html`** - Adds link to ADC monitor

## How This Works

### The Testing Strategy
1. **Loopback Testing**: Connect your DAC output to ADC input
2. **Known Signal**: Output a controlled ramp from DAC
3. **Correlation Check**: Verify the ADC reads the same signal
4. **BRAM Capture**: Use hardware BRAM to capture paired ADC/DAC data

### What You Get
- **Real-time correlation analysis** between output and input
- **Live graphing** of voltage vs time
- **Statistical analysis** of signal quality
- **Web-based monitoring** for continuous observation
- **Data export** capabilities

## Next Steps - How to Use

### 1. Hardware Setup
```bash
# Connect DAC output to ADC input for loopback testing
# This proves your ADC channel is working
```

### 2. Run Quick Test
```bash
cd examples/alpha250/currentramp/python
export HOST=192.168.1.20  # Your device IP
python3 test_adc_simple.py
```

### 3. Expected Results
```
✅ ADC readings working
✅ BRAM capture shows data variation
✅ Strong correlation - loopback working well!
🎉 ADC input appears to be working correctly!
```

### 4. Use Web Interface
- Navigate to: `http://your-device-ip/adc-monitor.html`
- Start live monitoring to see real-time plots
- Export data for analysis

## What the Tests Verify

### ✅ ADC Functionality
- Both ADC channels are reading data
- Values are changing (not stuck)
- Voltage ranges are sensible

### ✅ BRAM System
- Hardware data acquisition working
- 1000 samples captured successfully
- Timing synchronization correct

### ✅ Loopback Verification
- Output signal matches input signal
- High correlation (>0.7) indicates good connection
- Proves your ADC input path is working

### ✅ Ready for Photodiode
- Once loopback test passes, disconnect the loop
- Connect your photodiode to the ADC input
- Use the same monitoring tools to observe photodiode signal

## Integration with Your CurrentRamp

This builds on your existing CurrentRamp system:
- **Uses existing BRAM infrastructure** from your hardware design
- **Leverages existing DAC control** for test signal generation
- **Integrates with existing web interface** framework
- **Compatible with existing driver functions**

## Troubleshooting Guide

The tests provide detailed diagnostics:
- **Connection issues**: Clear error messages with solutions
- **Hardware problems**: Identifies BRAM, ADC, or DAC issues
- **Signal quality**: Correlation analysis shows connection quality
- **Performance metrics**: Timing and data rate analysis

## File Structure Summary
```
examples/alpha250/currentramp/
├── python/
│   ├── test_adc_simple.py          # Quick ADC test
│   ├── test_adc_input.py           # Comprehensive test
│   ├── README_ADC_TESTS.md         # Test documentation
│   └── ADC_INPUT_SETUP.md          # This summary
├── web/
│   ├── adc-monitor.ts              # Live monitoring component
│   ├── adc-monitor.html            # Web interface
│   └── index.html                  # Updated with ADC link
└── config.yml                      # Updated to include ADC components
```

## Ready to Test!

Your ADC input system is now ready for testing. Start with the simple test to verify basic functionality, then move to the comprehensive test for detailed analysis and live plotting.

Once the tests pass, you'll know your ADC input is working correctly and you can proceed with confidence to connect your photodiode and implement your current ramp measurement system.

Good luck with your testing! 🔬 