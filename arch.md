# Koheron SDK Architecture Documentation

## Overview
The Koheron SDK is a comprehensive development framework for building FPGA-based instruments on Zynq platforms. This document provides a complete architectural overview of the codebase structure.

## Root Directory Structure

```
koheron-sdk/
├── .git/                   # Git repository metadata
├── .gitignore             # Git ignore rules
├── .circleci/             # CI/CD configuration
├── LICENSE                # Apache 2.0 license
├── README.md              # Basic project readme
├── Makefile               # Main build system entry point
├── make.py                # Python configuration management script
├── version                # SDK version file
├── config.yml             # Default configuration
├── requirements.txt       # Python dependencies
├── package.json           # Node.js dependencies
├── package-lock.json      # Locked Node.js dependencies
├── build_examples.sh      # Script to build all examples
├── install_eigen.sh       # Eigen installation script
├── node_modules/          # Node.js modules (auto-generated)
├── tmp/                   # Temporary build artifacts
├── .Xil/                  # Xilinx temporary files
├── boards/                # Board-specific configurations
├── examples/              # Example projects and instruments
├── fpga/                  # FPGA development components
├── server/                # C++ server implementation
├── python/                # Python client library
├── web/                   # Web interface components
├── os/                    # Operating system components
├── docker/                # Docker build environment
└── tests/                 # Test suite
```

## Build System Architecture

### Core Files
- **`Makefile`**: Main entry point, orchestrates all build targets
- **`make.py`**: Python script that handles configuration parsing, template rendering, and build coordination
- **`config.yml`**: Global SDK configuration

### Key Build Targets
```bash
make all          # Build complete instrument (FPGA + server + web)
make fpga         # Build FPGA bitstream only
make server       # Build C++ server only  
make web          # Build web interface only
make run          # Deploy to target device
make setup        # Install dependencies
```

## FPGA Components (`fpga/`)

### Directory Structure
```
fpga/
├── config.tcl             # Global FPGA configuration template
├── fpga.mk               # FPGA-specific Makefile rules
├── install_vivado.sh     # Vivado installation script
├── cores/                # Custom IP cores
├── lib/                  # TCL library scripts
├── modules/              # Reusable FPGA modules
├── vivado/               # Vivado project templates
└── hsi/                  # Hardware Software Interface files
```

### Custom IP Cores (`fpga/cores/`)

The SDK includes 29+ custom IP cores for various functions:

#### Signal Processing Cores
- **`axis_constant_v1_0/`** - Constant value generator for AXI Stream
- **`axis_variable_v1_0/`** - Variable value generator for AXI Stream  
- **`boxcar_filter_v1_0/`** - Moving average filter implementation
- **`comparator_v1_0/`** - Digital comparator with configurable thresholds
- **`edge_detector_v1_0/`** - Rising/falling edge detection
- **`phase_unwrapper_v1_0/`** - Phase unwrapping for continuous signals
- **`saturation_v1_0/`** - Signal saturation/clipping
- **`double_saturation_v1_0/`** - Dual-threshold saturation

#### Control & Interface Cores  
- **`axi_ctl_register_v1_0/`** - AXI control register interface
- **`axi_sts_register_v1_0/`** - AXI status register interface
- **`bus_multiplexer_v1_0/`** - Multi-input bus multiplexer
- **`latched_mux_v1_0/`** - Latched multiplexer with clock enable

#### Data Acquisition Cores
- **`axis_red_pitaya_adc_v1_0/`** - Red Pitaya ADC interface
- **`axis_red_pitaya_dac_v1_0/`** - Red Pitaya DAC interface
- **`redp_adc_v1_0/`** - Red Pitaya ADC core
- **`redp_dac_v1_0/`** - Red Pitaya DAC core

#### Utility Cores
- **`address_generator_v1_0/`** - Address generation for memory access
- **`averager_counter_v1_0/`** - Averaging counter implementation
- **`delay_trig_v1_0/`** - Programmable delay trigger
- **`dna_reader_v1_0/`** - Device DNA reader for unique ID
- **`pulse_generator_v1_0/`** - Configurable pulse generation
- **`pwm_v1_0/`** - Pulse width modulation generator
- **`right_shifter_v1_0/`** - Bit shift operations
- **`tlast_gen_v1_0/`** - AXI Stream packet termination
- **`unrandomizer_v1_0/`** - Deterministic sequence generation

#### Advanced Cores
- **`kcpsm6_v1_0/`** - PicoBlaze 6 microcontroller integration
- **`pdm_v1_0/`** - Pulse density modulation
- **`psd_counter_v1_0/`** - Power spectral density counter
- **`vhdl_counter_v1_0/`** - VHDL-based counter implementation
- **`at93c46d_spi_v1_0/`** - SPI EEPROM interface
- **`axis_lfsr_v1_0/`** - Linear feedback shift register

#### Core Structure Pattern
Each core follows a standard structure:
```
<core_name>_v1_0/
├── <core_name>.v          # Verilog implementation
├── core_config.tcl        # Vivado IP configuration
└── <core_name>_tb.v       # Testbench (optional)
```

### TCL Library Scripts (`fpga/lib/`)

#### Core Library Files
- **`utilities.tcl`** (345 lines) - Fundamental utilities for block design
  - Pin connection functions
  - Constant generators  
  - Cell creation helpers
  - Parameter management

- **`ctl_sts.tcl`** (163 lines) - Control/status register management
  - AXI register creation
  - Memory mapping
  - Register interconnection

- **`starting_point.tcl`** (52 lines) - Base system initialization
  - Zynq PS configuration
  - Clock setup
  - Reset generation

#### Specialized Libraries
- **`bram.tcl`** (36 lines) - Block RAM management utilities
- **`bram_recorder.tcl`** (30 lines) - BRAM-based data recording
- **`interconnect.tcl`** (54 lines) - AXI interconnect helpers
- **`xadc.tcl`** (48 lines) - Zynq XADC integration
- **`redp_adc_dac.tcl`** (63 lines) - Red Pitaya ADC/DAC setup
- **`dac_controller.tcl`** (81 lines) - DAC control logic
- **`laser_controller.tcl`** (144 lines) - Laser control systems

### FPGA Modules (`fpga/modules/`)

Reusable higher-level modules:
- **`address/`** - Address generation module
- **`averager/`** - Signal averaging module  
- **`bram_accumulator/`** - BRAM-based accumulation
- **`peak_detector/`** - Signal peak detection
- **`spectrum/`** - Spectrum analysis module

Each module contains:
```
<module>/
├── <module>.tcl           # Module TCL implementation
├── block_design.tcl       # Block design template
├── config.yml            # Module configuration
└── test_bench.v          # Verification testbench
```

### Vivado Integration (`fpga/vivado/`)

Vivado project management:
- **`core.tcl`** - IP core integration script
- **`block_design.tcl`** - Main block design template
- **`test_module.tcl`** - Module testing framework

## Board Support (`boards/`)

### Supported Boards
- **`alpha250/`** - Koheron Alpha250 (main target)
- **`alpha250-1g/`** - Alpha250 1G variant
- **`alpha250-4/`** - Alpha250-4 variant  
- **`alpha250-4-1g/`** - Alpha250-4 1G variant
- **`alpha15/`** - Koheron Alpha15
- **`red-pitaya/`** - Red Pitaya STEMlab
- **`zedboard/`** - Xilinx ZedBoard
- **`zc706/`** - Xilinx ZC706
- **`microzed/`** - Avnet MicroZed
- **`te0745/`** - Trenz TE0745
- **`mydc7z015/`** - Custom DC7Z015 board

### Board Structure Pattern
```
boards/<board>/
├── PART                   # Xilinx part number
├── board.mk              # Board-specific make rules
├── starting_point.tcl    # Board initialization
├── adc_dac.tcl          # ADC/DAC configuration (if applicable)
├── spi.tcl              # SPI configuration (if applicable)
├── cores/               # Board-specific IP cores
├── config/              # Board configuration files
├── drivers/             # Board-specific drivers
└── patches/             # Board-specific patches
```

### Alpha250 Board Details
```
boards/alpha250/
├── PART                   # xc7z020clg400-2
├── board.mk              # Alpha250 build rules
├── starting_point.tcl    # Zynq PS + clock configuration
├── adc_dac.tcl          # LTC2195/DAC3283 interfaces
├── spi.tcl              # SPI peripherals (precision DAC, etc.)
├── cores/
│   ├── ad7124_v1_0/     # AD7124 ADC interface
│   ├── precision_dac_v1_0/ # Precision DAC control
│   └── spi_cfg_v1_0/    # SPI configuration core
├── config/
│   ├── board_preset.tcl # Vivado board preset
│   └── ports.xdc       # Pin constraints
├── drivers/             # Alpha250-specific drivers
└── patches/             # Board-specific patches
```

## Example Projects (`examples/`)

### Project Organization
Examples are organized by board type, with each board having multiple instrument examples.

### Alpha250 Examples (`examples/alpha250/`)
- **`currentramp/`** - Current ramping system (our focus project)
- **`fft/`** - FFT spectrum analyzer with DDS
- **`phase-noise-analyzer/`** - Phase noise measurement
- **`adc-dac-bram/`** - BRAM-based ADC/DAC
- **`adc-dac-dma/`** - DMA-based ADC/DAC  
- **`loopback/`** - Minimal loopback test
- **`vco/`** - Voltage controlled oscillator
- **`vouttest/`** - Voltage output testing
- **`dpll/`** - Digital phase-locked loop

### Red Pitaya Examples (`examples/red-pitaya/`)
- **`fft/`** - FFT analyzer
- **`oscillo/`** - Oscilloscope
- **`spectrum/`** - Spectrum analyzer
- **`laser-controller/`** - Laser control system
- **`phase-noise-analyzer/`** - Phase noise analysis
- **`pulse-generator/`** - Pulse generation
- **`decimator/`** - Signal decimation
- **`adc-dac/`** - Basic ADC/DAC
- **`adc-dac-bram/`** - BRAM ADC/DAC
- **`led-blinker/`** - LED control
- **`dual-dds/`** - Dual DDS system
- **`cluster/`** - Multi-board clustering

### Example Project Structure
```
examples/<board>/<project>/
├── config.yml           # Project configuration
├── block_design.tcl     # FPGA block design
├── <project>.hpp        # C++ driver header
├── <project>.py         # Python driver (optional)
├── test.py             # Test script
├── overlay.patch       # Kernel overlay patches (optional)
├── expansion_connector.xdc # Pin constraints (optional)
├── web/                # Web interface files
├── python/             # Python-specific files
└── tcl/                # Project-specific TCL modules
```

## Server Implementation (`server/`)

### Directory Structure
```
server/
├── __init__.py          # Python initialization
├── server.mk           # Server build rules
├── fft-windows.hpp     # FFT windowing functions
├── core/               # Core server implementation
├── drivers/            # Hardware drivers
├── context/            # Execution context management
├── templates/          # Code generation templates
├── client/             # Client library components
└── external_libs/      # Third-party libraries
```

### Key Components
- **C++ Server**: Multi-threaded TCP/WebSocket server
- **Driver Framework**: Hardware abstraction layer
- **Memory Management**: Safe FPGA register access
- **Template System**: Jinja2-based code generation

## Python Client (`python/`)

### Structure
```
python/
├── setup.py            # Package setup
├── python.mk          # Python build rules
├── requirements.txt    # Python dependencies
├── koheron/           # Main package
├── build/             # Build artifacts
└── koheron.egg-info/  # Package metadata
```

### Features
- **TCP Client**: Direct TCP communication with server
- **High-level API**: Instrument-specific interfaces
- **Type Safety**: Strong typing for FPGA registers
- **Async Support**: Asynchronous operation support

## Web Interface (`web/`)

### Core Files
```
web/
├── web.mk             # Web build rules
├── downloads.mk       # Web asset management
├── koheron.ts         # Core TypeScript client
├── koheron.d.ts       # TypeScript definitions
├── main.css           # Global styles
├── index.html         # Default landing page
├── navigation.html    # Navigation component
├── navigation.ts      # Navigation logic
└── <various>.ts/html  # Instrument-specific interfaces
```

### Features
- **TypeScript Client**: Type-safe WebSocket communication
- **Responsive UI**: Bootstrap-based responsive design
- **Real-time Plotting**: Flot.js integration for live data
- **Modular Design**: Component-based architecture

## Operating System (`os/`)

### Components
- **Kernel Building**: Custom Linux kernel compilation
- **Root Filesystem**: Buildroot-based system creation
- **Device Drivers**: FPGA and peripheral drivers
- **Init System**: Custom initialization scripts
- **API Server**: REST API for instrument management

## Docker Environment (`docker/`)

### Build Environment
- **Base Images**: Ubuntu-based development environment
- **Cross-compilation**: ARM toolchain setup
- **Vivado Integration**: Xilinx tools containerization
- **Dependency Management**: Reproducible build environment

## Configuration System

### Configuration Hierarchy
1. **Global Config** (`config.yml`) - SDK-wide defaults
2. **Board Config** (`boards/<board>/config/`) - Board-specific settings
3. **Project Config** (`examples/<board>/<project>/config.yml`) - Project overrides

### Key Configuration Elements
```yaml
name: project_name
board: boards/alpha250
version: "0.1"

parameters:
  adc_clk: 250000000
  n_samples: 8192

cores:
  - fpga/cores/axi_ctl_register_v1_0
  - fpga/cores/axi_sts_register_v1_0

drivers:
  - ./project_driver.hpp

control_registers:
  - ramp_enable
  - samples_per_cycle
  
status_registers:
  - ramp_phase
  - sample_trigger_count

memory:
  - name: ram
    offset: 0x10000000
    range: 512M
```

## TCL Script Ecosystem

### Script Categories

#### 1. Infrastructure Scripts
- **`fpga/lib/utilities.tcl`** - Core utilities (pin connections, constants)
- **`fpga/lib/starting_point.tcl`** - System initialization
- **`fpga/lib/ctl_sts.tcl`** - Register management
- **`fpga/lib/interconnect.tcl`** - AXI interconnect helpers

#### 2. Hardware Interface Scripts  
- **`boards/*/starting_point.tcl`** - Board-specific initialization
- **`boards/*/adc_dac.tcl`** - ADC/DAC configuration
- **`fpga/lib/redp_adc_dac.tcl`** - Red Pitaya interfaces
- **`fpga/lib/xadc.tcl`** - Zynq XADC integration

#### 3. Module Scripts
- **`fpga/modules/*/[module].tcl`** - Reusable module implementations
- **`fpga/lib/bram.tcl`** - Memory management
- **`fpga/lib/laser_controller.tcl`** - Laser control systems

#### 4. Project Scripts
- **`examples/*/*/block_design.tcl`** - Project-specific designs
- **`examples/*/tcl/*.tcl`** - Project-specific modules

#### 5. Vivado Integration Scripts
- **`fpga/vivado/core.tcl`** - IP core management
- **`fpga/vivado/block_design.tcl`** - Block design template
- **`fpga/config.tcl`** - Configuration template

### TCL Function Categories

#### Pin Management (utilities.tcl)
```tcl
ctl_pin <register_name>              # Control register pin
sts_pin <register_name>              # Status register pin  
get_constant_pin <value> <width>     # Constant value pin
get_slice_pin <signal> <msb> <lsb>   # Signal slice
get_concat_pin <signal_list>         # Signal concatenation
```

#### Cell Creation (utilities.tcl)
```tcl
cell <ip_type> <instance_name> {properties} {connections}
connect_pins <pin1> <pin2>           # Direct pin connection
connect_cell <instance> {connections} # Cell connection
```

#### Memory Management (bram.tcl, ctl_sts.tcl)
```tcl
add_bram <name>                      # Add BRAM instance
add_ctl_register <name>              # Add control register
add_sts_register <name>              # Add status register
get_memory_range <name>              # Get memory range
get_memory_offset <name>             # Get memory offset
```

## Build Flow

### 1. Configuration Processing
```
config.yml → make.py → Parsed configuration object
```

### 2. FPGA Build Flow
```
block_design.tcl → Vivado → Bitstream (.bit)
                ↓
        Core integration
                ↓
        Pin constraints (.xdc)
                ↓
        Implementation
```

### 3. Server Build Flow  
```
Driver headers → Template rendering → C++ source → Cross-compilation → ARM binary
```

### 4. Web Build Flow
```
TypeScript → Compilation → JavaScript + CSS → Asset bundling
```

### 5. Integration
```
Bitstream + Server + Web assets → ZIP package → Deployment to target
```

## Key Design Patterns

### 1. Template-Driven Development
- Jinja2 templates for code generation
- Configuration-driven customization
- Consistent interface patterns

### 2. Layered Architecture
- Hardware abstraction through drivers
- Clean separation of FPGA/software concerns
- Modular, reusable components

### 3. Configuration Management
- YAML-based declarative configuration
- Hierarchical parameter inheritance
- Template-based customization

### 4. Cross-Platform Build System
- Make-based orchestration
- Docker containerization
- Multi-target support

## Development Workflow

### Adding a New Core
1. Create directory in `fpga/cores/`
2. Implement Verilog + `core_config.tcl`  
3. Add to project `config.yml`
4. Use in `block_design.tcl`

### Creating a New Project
1. Create directory in `examples/<board>/`
2. Define `config.yml`
3. Implement `block_design.tcl`
4. Create driver headers
5. Add web interface (optional)
6. Write test scripts

### Debugging TCL Scripts
- Use `puts` for debugging output
- Check Vivado logs in `tmp/` directory
- Validate syntax with `tclsh`
- Use Vivado GUI for interactive debugging

## Performance Considerations

### FPGA Resource Usage
- Monitor LUT/DSP/BRAM utilization
- Optimize critical timing paths  
- Use appropriate IP core configurations

### Memory Management
- Align memory access patterns
- Use DMA for high-throughput transfers
- Consider cache coherency

### Real-time Performance
- Minimize interrupt latency
- Use dedicated DMA channels
- Optimize data path pipelines

## Testing Framework

### Test Types
- **Unit Tests**: Individual core testing
- **Integration Tests**: System-level testing
- **Hardware-in-the-Loop**: Real hardware testing
- **Performance Tests**: Throughput/latency measurement

### Test Organization
```
tests/
├── unit/              # Unit test suites
├── integration/       # Integration tests  
├── performance/       # Performance benchmarks
└── hardware/          # Hardware-specific tests
```

This documentation provides a comprehensive overview of the Koheron SDK architecture. Each component is designed to work together in a cohesive framework for FPGA-based instrument development.

## Current Issue Context (Sample Trigger System)

In our current work with the `examples/alpha250/currentramp/` project, we identified and are fixing a trigger generation system. The system was producing only ~1.5-6 triggers per cycle instead of the expected 1000. Key files involved:

- **`block_design.tcl`** - Contains the FPGA logic for trigger generation
- **`currentramp.hpp`** - C++ driver with trigger control functions  
- **`test_sample_triggers.py`** - Python test script for validation

The issue was in the trigger logic implementation in the TCL block design, where complex division IP was causing configuration errors. The solution involved simplifying to basic counter-based logic following patterns from successful examples like `adc-dac-bram`. 