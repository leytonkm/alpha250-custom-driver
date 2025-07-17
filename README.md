# FPGA Spectroscopy Driver

## Table of Contents:

## Introduction:

This repository contains my work on creating a custom spectroscopy control system using FPGA devices from Koheron. During my project, I used both the ALPHA250 and ALPHA150 instruments. The latest version uses the ALPHA150 board, due to its higher RF ADC resolution, which is more practical for spectroscopy applications.

## Technical Highlights:
- ### FPGA Hardware Design:
    - Custom DDS-based Hardware Ramp Generator
        - Includes customizable timing control, with programmable frequency (up to 120 MHz), amplitude, and offset controls
    - High-Performance Continuous DMA Streaming
        - Complete DMA infrastructure using cirucular buffer management
        - Supports streaming rate up to 250 MSPS (ALPHA250), 15 MSPS (ALPHA15)
    - Signal Processing Pipeline
        - Implemented CIC decimation that allowed for configurable capture rates
        - Used a FIR compensation filter chain for noise reduction and anti-aliasing
- ### C++ Driver:
    - Memory-Mapped I/O Management
        - Advanced memory manangement for multiple address spaces including control, status, DMA, OCM, and more
    - Perforamnce Optimizations
        - Achieved 100x speedup through optimized data transfer and array retrieval to record billions of millions of datapoints
    - Hardware Abstraction:
        - Created a clean abstraction over Vivado Xilinx IP blocks
- ### PyQt and Web Applications:
    - Real-Time Oscilloscope and Live View:
        - Live ADC visualization with autoscaling time and voltage scales
    - Trigger System
        - Built multiple advanced trigger modes including peak, zero-cross, hysteresis, and edge-detection to handle complex spectoscopy graphs
        - Automatically detected periods with template matching for stable triggering views
    - Performance Optimizations:
        - Efficient circular buffer management and optimized plot updates, interacting with the C++ Driver and Vivado block design
    - TypeScript Web Application:
        - Allows for hardware ramp and temperature control from any device over Wi-Fi
        - Clean and professional interface with toggle switches and real-time status indicators
- ### Testing & Validation Infrastructure:
    - Developed 15+ Test Scripts covering performance, scaling, DMA streaming, timing validation, and hardware verification
    - Automated tests validating 100x+ performance improvements and sustained operation
    - Real-time DMA diagnostics, buffer monitoring, and error detection

 

## Core Files & Modules:

### **Koheron ALPHA15 Instrument**
- [Complete ALPHA15 Instrument](/alpha15-laser-control)
- [C++ Driver](/alpha15-laser-control/laser-control.hpp)
- [Vivado Zynq Block Design (TCL)](/alpha15-laser-control/block_design.tcl)
- [Config File](/alpha15-laser-control/config.yml)
- [Live ADC PyQT App](/alpha15-laser-control/pyqt-app.py)
- [Webapp](/alpha15-laser-control/web)

### **Koheron ALPHA250 Instrument**
- [Complete ALPHA250 Instrument](/alpha250-laser-control)


## Koheron Instrument:

This project showcases a complete system built using Koheron FPGA instruments, known for their high-performance data acquisition and control capabilities in optics and signal processing applications. My work is built on top of Koheron’s open-source platform.

> *"The ALPHA15 is a programmable board built around a **Zynq 7020
> SoC**. It features **two 18-bit 15 Msps ADCs** with high dynamic range,
> low noise front-ends. The high input impedance (up to 1 MΩ) is
> easy to drive and allows to directly interface sensors. Two input
> ranges are selectionnable: 2 Vpp (± 1 V) or 8 Vpp (± 4 V). Thanks to
> the very low flicker noise corner frequency (below 50 Hz) the
> ALPHA15 excels in high oversampling applications. The ALPHA15
> also features a **dual-channel 16-bit 250 Msps low latency DAC** and
> a **4-channel 16-bit precision DAC**. The high speed data converters
> are clocked by a dual PLL, ultra-low jitter clock generator. The board
> comes with a comprehensive, open source, FPGA / Linux reference
> design."*

### Resources:

- [Koheron-SDK GitHub](https://github.com/Koheron/koheron-sdk)
- [ALPHA15 User Guide](https://www.koheron.com/support/user-guides/alpha15/)
- [ALPHA15 Data Sheet (PDF)](https://assets.koheron.com/datasheets/koheron_alpha15-18-bit-15-msps-acquisition-board.pdf)
- [ALPHA15 User Guide](https://www.koheron.com/support/user-guides/alpha250/)
- [ALPHA15 Data Sheet (PDF)](https://assets.koheron.com/datasheets/koheron_alpha250-signal-acquisition-generation.pdf)

## Simple Setup:

I have provided a brief tutorial to set up an ALPHA250 or ALPHA15 device using this source code.

