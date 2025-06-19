// Oscilloscope plot widget
// (c) Koheron

class OscilloscopePlot {
    public n_pts: number;
    public plot_data: Array<Array<number>>;
    public yLabel: string = "Voltage (V)";
    private plot_placeholder: JQuery;
    private plot: any;
    private oscilloscopeApp: any; // Reference to the app for getting scale settings
    private capturedData: Float32Array | null = null; // Store captured data
    private isAcquiring: boolean = false; // Track acquisition state

    constructor(private oscilloscope: any, oscilloscopeApp: any) {
        this.n_pts = this.oscilloscope.buffer_size;
        this.plot_data = [];
        this.oscilloscopeApp = oscilloscopeApp;
        this.plot_placeholder = $("#oscilloscope-plot-placeholder");
        this.startDisplayLoop();
    }

    // Trigger new data acquisition (simplified like adc-dac-bram example)
    triggerAcquisition() {
        if (this.isAcquiring) {
            console.log("Acquisition already in progress");
            return;
        }
        
        this.isAcquiring = true;
        console.log(`Triggering ADC acquisition for ${this.n_pts} samples...`);
        
        // Read ADC data directly (backend handles trigger automatically)
        console.log("Calling oscilloscope.readAdcDataVolts...");
        try {
            this.oscilloscope.readAdcDataVolts((voltageData: Float32Array) => {
                console.log("readAdcDataVolts callback called!");
                if (voltageData && voltageData.length > 0) {
                    this.capturedData = voltageData;
                    console.log(`Successfully captured ${voltageData.length} samples`);
                    this.updateDisplay(); // Update display with new data
                } else {
                    console.error("Failed to capture ADC data - received empty or null data", voltageData);
                }
                this.isAcquiring = false;
            });
        } catch (error) {
            console.error("Error calling readAdcDataVolts:", error);
            this.isAcquiring = false;
        }
    }
    
    // Update display with current captured data (doesn't retrigger)
    updateDisplay() {
        if (!this.capturedData) {
            console.log("No data captured yet");
            return;
        }
        
        // Calculate time axis based on actual sampling rate
        let sampling_rate_hz: number = this.oscilloscope.status.sampling_rate;
        let time_step_us: number = (1e6 / sampling_rate_hz); // Time step in microseconds
        
        // Get current scale settings from the app
        let timePerDiv_us: number = this.oscilloscopeApp ? this.oscilloscopeApp.getTimePerDiv() : 2;
        let voltagePerDiv: number = this.oscilloscopeApp ? this.oscilloscopeApp.getVoltagePerDiv() : 0.002;
        
        // Calculate display ranges based on oscilloscope-style divisions (10 divisions typical)
        let time_range_us: number = timePerDiv_us * 10; // 10 divisions
        let voltage_range: number = voltagePerDiv * 8; // 8 divisions (±4 divisions)
        
        // Convert voltage data to plot format, but only show data within time range
        let samples_to_show: number = Math.min(this.n_pts, Math.floor(time_range_us / time_step_us));
        
        // Debug info (enhanced)
        let total_buffer_time_us = this.n_pts * time_step_us;
        let minVoltage = Math.min.apply(Math, Array.from(this.capturedData));
        let maxVoltage = Math.max.apply(Math, Array.from(this.capturedData));
        let voltageSpan = maxVoltage - minVoltage;
        
        console.log(`Display: ${timePerDiv_us}µs/div (${time_range_us}µs total), ${voltagePerDiv*1000}mV/div (±${voltage_range*1000}mV range)`);
        console.log(`Buffer: ${total_buffer_time_us.toFixed(1)}µs, Samples to show: ${samples_to_show}/${this.n_pts}`);
        console.log(`Signal: ${minVoltage.toFixed(4)}V to ${maxVoltage.toFixed(4)}V (span: ${voltageSpan.toFixed(4)}V = ${(voltageSpan*1000).toFixed(2)}mV)`);
        
        // Clear plot data array
        this.plot_data = [];
        
        for (let i: number = 0; i < samples_to_show; i++) {
            let time_us: number = i * time_step_us;
            this.plot_data[i] = [time_us, this.capturedData[i]];
        }

        // Update plot using jQuery Flot
        this.plot = (<any>$).plot(this.plot_placeholder, [this.plot_data], {
            series: {
                lines: { show: true, lineWidth: 1 },
                points: { show: false }
            },
            grid: { 
                show: true,
                backgroundColor: "#000",
                borderColor: "#666",
                borderWidth: 1
            },
            xaxis: {
                min: 0,
                max: time_range_us,
                axisLabel: "Time (µs)",
                axisLabelUseCanvas: true,
                axisLabelFontSizePixels: 12,
                axisLabelFontFamily: 'Arial',
                axisLabelPadding: 10,
                color: "#fff"
            },
            yaxis: {
                min: -voltage_range,
                max: voltage_range,
                axisLabel: this.yLabel,
                axisLabelUseCanvas: true,
                axisLabelFontSizePixels: 12,
                axisLabelFontFamily: 'Arial',
                axisLabelPadding: 3,
                color: "#fff"
            },
            colors: ["#0f0"]
        });
    }
    
    // Start the display refresh loop (updates display, doesn't retrigger data)
    startDisplayLoop() {
        // Trigger initial acquisition to get test data
        setTimeout(() => {
            console.log("Starting initial data acquisition...");
            this.triggerAcquisition();
        }, 1000); // Wait 1 second for initialization
        
        setInterval(() => {
            this.updateDisplay(); // Only update display, don't retrigger
        }, 50); // 20 FPS display refresh
    }
} 