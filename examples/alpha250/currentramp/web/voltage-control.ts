// Current Ramp Control driver
// (c) Koheron

class VoltageControl {
    private driver: Driver;
    private id: number;
    private cmds: Commands;

    constructor(private client: Client) {
        this.driver = this.client.getDriver('CurrentRamp');
        this.id = this.driver.id;
        this.cmds = this.driver.getCmds();
        
        // Debug: Log available commands
        console.log('Available CurrentRamp commands:', Object.keys(this.cmds));
    }

    // === DC Temperature Control Functions ===
    
    setTemperatureDcVoltage(voltage: number): void {
        console.log('Setting DC voltage to:', voltage);
        this.client.send(Command(this.id, this.cmds['set_temperature_dc_voltage'], voltage));
    }

    enableTemperatureDcOutput(enable: boolean): void {
        console.log('Enabling DC output:', enable);
        this.client.send(Command(this.id, this.cmds['enable_temperature_dc_output'], enable));
    }

    getTemperatureDcVoltage(callback: (voltage: number) => void): void {
        this.client.readFloat32(Command(this.id, this.cmds['get_temperature_dc_voltage']), (voltage) => {
            console.log('Read DC voltage:', voltage);
            callback(voltage);
        });
    }

    getTemperatureDcEnabled(callback: (enabled: boolean) => void): void {
        this.client.readBool(Command(this.id, this.cmds['get_temperature_dc_enabled']), (enabled) => {
            console.log('Read DC enabled:', enabled);
            callback(enabled);
        });
    }

    // === Current Ramp Control Functions ===
    
    setRampOffset(offset: number): void {
        this.client.send(Command(this.id, this.cmds['set_ramp_offset'], offset));
    }

    setRampAmplitude(amplitude: number): void {
        this.client.send(Command(this.id, this.cmds['set_ramp_amplitude'], amplitude));
    }

    setRampFrequency(frequency: number): void {
        this.client.send(Command(this.id, this.cmds['set_ramp_frequency'], frequency));
    }

    startRamp(): void {
        this.client.send(Command(this.id, this.cmds['start_ramp']));
    }

    stopRamp(): void {
        this.client.send(Command(this.id, this.cmds['stop_ramp']));
    }

    generateRampWaveform(): void {
        this.client.send(Command(this.id, this.cmds['generate_ramp_waveform']));
    }

    getRampOffset(callback: (offset: number) => void): void {
        this.client.readFloat32(Command(this.id, this.cmds['get_ramp_offset']), callback);
    }

    getRampAmplitude(callback: (amplitude: number) => void): void {
        this.client.readFloat32(Command(this.id, this.cmds['get_ramp_amplitude']), callback);
    }

    getRampFrequency(callback: (frequency: number) => void): void {
        this.client.readFloat64(Command(this.id, this.cmds['get_ramp_frequency']), callback);
    }

    getRampEnabled(callback: (enabled: boolean) => void): void {
        this.client.readBool(Command(this.id, this.cmds['get_ramp_enabled']), callback);
    }

    // === Precision ADC Functions ===
    
    getPhotodidePrecision(channel: number, callback: (voltage: number) => void): void {
        this.client.readFloat32(Command(this.id, this.cmds['get_photodiode_precision'], channel), callback);
    }

    // === High-Speed BRAM Data Acquisition ===
    
    triggerAcquisition(): void {
        this.client.send(Command(this.id, this.cmds['trigger_acquisition']));
    }

    isAcquisitionComplete(callback: (complete: boolean) => void): void {
        this.client.readBool(Command(this.id, this.cmds['is_acquisition_complete']), callback);
    }

    getAdcSamples1000(callback: (samples: number[]) => void): void {
        this.client.readUint32Array(Command(this.id, this.cmds['get_adc_samples_1000']), (data: Uint32Array) => {
            callback(Array.from(data));
        });
    }

    getDacSamples1000(callback: (samples: number[]) => void): void {
        this.client.readUint32Array(Command(this.id, this.cmds['get_dac_samples_1000']), (data: Uint32Array) => {
            callback(Array.from(data));
        });
    }

    // High-speed paired data acquisition (1000 samples at 250 MHz = 4 microseconds)
    getPairedHighSpeedData(callback: (data: { time: number[], dac: number[], adc: number[] }) => void): void {
        // Trigger hardware acquisition
        this.triggerAcquisition();
        
        // Wait for completion with timeout
        const checkComplete = () => {
            this.isAcquisitionComplete((complete) => {
                if (complete) {
                    // Get both ADC and DAC data
                    this.getAdcSamples1000((adcRaw) => {
                        this.getDacSamples1000((dacRaw) => {
                            // Convert raw data to voltages
                            const data = this.convertBramData(adcRaw, dacRaw);
                            callback(data);
                        });
                    });
                } else {
                    // Check again in 1ms
                    setTimeout(checkComplete, 1);
                }
            });
        };
        
        // Start checking for completion
        setTimeout(checkComplete, 1);
    }

    // Continuous decimated data acquisition (10,000 samples at 10kHz = 1 second)
    getPairedContinuousData10k(callback: (data: { time: number[], dac: number[], adc: number[] }) => void): void {
        console.log('Requesting 10k continuous data...');
        
        // Get 10,000 samples for 1-second window (10 complete periods at 10Hz)
        this.client.readUint32Array(Command(this.id, this.cmds['get_adc_samples_10000']), (adcRaw: Uint32Array) => {
            console.log('Received ADC data:', adcRaw.length, 'samples');
            console.log('First few ADC values:', Array.from(adcRaw.slice(0, 5)));
            
            this.client.readUint32Array(Command(this.id, this.cmds['get_dac_samples_10000']), (dacRaw: Uint32Array) => {
                console.log('Received DAC data:', dacRaw.length, 'samples');
                console.log('First few DAC values:', Array.from(dacRaw.slice(0, 5)));
                
                // Convert raw data to voltages with proper time base
                const data = this.convertContinuousData(Array.from(adcRaw), Array.from(dacRaw));
                console.log('Converted data - ADC range:', Math.min(...data.adc).toFixed(3), 'to', Math.max(...data.adc).toFixed(3), 'V');
                console.log('Converted data - DAC range:', Math.min(...data.dac).toFixed(3), 'to', Math.max(...data.dac).toFixed(3), 'V');
                
                callback(data);
            });
        });
    }

    private convertBramData(adcRaw: number[], dacRaw: number[]): { time: number[], dac: number[], adc: number[] } {
        const fs = 250e6; // 250 MHz sampling rate (full speed, no decimation)
        const n_samples = Math.min(adcRaw.length, dacRaw.length);
        
        const time: number[] = [];
        const dac: number[] = [];
        const adc: number[] = [];
        
        for (let i = 0; i < n_samples; i++) {
            // Time base: sample index / sampling frequency
            time.push(i / fs);
            
            // DAC conversion: Precision DAC data (32-bit format)
            // Lower 16 bits = DAC channel 2 (ramp), Upper 16 bits = DAC channel 3 (unused)
            const dacCh2Raw = dacRaw[i] & 0xFFFF;
            const dacCh3Raw = (dacRaw[i] >> 16) & 0xFFFF;
            
            // DAC channel 2 is the ramp output (16-bit unsigned, 0-2.5V range)
            const dacVoltage = (dacCh2Raw / 65535.0) * 2.5;
            dac.push(dacVoltage);
            
            // ADC conversion: Raw ADC data from fast ADC (LTC2157)
            // Lower 16 bits = ADC channel 0, Upper 16 bits = ADC channel 1
            const adcCh0Raw = adcRaw[i] & 0xFFFF;
            const adcCh1Raw = (adcRaw[i] >> 16) & 0xFFFF;
            
            // Use ADC channel 0 (your input signal)
            // LTC2157: 16-bit signed, ±1.8V range (Alpha250 standard)
            const adcSigned = adcCh0Raw > 32767 ? adcCh0Raw - 65536 : adcCh0Raw;
            const adcVoltage = (adcSigned / 32768.0) * 1.8;
            adc.push(adcVoltage);
        }
        
        return { time, dac, adc };
    }

    private convertContinuousData(adcRaw: number[], dacRaw: number[]): { time: number[], dac: number[], adc: number[] } {
        const fs = 10e3; // 10 kHz effective sampling rate (decimated from 250MHz)
        const n_samples = Math.min(adcRaw.length, dacRaw.length);
        
        const time: number[] = [];
        const dac: number[] = [];
        const adc: number[] = [];
        
        for (let i = 0; i < n_samples; i++) {
            // Time base: decimated samples at 10kHz
            // For 10,000 samples: 10,000 / 10kHz = 1 second total (10 complete periods at 10Hz)
            time.push(i / fs);
            
            // DAC conversion: Precision DAC data (32-bit format)
            // Lower 16 bits = DAC channel 2 (ramp), Upper 16 bits = DAC channel 3 (unused)
            const dacCh2Raw = dacRaw[i] & 0xFFFF;
            
            // DAC channel 2 is the ramp output (16-bit unsigned, 0-2.5V range)
            const dacVoltage = (dacCh2Raw / 65535.0) * 2.5;
            dac.push(dacVoltage);
            
            // ADC conversion: Raw ADC data from fast ADC (LTC2157)
            // Lower 16 bits = ADC channel 0, Upper 16 bits = ADC channel 1
            const adcCh0Raw = adcRaw[i] & 0xFFFF;
            
            // Use ADC channel 0 (your input signal)
            // LTC2157: 16-bit signed, ±1.8V range (Alpha250 standard)
            const adcSigned = adcCh0Raw > 32767 ? adcCh0Raw - 65536 : adcCh0Raw;
            const adcVoltage = (adcSigned / 32768.0) * 1.8;
            adc.push(adcVoltage);
        }
        
        return { time, dac, adc };
    }

    // Legacy functions for backward compatibility
    setVoltageOutput(voltage: number): void {
        this.setTemperatureDcVoltage(voltage);
    }

    enableOutput(enable: boolean): void {
        this.enableTemperatureDcOutput(enable);
    }
}

// Voltage vs Voltage Plot Class - Optimized for Continuous 10kHz Stream
class VoltageVsVoltagePlot {
    public plot_data: Array<Array<Array<number>>>;
    private peakDatapoint: number[];
    private isMonitoring: boolean = false;
    private monitoringInterval: number | null = null;
    private dataBuffer: { time: number[], dac: number[], adc: number[] } = { time: [], dac: [], adc: [] };
    private maxDataPoints: number = 10000; // 10,000 points = 1 second of 10kHz data
    private updateInterval: number = 1000; // Update display every 1000ms (1 second)
    private adcChannel: number = 0;
    private startTime: number = 0;
    public timeWindow: number = 1.0; // Display 1 second = 10 complete periods at 10Hz
    private timeMode: string = "rolling"; // "rolling", "fixed", "full"

    public yLabel: string = "Voltage (V)";

    constructor(private document: Document, private voltageControl: VoltageControl, private plotBasics: PlotBasics) {
        this.peakDatapoint = [];
        this.plot_data = [];
        
        // Initialize the plot basics for our specific requirements
        // Note: PlotBasics doesn't have setXLabel/setYLabel methods
        // The labels are set via the plot title and redraw function
        this.plotBasics.setRangeX(0, 1000); // 1000ms = 1 second window
    }

    startMonitoring(): void {
        if (this.isMonitoring) {
            console.log('Monitoring already started');
            return;
        }

        this.isMonitoring = true;
        this.startTime = Date.now();
        this.clearData();
        
        console.log('Starting continuous 10kHz stream monitoring - updates every 1000ms');

        // Start the continuous monitoring loop
        const updateLoop = () => {
            if (!this.isMonitoring) return;

            console.log('Requesting new data update...');
            
            // Get 10,000 samples (1 second of 10kHz data = 10 complete periods)
            this.voltageControl.getPairedContinuousData10k((data) => {
                if (!this.isMonitoring) return;

                console.log('Received data update with', data.time.length, 'points');
                
                // Update our data buffer with the latest 1-second window
                this.dataBuffer = data;
                this.updatePlotData();
                this.updateStatistics();

                // Schedule next update in 1000ms
                this.monitoringInterval = window.setTimeout(updateLoop, this.updateInterval);
            });
        };

        // Start the first update
        updateLoop();
    }

    stopMonitoring(): void {
        this.isMonitoring = false;
        if (this.monitoringInterval) {
            window.clearTimeout(this.monitoringInterval);
            this.monitoringInterval = null;
        }
        console.log('Stopped continuous monitoring');
    }

    pauseMonitoring(): void {
        if (this.monitoringInterval) {
            window.clearTimeout(this.monitoringInterval);
            this.monitoringInterval = null;
        }
        console.log('Paused monitoring (data retained)');
    }

    clearData(): void {
        this.dataBuffer = { time: [], dac: [], adc: [] };
        this.plot_data = [];
        this.peakDatapoint = [];
        
        // Clear the plot
        if (this.plotBasics) {
            this.plotBasics.redraw([], 0, [], this.yLabel, () => {});
        }
        
        console.log('Cleared plot data');
    }

    private updatePlotData(): void {
        if (this.dataBuffer.time.length === 0) return;

        // Convert time from seconds to milliseconds for display
        const timeMs = this.dataBuffer.time.map(t => t * 1000);
        
        // Create plot data arrays in the format expected by PlotBasics
        // PlotBasics expects a single array of [x, y] pairs for redraw()
        const combinedPlotData: Array<Array<number>> = [];

        // Add ADC data points
        for (let i = 0; i < timeMs.length; i++) {
            combinedPlotData.push([timeMs[i], this.dataBuffer.adc[i]]);
        }

        // Store for dual-channel plotting if needed later
        this.plot_data = [combinedPlotData]; // Single channel for now

        // Find peak for display
        if (this.dataBuffer.adc.length > 0) {
            const maxIndex = this.dataBuffer.adc.reduce((maxIdx, val, idx, arr) => 
                Math.abs(val) > Math.abs(arr[maxIdx]) ? idx : maxIdx, 0);
            this.peakDatapoint = [timeMs[maxIndex], this.dataBuffer.adc[maxIndex]];
        }

        // Update the plot with ADC data
        this.plotBasics.redraw(
            combinedPlotData, 
            this.dataBuffer.time.length, 
            this.peakDatapoint, 
            "ADC Input (V)", 
            () => {} // No callback needed for continuous mode
        );
    }

    private updateStatistics(): void {
        if (this.dataBuffer.time.length === 0) return;

        // Calculate statistics for both channels
        const adcStats = this.calculateChannelStats(this.dataBuffer.adc, "ADC Input");
        const dacStats = this.calculateChannelStats(this.dataBuffer.dac, "DAC Output");
        
        // Calculate correlation between ADC and DAC
        const correlation = this.calculateCorrelation(this.dataBuffer.adc, this.dataBuffer.dac);
        
        // Update display elements
        const statsElement = this.document.getElementById('plot-statistics');
        if (statsElement) {
            statsElement.innerHTML = `
                <div class="row">
                    <div class="col-md-4">
                        <h5>ADC Input (Channel ${this.adcChannel})</h5>
                        <p>Min: ${adcStats.min.toFixed(3)}V</p>
                        <p>Max: ${adcStats.max.toFixed(3)}V</p>
                        <p>Avg: ${adcStats.avg.toFixed(3)}V</p>
                        <p>RMS: ${adcStats.rms.toFixed(3)}V</p>
                    </div>
                    <div class="col-md-4">
                        <h5>DAC Output</h5>
                        <p>Min: ${dacStats.min.toFixed(3)}V</p>
                        <p>Max: ${dacStats.max.toFixed(3)}V</p>
                        <p>Avg: ${dacStats.avg.toFixed(3)}V</p>
                        <p>RMS: ${dacStats.rms.toFixed(3)}V</p>
                    </div>
                    <div class="col-md-4">
                        <h5>System Status</h5>
                        <p>Correlation: ${correlation.toFixed(3)}</p>
                        <p>Data Points: ${this.dataBuffer.time.length}</p>
                        <p>Sample Rate: 10 kHz</p>
                        <p>Window: ${this.timeWindow.toFixed(1)}s (10 periods)</p>
                        <p>Update Rate: ${this.updateInterval}ms</p>
                    </div>
                </div>
            `;
        }
    }

    private calculateChannelStats(data: number[], name: string) {
        if (data.length === 0) return { min: 0, max: 0, avg: 0, rms: 0 };
        
        const min = Math.min(...data);
        const max = Math.max(...data);
        const avg = data.reduce((sum, val) => sum + val, 0) / data.length;
        const rms = Math.sqrt(data.reduce((sum, val) => sum + val * val, 0) / data.length);
        
        return { min, max, avg, rms };
    }

    private calculateCorrelation(x: number[], y: number[]): number {
        if (x.length !== y.length || x.length === 0) return 0;
        
        const n = x.length;
        const sumX = x.reduce((a, b) => a + b, 0);
        const sumY = y.reduce((a, b) => a + b, 0);
        const sumXY = x.reduce((sum, xi, i) => sum + xi * y[i], 0);
        const sumX2 = x.reduce((sum, xi) => sum + xi * xi, 0);
        const sumY2 = y.reduce((sum, yi) => sum + yi * yi, 0);
        
        const numerator = n * sumXY - sumX * sumY;
        const denominator = Math.sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY));
        
        return denominator === 0 ? 0 : numerator / denominator;
    }

    // Configuration methods
    setUpdateInterval(intervalMs: number): void {
        this.updateInterval = Math.max(100, intervalMs); // Minimum 100ms
        console.log(`Update interval set to ${this.updateInterval}ms`);
    }

    setMaxDataPoints(points: number): void {
        // For 10kHz stream, this should be 10000 for 1-second window
        this.maxDataPoints = Math.max(1000, Math.min(points, 50000));
        console.log(`Max data points set to ${this.maxDataPoints}`);
    }

    setAdcChannel(channel: number): void {
        this.adcChannel = channel;
        console.log(`ADC channel set to ${channel}`);
    }

    setTimeWindow(seconds: number): void {
        this.timeWindow = Math.max(0.1, Math.min(seconds, 5.0));
        this.plotBasics.setRangeX(0, this.timeWindow * 1000); // Convert to ms
        console.log(`Time window set to ${this.timeWindow}s`);
    }

    setTimeMode(mode: string): void {
        this.timeMode = mode;
        console.log(`Time mode set to ${mode}`);
    }

    // Zoom controls
    zoomIn(): void {
        this.timeWindow = Math.max(0.1, this.timeWindow * 0.5);
        this.setTimeWindow(this.timeWindow);
    }

    zoomOut(): void {
        this.timeWindow = Math.min(5.0, this.timeWindow * 2.0);
        this.setTimeWindow(this.timeWindow);
    }

    resetZoom(): void {
        this.setTimeWindow(1.0); // Reset to 1 second = 10 periods
    }

    // Data export
    exportData(): string {
        if (this.dataBuffer.time.length === 0) {
            return "No data to export";
        }

        let csv = "Time(s),ADC_Input(V),DAC_Output(V)\n";
        for (let i = 0; i < this.dataBuffer.time.length; i++) {
            csv += `${this.dataBuffer.time[i]},${this.dataBuffer.adc[i]},${this.dataBuffer.dac[i]}\n`;
        }

        const blob = new Blob([csv], { type: 'text/csv' });
        const url = URL.createObjectURL(blob);
        const a = this.document.createElement('a');
        a.href = url;
        a.download = `currentramp_data_${new Date().toISOString().slice(0, 19).replace(/:/g, '-')}.csv`;
        a.click();
        URL.revokeObjectURL(url);

        return `Exported ${this.dataBuffer.time.length} data points`;
    }
}

class VoltageControlApp {

    // Default values for ramp parameters
    private static readonly DEFAULT_FREQUENCY = 10.0;
    private static readonly DEFAULT_AMPLITUDE = 1.0;
    private static readonly DEFAULT_OFFSET = 1.5;

    // DC Control Elements
    private dcOutputToggle: HTMLInputElement;
    private dcVoltageInput: HTMLInputElement;
    private dcVoltageDisplay: HTMLSpanElement;
    private dcStatusDisplay: HTMLSpanElement;
    private setDcVoltageBtn: HTMLButtonElement;

    // Ramp Control Elements
    private rampEnableToggle: HTMLInputElement;
    private rampStatusDisplay: HTMLSpanElement;
    
    private frequencyInput: HTMLInputElement;
    private amplitudeInput: HTMLInputElement;
    private offsetInput: HTMLInputElement;
    
    private setFrequencyBtn: HTMLButtonElement;
    private setAmplitudeBtn: HTMLButtonElement;
    private setOffsetBtn: HTMLButtonElement;
    private resetDefaultsBtn: HTMLButtonElement;
    
    private currentFrequencySpan: HTMLSpanElement;
    private currentAmplitudeSpan: HTMLSpanElement;
    private currentOffsetSpan: HTMLSpanElement;

    // Graphing Elements
    private graphStatusDisplay: HTMLSpanElement;
    private samplingRateSelect: HTMLSelectElement;
    private dataPointsSelect: HTMLSelectElement;
    private adcChannelSelect: HTMLSelectElement;
    
    private startGraphingBtn: HTMLButtonElement;
    private pauseGraphingBtn: HTMLButtonElement;
    private stopGraphingBtn: HTMLButtonElement;
    private clearGraphBtn: HTMLButtonElement;
    private exportDataBtn: HTMLButtonElement;

    // Time Control Elements
    private timeWindowSelect: HTMLSelectElement;

    // Plotting infrastructure
    private plotBasics: PlotBasics | null = null;
    private voltagePlot: VoltageVsVoltagePlot | null = null;

    constructor(document: Document, private voltageControl: VoltageControl) {
        // DC Control Elements
        this.dcOutputToggle = <HTMLInputElement>document.getElementById('dc-output-toggle');
        this.dcVoltageInput = <HTMLInputElement>document.getElementById('dc-voltage-input');
        this.dcVoltageDisplay = <HTMLSpanElement>document.getElementById('dc-voltage-display');
        this.dcStatusDisplay = <HTMLSpanElement>document.getElementById('dc-status-display');
        this.setDcVoltageBtn = <HTMLButtonElement>document.getElementById('set-dc-voltage');

        // Ramp Control Elements
        this.rampEnableToggle = <HTMLInputElement>document.getElementById('ramp-enable-toggle');
        this.rampStatusDisplay = <HTMLSpanElement>document.getElementById('ramp-status-display');
        
        this.frequencyInput = <HTMLInputElement>document.getElementById('frequency-input');
        this.amplitudeInput = <HTMLInputElement>document.getElementById('amplitude-input');
        this.offsetInput = <HTMLInputElement>document.getElementById('offset-input');
        
        this.setFrequencyBtn = <HTMLButtonElement>document.getElementById('set-frequency');
        this.setAmplitudeBtn = <HTMLButtonElement>document.getElementById('set-amplitude');
        this.setOffsetBtn = <HTMLButtonElement>document.getElementById('set-offset');
        this.resetDefaultsBtn = <HTMLButtonElement>document.getElementById('reset-defaults');
        
        this.currentFrequencySpan = <HTMLSpanElement>document.getElementById('current-frequency');
        this.currentAmplitudeSpan = <HTMLSpanElement>document.getElementById('current-amplitude');
        this.currentOffsetSpan = <HTMLSpanElement>document.getElementById('current-offset');

        // Graphing Elements
        this.graphStatusDisplay = <HTMLSpanElement>document.getElementById('graph-status-display');
        this.samplingRateSelect = <HTMLSelectElement>document.getElementById('sampling-rate-select');
        this.dataPointsSelect = <HTMLSelectElement>document.getElementById('data-points-select');
        this.adcChannelSelect = <HTMLSelectElement>document.getElementById('adc-channel-select');
        
        this.startGraphingBtn = <HTMLButtonElement>document.getElementById('start-graphing');
        this.pauseGraphingBtn = <HTMLButtonElement>document.getElementById('pause-graphing');
        this.stopGraphingBtn = <HTMLButtonElement>document.getElementById('stop-graphing');
        this.clearGraphBtn = <HTMLButtonElement>document.getElementById('clear-graph');
        this.exportDataBtn = <HTMLButtonElement>document.getElementById('export-data');

        // Time Control Elements
        this.timeWindowSelect = <HTMLSelectElement>document.getElementById('time-window-select');

        // Initialize plotting infrastructure after DOM elements are ready
        setTimeout(() => this.initializePlotting(), 100);

        this.setupEventListeners();
        this.updateStatus();
    }

    private initializePlotting(): void {
        const plotPlaceholder = $('#voltage-plot-placeholder');
        if (plotPlaceholder.length > 0) {
            // Initialize plot basics for time-series plot
            const n_pts = 1000;
            const x_min = 0;    // Time min (seconds)
            const x_max = 10;   // Time max (seconds) - initial range
            const y_min = -0.6; // Voltage min
            const y_max = 2.5;  // Voltage max (to accommodate both DAC and ADC)

            this.plotBasics = new PlotBasics(
                document, 
                plotPlaceholder, 
                n_pts, 
                x_min, x_max, 
                y_min, y_max, 
                this.voltageControl, 
                "", 
                "Time (seconds)"
            );

            this.voltagePlot = new VoltageVsVoltagePlot(document, this.voltageControl, this.plotBasics);
            
            console.log('Plotting infrastructure initialized');
        } else {
            console.error('Plot placeholder not found');
        }
    }

    private setupEventListeners(): void {
        // DC Control Event Listeners
        this.dcOutputToggle.addEventListener('change', (event) => {
            const enabled = (<HTMLInputElement>event.target).checked;
            if (enabled) {
                // When enabling, first set the voltage then enable output
                const voltage = parseFloat(this.dcVoltageInput.value);
                if (!isNaN(voltage) && voltage >= 0 && voltage <= 2.5) {
                    // Update UI immediately for responsiveness
                    this.dcStatusDisplay.textContent = 'Enabled';
                    this.dcStatusDisplay.className = 'label label-success';
                    
                    this.voltageControl.setTemperatureDcVoltage(voltage);
                    // Shorter delay between commands
                    setTimeout(() => {
                        this.voltageControl.enableTemperatureDcOutput(enabled);
                        // Shorter delay before status check
                        setTimeout(() => this.updateStatus(), 150);
                    }, 50);
                } else {
                    // If voltage is invalid, don't enable and show error
                    this.dcOutputToggle.checked = false;
                    alert('Please set a valid voltage between 0 and 2.5V before enabling');
                    return;
                }
            } else {
                // When disabling, update UI immediately and disable quickly
                this.dcStatusDisplay.textContent = 'Disabled';
                this.dcStatusDisplay.className = 'label label-default';
                this.voltageControl.enableTemperatureDcOutput(enabled);
                setTimeout(() => this.updateStatus(), 100);
            }
        });

        this.setDcVoltageBtn.addEventListener('click', () => {
            const voltage = parseFloat(this.dcVoltageInput.value);
            if (!isNaN(voltage) && voltage >= 0 && voltage <= 2.5) {
                this.voltageControl.setTemperatureDcVoltage(voltage);
                setTimeout(() => this.updateStatus(), 100);
            } else {
                alert('Please enter a valid voltage between 0 and 2.5V');
            }
        });

        // Ramp Control Event Listeners
        this.rampEnableToggle.addEventListener('change', (event) => {
            const enabled = (<HTMLInputElement>event.target).checked;
            if (enabled) {
                // Update UI immediately
                this.rampStatusDisplay.textContent = 'Enabled';
                this.rampStatusDisplay.className = 'label label-success';
                
                // Generate waveform and start ramp
                this.voltageControl.generateRampWaveform();
                setTimeout(() => {
                    this.voltageControl.startRamp();
                    setTimeout(() => this.updateStatus(), 150);
                }, 50);
            } else {
                // Update UI immediately
                this.rampStatusDisplay.textContent = 'Disabled';
                this.rampStatusDisplay.className = 'label label-default';
                
                this.voltageControl.stopRamp();
                setTimeout(() => this.updateStatus(), 100);
            }
        });

        this.setFrequencyBtn.addEventListener('click', () => {
            const frequency = parseFloat(this.frequencyInput.value);
            if (!isNaN(frequency) && frequency >= 0.1 && frequency <= 1000) {
                this.voltageControl.setRampFrequency(frequency);
                setTimeout(() => this.updateStatus(), 100);
            } else {
                alert('Please enter a valid frequency between 0.1 and 1000 Hz');
            }
        });

        this.setAmplitudeBtn.addEventListener('click', () => {
            const amplitude = parseFloat(this.amplitudeInput.value);
            if (!isNaN(amplitude) && amplitude >= 0 && amplitude <= 2.5) {
                this.voltageControl.setRampAmplitude(amplitude);
                setTimeout(() => this.updateStatus(), 100);
            } else {
                alert('Please enter a valid amplitude between 0 and 2.5V');
            }
        });

        this.setOffsetBtn.addEventListener('click', () => {
            const offset = parseFloat(this.offsetInput.value);
            if (!isNaN(offset) && offset >= 0 && offset <= 2.5) {
                this.voltageControl.setRampOffset(offset);
                setTimeout(() => this.updateStatus(), 100);
            } else {
                alert('Please enter a valid offset between 0 and 2.5V');
            }
        });

        this.resetDefaultsBtn.addEventListener('click', () => {
            this.frequencyInput.value = VoltageControlApp.DEFAULT_FREQUENCY.toString();
            this.amplitudeInput.value = VoltageControlApp.DEFAULT_AMPLITUDE.toString();
            this.offsetInput.value = VoltageControlApp.DEFAULT_OFFSET.toString();
            
            this.voltageControl.setRampFrequency(VoltageControlApp.DEFAULT_FREQUENCY);
            this.voltageControl.setRampAmplitude(VoltageControlApp.DEFAULT_AMPLITUDE);
            this.voltageControl.setRampOffset(VoltageControlApp.DEFAULT_OFFSET);
            
            setTimeout(() => this.updateStatus(), 200);
        });

        // Graphing Event Listeners
        this.setupGraphingEventListeners();
    }

    private setupGraphingEventListeners(): void {
        // Start graphing button
        this.startGraphingBtn.addEventListener('click', () => {
            this.startGraphing();
        });

        // Pause graphing button
        this.pauseGraphingBtn.addEventListener('click', () => {
            this.pauseGraphing();
        });

        // Stop graphing button
        this.stopGraphingBtn.addEventListener('click', () => {
            this.stopGraphing();
        });

        // Clear graph button
        this.clearGraphBtn.addEventListener('click', () => {
            if (this.voltagePlot) {
                this.voltagePlot.clearData();
            }
        });

        // Export data button
        this.exportDataBtn.addEventListener('click', () => {
            this.exportData();
        });

        // Sampling rate change
        this.samplingRateSelect.addEventListener('change', () => {
            const interval = parseInt(this.samplingRateSelect.value);
            if (this.voltagePlot) {
                this.voltagePlot.setUpdateInterval(interval);
            }
        });

        // Data points change
        this.dataPointsSelect.addEventListener('change', () => {
            const points = parseInt(this.dataPointsSelect.value);
            if (this.voltagePlot) {
                this.voltagePlot.setMaxDataPoints(points);
            }
        });

        // ADC channel change
        this.adcChannelSelect.addEventListener('change', () => {
            const channel = parseInt(this.adcChannelSelect.value);
            if (this.voltagePlot) {
                this.voltagePlot.setAdcChannel(channel);
            }
        });

        // Time window change
        this.timeWindowSelect.addEventListener('change', () => {
            const timeWindow = parseFloat(this.timeWindowSelect.value);
            if (this.voltagePlot) {
                this.voltagePlot.setTimeWindow(timeWindow);
            }
        });
    }

    private startGraphing(): void {
        if (!this.voltagePlot) {
            console.error('Voltage plot not initialized');
            return;
        }

        // Update UI
        this.graphStatusDisplay.textContent = 'Running';
        this.graphStatusDisplay.className = 'label label-success';
        
        this.startGraphingBtn.disabled = true;
        this.pauseGraphingBtn.disabled = false;
        this.stopGraphingBtn.disabled = false;

        // Apply current settings
        const updateInterval = parseInt(this.samplingRateSelect.value);
        const maxPoints = parseInt(this.dataPointsSelect.value);
        const adcChannel = parseInt(this.adcChannelSelect.value);
        const timeWindow = parseFloat(this.timeWindowSelect.value);

        this.voltagePlot.setUpdateInterval(updateInterval);
        this.voltagePlot.setMaxDataPoints(maxPoints);
        this.voltagePlot.setAdcChannel(adcChannel);
        this.voltagePlot.setTimeWindow(timeWindow);

        // Start monitoring
        this.voltagePlot.startMonitoring();
        
        console.log('Started continuous monitoring');
    }

    private pauseGraphing(): void {
        if (!this.voltagePlot) return;

        // Update UI
        this.graphStatusDisplay.textContent = 'Paused';
        this.graphStatusDisplay.className = 'label label-warning';
        
        this.startGraphingBtn.disabled = false;
        this.pauseGraphingBtn.disabled = true;

        // Pause monitoring
        this.voltagePlot.pauseMonitoring();
        
        console.log('Paused continuous monitoring');
    }

    private stopGraphing(): void {
        if (!this.voltagePlot) return;

        // Update UI
        this.graphStatusDisplay.textContent = 'Stopped';
        this.graphStatusDisplay.className = 'label label-default';
        
        this.startGraphingBtn.disabled = false;
        this.pauseGraphingBtn.disabled = true;
        this.stopGraphingBtn.disabled = true;

        // Stop monitoring
        this.voltagePlot.stopMonitoring();
        
        console.log('Stopped continuous monitoring');
    }

    private exportData(): void {
        if (!this.voltagePlot) return;

        const data = this.voltagePlot.exportData();
        const blob = new Blob([data], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        
        const a = document.createElement('a');
        a.href = url;
        a.download = `currentramp_data_${new Date().toISOString().slice(0, 19).replace(/:/g, '-')}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        
        console.log('Exported continuous data');
    }

    private updateStatus(): void {
        // Update DC status
        this.voltageControl.getTemperatureDcEnabled((enabled) => {
            this.dcOutputToggle.checked = enabled;
            this.dcStatusDisplay.textContent = enabled ? 'Enabled' : 'Disabled';
            this.dcStatusDisplay.className = enabled ? 'label label-success' : 'label label-default';
        });

        this.voltageControl.getTemperatureDcVoltage((voltage) => {
            this.dcVoltageDisplay.textContent = voltage.toFixed(3);
        });

        // Update ramp status
        this.voltageControl.getRampEnabled((enabled) => {
            this.rampEnableToggle.checked = enabled;
            this.rampStatusDisplay.textContent = enabled ? 'Enabled' : 'Disabled';
            this.rampStatusDisplay.className = enabled ? 'label label-success' : 'label label-default';
        });

        // Update current ramp parameters
        this.voltageControl.getRampFrequency((frequency) => {
            this.currentFrequencySpan.textContent = frequency.toFixed(1);
        });

        this.voltageControl.getRampAmplitude((amplitude) => {
            this.currentAmplitudeSpan.textContent = amplitude.toFixed(3);
        });

        this.voltageControl.getRampOffset((offset) => {
            this.currentOffsetSpan.textContent = offset.toFixed(3);
        });
    }
} 