// Oscilloscope plot widget
// (c) Koheron

class OscilloscopePlot {
    public n_pts: number;
    public plot_data: Array<Array<number>>;
    public yLabel: string = "Voltage (V)";
    private simplePlot: SimplePlot;

    constructor(private oscilloscope: any) {
        this.n_pts = this.oscilloscope.buffer_size;
        this.plot_data = [];
        this.simplePlot = new SimplePlot("#oscilloscope-plot-placeholder");
        this.updatePlot();
    }

    updatePlot() {
        // Trigger ADC acquisition first
        this.oscilloscope.triggerAdc();
        
        // Small delay to allow acquisition to complete
        setTimeout(() => {
            this.oscilloscope.readAdcDataVolts((voltageData: Float32Array) => {
                // Calculate time axis based on actual sampling rate
                let sampling_rate_hz: number = this.oscilloscope.status.sampling_rate;
                let time_step_us: number = (1e6 / sampling_rate_hz); // Time step in microseconds
                
                // Calculate total time span of the buffer
                let total_time_us: number = this.n_pts * time_step_us;
                let total_time_ms: number = total_time_us / 1000;
                
                // Set X-axis range based on actual data time span
                this.simplePlot.setRangeX(0, total_time_us);
                
                // Set Y-axis range for voltage (±0.6V for some margin around ±0.5V)
                this.simplePlot.setRangeY(-0.6, 0.6);

                // Convert voltage data to plot format
                for (let i: number = 0; i < this.n_pts; i++) {
                    let time_us: number = i * time_step_us;
                    this.plot_data[i] = [time_us, voltageData[i]];
                }

                this.simplePlot.redraw(this.plot_data, this.yLabel, () => {
                    requestAnimationFrame(() => { this.updatePlot(); });
                });
            });
        }, 10); // 10ms delay for acquisition
    }
} 