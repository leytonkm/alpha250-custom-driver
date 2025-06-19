// Oscilloscope app widget
// (c) Koheron

class OscilloscopeApp {
    private oscilloscopeInputs: HTMLInputElement[];
    private oscilloscopeSelects: HTMLSelectElement[];
    private runStopBtn: HTMLButtonElement;
    private singleBtn: HTMLButtonElement;
    private autoScaleBtn: HTMLButtonElement;
    private samplingRateDisplay: HTMLSpanElement;
    private bufferSizeDisplay: HTMLSpanElement;
    private timeSpanDisplay: HTMLSpanElement;
    private acquisitionStateDisplay: HTMLSpanElement;
    
    private isRunning: boolean = false;
    private animationId: number = 0;
    private oscilloscopePlot: any = null; // Reference to the plot
    
    // Oscilloscope settings
    private timePerDiv: number = 26.2; // microseconds (default 26.2µs/div - fits buffer exactly)
    private voltagePerDiv: number = 0.002; // volts (default 2mV/div)
    private triggerLevel: number = 0.0; // volts

    constructor(document: Document, private oscilloscope: any) {
        this.oscilloscopeInputs = <HTMLInputElement[]><any>document.getElementsByClassName("oscilloscope-input");
        this.oscilloscopeSelects = <HTMLSelectElement[]><any>document.querySelectorAll("select.oscilloscope-input");
        this.runStopBtn = <HTMLButtonElement>document.getElementById('run-stop-btn');
        this.singleBtn = <HTMLButtonElement>document.getElementById('single-btn');
        this.autoScaleBtn = <HTMLButtonElement>document.getElementById('auto-scale-btn');
        this.samplingRateDisplay = <HTMLSpanElement>document.getElementById('sampling-rate-display');
        this.bufferSizeDisplay = <HTMLSpanElement>document.getElementById('buffer-size-display');
        this.timeSpanDisplay = <HTMLSpanElement>document.getElementById('time-span-display');
        this.acquisitionStateDisplay = <HTMLSpanElement>document.getElementById('acquisition-state');
        
        this.initInputs();
        this.initSelects();
        this.initButtons();
        this.updateStatus();
    }

    private initInputs(): void {
        for (let i = 0; i < this.oscilloscopeInputs.length; i++) {
            this.oscilloscopeInputs[i].addEventListener('change', (event) => {
                const value = parseFloat((<HTMLInputElement>event.currentTarget).value);
                const command = (<HTMLInputElement>event.currentTarget).dataset.command;
                
                if (command === 'setTriggerLevel') {
                    this.triggerLevel = value;
                    // TODO: Implement trigger level functionality
                }
            });
        }
    }

    private initSelects(): void {
        for (let i = 0; i < this.oscilloscopeSelects.length; i++) {
            this.oscilloscopeSelects[i].addEventListener('change', (event) => {
                const value = parseFloat((<HTMLSelectElement>event.currentTarget).value);
                const command = (<HTMLSelectElement>event.currentTarget).dataset.command;
                
                if (command === 'setTimePerDiv') {
                    this.timePerDiv = value;
                    this.updateTimeScale();
                } else if (command === 'setVoltagePerDiv') {
                    this.voltagePerDiv = value;
                    this.updateVoltageScale();
                }
            });
        }
    }

    private initButtons(): void {
        this.runStopBtn.addEventListener('click', () => {
            this.toggleRunStop();
        });

        this.singleBtn.addEventListener('click', () => {
            this.singleTrigger();
        });

        this.autoScaleBtn.addEventListener('click', () => {
            this.autoScale();
        });
    }

    private toggleRunStop(): void {
        if (this.isRunning) {
            this.stopAcquisition();
        } else {
            this.startContinuous();
        }
    }

    private singleTrigger(): void {
        this.stopAcquisition(); // Stop continuous if running
        // Trigger acquisition through the plot (which handles capture and display)
        if (this.oscilloscopePlot) {
            this.oscilloscopePlot.triggerAcquisition();
        }
    }

    private startContinuous(): void {
        this.isRunning = true;
        this.runStopBtn.textContent = 'Stop';
        this.runStopBtn.className = 'btn btn-danger btn-sm';
        this.acquisitionStateDisplay.textContent = 'Running';
        
        // Start continuous acquisition loop
        this.continuousAcquisition();
    }

    private stopAcquisition(): void {
        this.isRunning = false;
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
            this.animationId = 0;
        }
        
        this.runStopBtn.textContent = 'Run';
        this.runStopBtn.className = 'btn btn-success btn-sm';
        this.acquisitionStateDisplay.textContent = 'Stopped';
    }

    private continuousAcquisition(): void {
        if (this.isRunning) {
            // Trigger acquisition through the plot
            if (this.oscilloscopePlot) {
                this.oscilloscopePlot.triggerAcquisition();
            }
            this.animationId = requestAnimationFrame(() => {
                setTimeout(() => {
                    this.continuousAcquisition();
                }, 200); // Slower rate for continuous acquisition (5 FPS)
            });
        }
    }

    private updateTimeScale(): void {
        // Notify the plot about time scale change
        // The plot will use this.timePerDiv to calculate display range
    }

    private updateVoltageScale(): void {
        // Notify the plot about voltage scale change
        // The plot will use this.voltagePerDiv to calculate display range
    }

    private autoScale(): void {
        // TODO: Implement auto-scaling functionality
        // This should analyze the signal and set appropriate time/voltage scales
    }

    private updateStatus(): void {
        this.oscilloscope.getOscilloscopeParameters((status) => {
            this.samplingRateDisplay.textContent = (status.sampling_rate / 1e6).toFixed(1);
            this.bufferSizeDisplay.textContent = status.buffer_size.toString();
            
            // Calculate and display time span
            const timeSpanUs = (status.buffer_size / status.sampling_rate) * 1e6;
            this.timeSpanDisplay.textContent = timeSpanUs.toFixed(1);
        });
        
        // Update status every 2 seconds
        setTimeout(() => this.updateStatus(), 2000);
    }

    // Getter methods for the plot to access current settings
    getTimePerDiv(): number {
        return this.timePerDiv;
    }

    getVoltagePerDiv(): number {
        return this.voltagePerDiv;
    }

    getTriggerLevel(): number {
        return this.triggerLevel;
    }

    // Set reference to the plot for triggering acquisitions
    setPlotReference(plot: any): void {
        this.oscilloscopePlot = plot;
    }
} 