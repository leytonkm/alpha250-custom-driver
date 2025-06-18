// Oscilloscope app widget
// (c) Koheron

class OscilloscopeApp {
    private oscilloscopeInputs: HTMLInputElement[];
    private triggerSingleBtn: HTMLButtonElement;
    private triggerContinuousBtn: HTMLButtonElement;
    private triggerStopBtn: HTMLButtonElement;
    private samplingRateDisplay: HTMLSpanElement;
    private bufferSizeDisplay: HTMLSpanElement;
    
    private continuousMode: boolean = false;
    private animationId: number = 0;

    constructor(document: Document, private oscilloscope: any) {
        this.oscilloscopeInputs = <HTMLInputElement[]><any>document.getElementsByClassName("oscilloscope-input");
        this.triggerSingleBtn = <HTMLButtonElement>document.getElementById('trigger-single');
        this.triggerContinuousBtn = <HTMLButtonElement>document.getElementById('trigger-continuous');
        this.triggerStopBtn = <HTMLButtonElement>document.getElementById('trigger-stop');
        this.samplingRateDisplay = <HTMLSpanElement>document.getElementById('sampling-rate-display');
        this.bufferSizeDisplay = <HTMLSpanElement>document.getElementById('buffer-size-display');
        
        this.initInputs();
        this.initButtons();
        this.updateStatus();
    }

    private initInputs(): void {
        for (let i = 0; i < this.oscilloscopeInputs.length; i++) {
            this.oscilloscopeInputs[i].addEventListener('change', (event) => {
                const value = parseFloat((<HTMLInputElement>event.currentTarget).value);
                const command = (<HTMLInputElement>event.currentTarget).dataset.command;
                
                if (command === 'setTimeRange') {
                    this.oscilloscope.setTimeRange(value);
                    // Update status to reflect new time range
                    this.oscilloscope.getOscilloscopeParameters((status) => {
                        // Status updated, plot will use new time range
                    });
                }
            });
        }
    }

    private initButtons(): void {
        this.triggerSingleBtn.addEventListener('click', () => {
            this.singleTrigger();
        });

        this.triggerContinuousBtn.addEventListener('click', () => {
            this.startContinuous();
        });

        this.triggerStopBtn.addEventListener('click', () => {
            this.stopAcquisition();
        });
    }

    private singleTrigger(): void {
        this.oscilloscope.triggerAdc();
    }

    private startContinuous(): void {
        this.continuousMode = true;
        this.triggerContinuousBtn.disabled = true;
        this.triggerSingleBtn.disabled = true;
        this.triggerStopBtn.disabled = false;
        
        // Start continuous acquisition loop
        this.continuousAcquisition();
    }

    private stopAcquisition(): void {
        this.continuousMode = false;
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
            this.animationId = 0;
        }
        
        this.triggerContinuousBtn.disabled = false;
        this.triggerSingleBtn.disabled = false;
        this.triggerStopBtn.disabled = true;
    }

    private continuousAcquisition(): void {
        if (this.continuousMode) {
            this.oscilloscope.triggerAdc();
            this.animationId = requestAnimationFrame(() => {
                setTimeout(() => {
                    this.continuousAcquisition();
                }, 50); // 20 FPS update rate
            });
        }
    }

    private updateStatus(): void {
        this.oscilloscope.getOscilloscopeParameters((status) => {
            this.samplingRateDisplay.textContent = (status.sampling_rate / 1e6).toFixed(1);
            this.bufferSizeDisplay.textContent = status.buffer_size.toString();
        });
        
        // Update status every 2 seconds
        setTimeout(() => this.updateStatus(), 2000);
    }
} 