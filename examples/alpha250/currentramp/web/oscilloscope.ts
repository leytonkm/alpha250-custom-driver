// ADC Oscilloscope driver
// (c) Koheron

interface IOscilloscopeStatus {
    sampling_rate: number;  // ADC sampling rate (Hz)
    buffer_size: number;    // ADC buffer size
    time_range: number;     // Time range (ms)
}

class Oscilloscope {
    private driver: Driver;
    private id: number;
    private cmds: Commands;

    public buffer_size: number;
    public status: IOscilloscopeStatus;

    constructor(private client: Client) {
        // Don't access driver here - wait for init()
        this.status = <IOscilloscopeStatus>{};
        this.buffer_size = 0;  // Will be set by getBufferSize from backend
    }

    init(cb: () => void): void {
        // Now access the driver after client is initialized
        this.driver = this.client.getDriver('CurrentRamp');
        this.id = this.driver.id;
        this.cmds = this.driver.getCmds();
        
        // Debug: Log available commands
        console.log('Available CurrentRamp commands:', Object.keys(this.cmds));
        
        this.getBufferSize((size: number) => {
            this.buffer_size = size;
            this.getOscilloscopeParameters(() => {
                cb();
            });
        });
    }

    getBufferSize(cb: (size: number) => void): void {
        console.log('Calling get_adc_buffer_size...');
        this.client.readUint32(Command(this.id, this.cmds['get_adc_buffer_size']),
                              (size) => {
                                  console.log('Buffer size received:', size);
                                  cb(size);
                              });
    }

    readAdcDataVolts(cb: (data: Float32Array) => void): void {
        console.log('Calling get_adc_data_volts...');
        this.client.readFloat32Array(Command(this.id, this.cmds['get_adc_data_volts']), (data: Float32Array) => {
            console.log('ADC data received, length:', data.length);
            cb(data);
        });
    }

    readAdcDataRaw(cb: (data: Int16Array) => void): void {
        this.client.readInt32Array(Command(this.id, this.cmds['get_adc_data']), (data: Int32Array) => {
            // Convert Int32Array to Int16Array (the server sends data as 32-bit but it's really 16-bit data)
            const int16Data = new Int16Array(data.length * 2);
            for (let i = 0; i < data.length; i++) {
                int16Data[i * 2] = data[i] & 0xFFFF;
                int16Data[i * 2 + 1] = (data[i] >> 16) & 0xFFFF;
            }
            cb(int16Data);
        });
    }

    setTimeRange(time_range_ms: number): void {
        console.log('Setting time range to:', time_range_ms);
        this.client.send(Command(this.id, this.cmds['set_time_range'], time_range_ms));
    }

    getOscilloscopeParameters(cb: (status: IOscilloscopeStatus) => void): void {
        console.log('Getting oscilloscope parameters...');
        this.client.readFloat64(Command(this.id, this.cmds['get_adc_sampling_rate']), (sampling_rate: number) => {
            console.log('Sampling rate received:', sampling_rate);
            this.status.sampling_rate = sampling_rate;
            this.status.buffer_size = this.buffer_size;
            
            this.client.readFloat64(Command(this.id, this.cmds['get_time_range']), (time_range: number) => {
                console.log('Time range received:', time_range);
                this.status.time_range = time_range;
                cb(this.status);
            });
        });
    }

    triggerAdc(): void {
        this.client.send(Command(this.id, this.cmds['trigger_adc_acquisition']));
    }
} 