class App {

    private imports: Imports;
    private voltageControl: VoltageControl;
    private voltageControlApp: VoltageControlApp;
    private oscilloscope: Oscilloscope;
    private oscilloscopePlot: OscilloscopePlot;
    private oscilloscopeApp: OscilloscopeApp;

    constructor(window: Window, document: Document, ip: string) {
        let sockpoolSize: number = 5;
        let client = new Client(ip, sockpoolSize);

        window.addEventListener('HTMLImportsLoaded', () => {
            console.log('HTMLImportsLoaded event fired');
            this.imports = new Imports(document);
            console.log('Imports created');
            
            client.init( () => {
                console.log('Client initialized');
                this.voltageControl = new VoltageControl(client);
                this.voltageControlApp = new VoltageControlApp(document, this.voltageControl);
                
                // Initialize oscilloscope (with error handling)
                try {
                    console.log('Creating Oscilloscope...');
                    this.oscilloscope = new Oscilloscope(client);
                    console.log('Oscilloscope created successfully');
                    
                    // Initialize the oscilloscope driver
                    this.oscilloscope.init(() => {
                        console.log('Oscilloscope initialized successfully');
                        
                        console.log('Creating OscilloscopeApp...');
                        this.oscilloscopeApp = new OscilloscopeApp(document, this.oscilloscope);
                        console.log('OscilloscopeApp created successfully');
                        
                        console.log('Creating OscilloscopePlot...');
                        this.oscilloscopePlot = new OscilloscopePlot(this.oscilloscope, this.oscilloscopeApp);
                        console.log('OscilloscopePlot created successfully');
                        
                        // Connect the app and plot
                        this.oscilloscopeApp.setPlotReference(this.oscilloscopePlot);
                        console.log('Connected app and plot');
                    });
                } catch (error) {
                    console.error('Error initializing oscilloscope:', error);
                }
            });
        }, false);

        window.onbeforeunload = () => { client.exit(); };
    }
}

let app = new App(window, document, location.hostname); 