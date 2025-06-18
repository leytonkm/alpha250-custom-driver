// Simple Plot widget 
// (c) Koheron

class SimplePlot {
    public x_min: number = 0;
    public x_max: number = 1;
    public y_min: number = -0.6;
    public y_max: number = 0.6;
    private plot_placeholder: JQuery;
    private plot: any;
    
    constructor(placeholder_id: string) {
        this.plot_placeholder = $(placeholder_id);
    }

    setRangeX(x_min: number, x_max: number): void {
        this.x_min = x_min;
        this.x_max = x_max;
    }

    setRangeY(y_min: number, y_max: number): void {
        this.y_min = y_min;
        this.y_max = y_max;
    }

    redraw(plot_data: number[][], ylabel: string, callback: () => void): void {
        const plt_data = [{label: ylabel, data: plot_data}];

        const options = {
            series: {
                lines: { show: true },
                points: { show: false }
            },
            grid: {
                hoverable: true,
                clickable: true
            },
            xaxis: {
                min: this.x_min,
                max: this.x_max
            },
            yaxis: {
                min: this.y_min,
                max: this.y_max
            }
        };

        // Use jQuery Flot to create the plot
        this.plot = (<any>$).plot(this.plot_placeholder, plt_data, options);
        
        callback();
    }
} 