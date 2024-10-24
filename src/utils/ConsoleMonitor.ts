type HistoryLog = {
    message: string,
    aparence: string,
    classes: string[]
}

type ConsoleMonitorConfig = {
    name:string
}

export default class ConsoleMonitor{
    private lines:string;
    private history:HistoryLog[];
    private isIntegrator:boolean;
    private config:ConsoleMonitorConfig;
    private name:string;

    constructor( config:ConsoleMonitorConfig ){
        this.lines = '';
        this.history = [];
        this.config = config;
        this.name = config.name;
    }

    public getConsoleName(): string{
        return this.name;
    }

    public asString(): string{
        return this.lines;
    }

    public getHistory(): HistoryLog[]{
        return this.history;
    }

    public push( info:HistoryLog): void{
        this.getHistory().push( info );
    }

    public integrate( from:ConsoleMonitor[] ): void{
        this.isIntegrator = true;

        //Para cada console vinculado
        for( let i = 0 ; i < from.length ; i++ )
        {
            //Extrai as informações e acrescenta elas na lista
            let currentLogs:HistoryLog[] = from[ i ].getHistory();
            let consoleName:string       = from[ i ].getConsoleName();

            //this.log(`CONSOLE: ${consoleName}`);

            currentLogs.forEach(( info:HistoryLog )=>{
                this.push( {...info} );
            });
        }   
    }

    public log( message:string, aparence:string='white', classes:string[]=[] ): void{
        console.log(message);

        this.lines = this.lines + message + '\n';

        this.history.push({
            aparence: aparence,
            message:  message,
            classes:  classes
        });
    }

    public reset(){
        this.lines = '';
        this.history = [];
    }
}