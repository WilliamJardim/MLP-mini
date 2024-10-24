type HistoryLog = {
    message: string,
    aparence: string,
    classes: string[],
    timestamp: number,
    date:Date
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

    public updateString(): void{
        let currentHistory = this.getHistory();
        this.lines = '';

        //Para cada console vinculado
        for( let i = 0 ; i < currentHistory.length ; i++ )
        {
            this.lines += currentHistory[i].message + '\n';
        }
    }

    /**
    * Integra o conteudo de outros ConsoleMonitor(es) a esse
    */
    public integrate( from:ConsoleMonitor[] ): void{
        this.isIntegrator = true;

        //Para cada console vinculado
        for( let i = 0 ; i < from.length ; i++ )
        {
            //Extrai as informações e acrescenta elas na lista
            let currentLogs:HistoryLog[] = from[ i ].getHistory();
            let consoleName:string       = from[ i ].getConsoleName();

            currentLogs.forEach(( info:HistoryLog )=>{
                this.push( {...info} );
            });
        }   

        this.updateString();
    }

    public log( message:string, aparence:string='white', classes:string[]=[] ): void{
        console.log(message);

        this.lines = this.lines + message + '\n';

        this.history.push({
            aparence: aparence,
            message:  message,
            classes:  classes,
            timestamp: new Date().getTime(),
            date: new Date()
        });
    }

    public reset(){
        this.lines = '';
        this.history = [];
    }
}